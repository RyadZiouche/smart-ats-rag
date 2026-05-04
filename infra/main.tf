# ─────────────────────────────────────────────────────────────────────────────
# smart-meeting-rag — Infrastructure as Code
# Terraform >= 1.5
#
# Ressources provisionnées :
#   - S3 : bucket audio + transcriptions + lifecycle rules
#   - DynamoDB : table FeedbackStore + 2 GSI + TTL
#   - IAM : rôles et policies pour chaque Lambda
#   - Lambda : 4 fonctions (ingestion, search, feedback, reindexation)
#   - API Gateway : routes POST /search et POST /feedback
#   - EventBridge : règle cron hebdomadaire → Lambda reindexation
#   - CloudWatch : log groups + dashboard
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. S3 — Stockage audio + transcriptions
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_s3_bucket" "audio" {
  bucket        = var.s3_bucket_name
  force_destroy = false # sécurité : empêche la suppression accidentelle en prod

  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "audio" {
  bucket = aws_s3_bucket.audio.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "audio" {
  bucket = aws_s3_bucket.audio.id

  rule {
    id     = "expire-audio-files"
    status = "Enabled"
    filter { prefix = "audio/" }
    expiration { days = 180 }
  }

  rule {
    id     = "expire-transcriptions"
    status = "Enabled"
    filter { prefix = "transcriptions/" }
    expiration { days = 365 }
  }
}

# Bloquer tout accès public (bonne pratique sécurité)
resource "aws_s3_bucket_public_access_block" "audio" {
  bucket                  = aws_s3_bucket.audio.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Notification S3 → Lambda ingestion (déclenchée à chaque upload audio)
resource "aws_s3_bucket_notification" "trigger_ingestion" {
  bucket = aws_s3_bucket.audio.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.ingestion.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "audio/"
    filter_suffix       = ".mp3"
  }

  # Attend que la permission Lambda soit créée avant
  depends_on = [aws_lambda_permission.s3_invoke_ingestion]
}

# Permission permettant à S3 d'invoquer la Lambda
resource "aws_lambda_permission" "s3_invoke_ingestion" {
  statement_id  = "AllowS3Invoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.ingestion.function_name
  principal     = "s3.amazonaws.com"
  source_arn    = aws_s3_bucket.audio.arn
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. DYNAMODB — Table FeedbackStore
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_dynamodb_table" "feedback_store" {
  name         = var.dynamodb_table_name
  billing_mode = "PAY_PER_REQUEST" # serverless, 0 coût au repos
  hash_key     = "query_id"
  range_key    = "timestamp"

  # Clé primaire
  attribute {
    name = "query_id"
    type = "S"
  }
  attribute {
    name = "timestamp"
    type = "S"
  }

  # Attributs pour les GSI
  attribute {
    name = "meeting_id"
    type = "S"
  }
  attribute {
    name = "score"
    type = "N"
  }

  # GSI 1 : récupérer tous les feedbacks d'une réunion
  # Usage : Lambda reindexation group-by meeting_id
  global_secondary_index {
    name            = "meeting_id-index"
    hash_key        = "meeting_id"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  # GSI 2 : filtrer par score (-1 négatif / 1 positif)
  # Usage : scan rapide des feedbacks négatifs à retraiter
  global_secondary_index {
    name            = "score-index"
    hash_key        = "score"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  # TTL : suppression automatique des feedbacks après 90 jours
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  tags = local.common_tags
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. IAM — Rôle partagé pour les Lambdas
# ─────────────────────────────────────────────────────────────────────────────

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_exec" {
  name               = "${var.project_name}-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
  tags               = local.common_tags
}

# Policy : accès à tous les services nécessaires
resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # CloudWatch Logs
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      # S3 : lecture/écriture sur le bucket audio
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.audio.arn,
          "${aws_s3_bucket.audio.arn}/*"
        ]
      },
      # Bedrock : embedding Titan + Claude Haiku
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = "*"
      },
      # AWS Transcribe
      {
        Effect = "Allow"
        Action = [
          "transcribe:StartTranscriptionJob",
          "transcribe:GetTranscriptionJob",
          "transcribe:ListTranscriptionJobs"
        ]
        Resource = "*"
      },
      # DynamoDB : opérations sur FeedbackStore
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:UpdateItem",
          "dynamodb:Scan",
          "dynamodb:Query"
        ]
        Resource = [
          aws_dynamodb_table.feedback_store.arn,
          "${aws_dynamodb_table.feedback_store.arn}/index/*"
        ]
      },
      # CloudWatch : métriques custom
      {
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
      }
    ]
  })
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. LAMBDA — 4 fonctions
# Packaging : chaque Lambda est zippée depuis son dossier src/lambdas/
# ─────────────────────────────────────────────────────────────────────────────

# Archive ZIP générées par Terraform au plan/apply
data "archive_file" "ingestion" {
  type        = "zip"
  source_dir  = "${path.module}/../src/lambdas/ingestion"
  output_path = "${path.module}/.builds/ingestion.zip"
}

data "archive_file" "search" {
  type        = "zip"
  source_dir  = "${path.module}/../src/lambdas/search"
  output_path = "${path.module}/.builds/search.zip"
}

data "archive_file" "feedback" {
  type        = "zip"
  source_dir  = "${path.module}/../src/lambdas/feedback"
  output_path = "${path.module}/.builds/feedback.zip"
}

data "archive_file" "reindexation" {
  type        = "zip"
  source_dir  = "${path.module}/../src/lambdas/reindexation"
  output_path = "${path.module}/.builds/reindexation.zip"
}

# Lambda Ingestion — déclenchée par S3
resource "aws_lambda_function" "ingestion" {
  function_name    = "${var.project_name}-ingestion"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "app.lambda_handler"
  runtime          = "python3.11"
  filename         = data.archive_file.ingestion.output_path
  source_code_hash = data.archive_file.ingestion.output_base64sha256
  timeout          = 600 # 10 min max (Transcribe peut être long)
  memory_size      = 512

  environment {
    variables = {
      PINECONE_API_KEY   = var.pinecone_api_key
      PINECONE_INDEX_NAME = var.pinecone_index_name
      S3_BUCKET_NAME     = var.s3_bucket_name
    }
  }

  tags = local.common_tags
}

# Lambda Search — déclenchée par API Gateway
resource "aws_lambda_function" "search" {
  function_name    = "${var.project_name}-search"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "app.lambda_handler"
  runtime          = "python3.11"
  filename         = data.archive_file.search.output_path
  source_code_hash = data.archive_file.search.output_base64sha256
  timeout          = 30
  memory_size      = 256

  environment {
    variables = {
      PINECONE_API_KEY    = var.pinecone_api_key
      PINECONE_INDEX_NAME = var.pinecone_index_name
    }
  }

  tags = local.common_tags
}

# Lambda Feedback — déclenchée par API Gateway
resource "aws_lambda_function" "feedback" {
  function_name    = "${var.project_name}-feedback"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "app.lambda_handler"
  runtime          = "python3.11"
  filename         = data.archive_file.feedback.output_path
  source_code_hash = data.archive_file.feedback.output_base64sha256
  timeout          = 10
  memory_size      = 128

  environment {
    variables = {
      DYNAMODB_TABLE = var.dynamodb_table_name
    }
  }

  tags = local.common_tags
}

# Lambda Reindexation — déclenchée par EventBridge (hebdomadaire)
resource "aws_lambda_function" "reindexation" {
  function_name    = "${var.project_name}-reindexation"
  role             = aws_iam_role.lambda_exec.arn
  handler          = "app.lambda_handler"
  runtime          = "python3.11"
  filename         = data.archive_file.reindexation.output_path
  source_code_hash = data.archive_file.reindexation.output_base64sha256
  timeout          = 900 # 15 min max (réindexation peut être longue)
  memory_size      = 512

  environment {
    variables = {
      PINECONE_API_KEY    = var.pinecone_api_key
      PINECONE_INDEX_NAME = var.pinecone_index_name
      DYNAMODB_TABLE      = var.dynamodb_table_name
      S3_BUCKET_NAME      = var.s3_bucket_name
    }
  }

  tags = local.common_tags
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. API GATEWAY — Routes /search et /feedback
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_apigatewayv2_api" "main" {
  name          = "${var.project_name}-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["Content-Type"]
  }

  tags = local.common_tags
}

resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "prod"
  auto_deploy = true
}

# Intégrations Lambda
resource "aws_apigatewayv2_integration" "search" {
  api_id             = aws_apigatewayv2_api.main.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.search.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_integration" "feedback" {
  api_id             = aws_apigatewayv2_api.main.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.feedback.invoke_arn
  payload_format_version = "2.0"
}

# Routes
resource "aws_apigatewayv2_route" "search" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "POST /search"
  target    = "integrations/${aws_apigatewayv2_integration.search.id}"
}

resource "aws_apigatewayv2_route" "feedback" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "POST /feedback"
  target    = "integrations/${aws_apigatewayv2_integration.feedback.id}"
}

# Permissions API Gateway → Lambda
resource "aws_lambda_permission" "apigw_search" {
  statement_id  = "AllowAPIGWSearch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.search.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}

resource "aws_lambda_permission" "apigw_feedback" {
  statement_id  = "AllowAPIGWFeedback"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.feedback.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. EVENTBRIDGE — Réindexation hebdomadaire (lundi 2h UTC)
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_cloudwatch_event_rule" "weekly_reindex" {
  name                = "${var.project_name}-weekly-reindex"
  description         = "Déclenche la Lambda reindexation chaque lundi à 2h UTC"
  schedule_expression = "cron(0 2 ? * MON *)"
  tags                = local.common_tags
}

resource "aws_cloudwatch_event_target" "reindex_lambda" {
  rule      = aws_cloudwatch_event_rule.weekly_reindex.name
  target_id = "ReindexLambdaTarget"
  arn       = aws_lambda_function.reindexation.arn
}

resource "aws_lambda_permission" "eventbridge_invoke_reindexation" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.reindexation.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.weekly_reindex.arn
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. CLOUDWATCH — Log groups + Alarme taux de feedback négatif
# ─────────────────────────────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "ingestion" {
  name              = "/aws/lambda/${aws_lambda_function.ingestion.function_name}"
  retention_in_days = 14
  tags              = local.common_tags
}

resource "aws_cloudwatch_log_group" "search" {
  name              = "/aws/lambda/${aws_lambda_function.search.function_name}"
  retention_in_days = 14
  tags              = local.common_tags
}

resource "aws_cloudwatch_log_group" "feedback" {
  name              = "/aws/lambda/${aws_lambda_function.feedback.function_name}"
  retention_in_days = 14
  tags              = local.common_tags
}

resource "aws_cloudwatch_log_group" "reindexation" {
  name              = "/aws/lambda/${aws_lambda_function.reindexation.function_name}"
  retention_in_days = 14
  tags              = local.common_tags
}

# Alarme : trop de feedbacks négatifs → SNS (email/Slack)
resource "aws_cloudwatch_metric_alarm" "high_negative_feedback" {
  alarm_name          = "${var.project_name}-high-negative-feedback"
  alarm_description   = "Taux de feedback négatif élevé — qualité RAG dégradée"
  namespace           = "SmartMeetingRAG"
  metric_name         = "NegativeFeedback"
  statistic           = "Sum"
  period              = 86400  # fenêtre de 24h
  evaluation_periods  = 1
  threshold           = 10     # alerte si >10 feedbacks négatifs en 24h
  comparison_operator = "GreaterThanThreshold"
  alarm_actions       = [aws_sns_topic.alerts.arn]
  tags                = local.common_tags
}

# SNS topic pour les alertes
resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
  tags = local.common_tags
}

resource "aws_sns_topic_subscription" "email_alert" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. LOCALS — Tags communs à toutes les ressources
# ─────────────────────────────────────────────────────────────────────────────

locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
