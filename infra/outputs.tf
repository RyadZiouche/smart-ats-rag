output "api_base_url" {
  description = "URL API Gateway — à coller dans API_BASE_URL du .env"
  value       = aws_apigatewayv2_stage.prod.invoke_url
}

output "s3_bucket_name" {
  description = "Nom du bucket S3 créé"
  value       = aws_s3_bucket.audio.bucket
}

output "dynamodb_table_name" {
  description = "Nom de la table DynamoDB créée"
  value       = aws_dynamodb_table.feedback_store.name
}

output "lambda_ingestion_arn" {
  value = aws_lambda_function.ingestion.arn
}

output "lambda_search_arn" {
  value = aws_lambda_function.search.arn
}

output "lambda_reindexation_arn" {
  value = aws_lambda_function.reindexation.arn
}

output "cloudwatch_alarm_name" {
  value = aws_cloudwatch_metric_alarm.high_negative_feedback.alarm_name
}

output "next_steps" {
  description = "Instructions post-deploy"
  value       = <<-EOT
    ✅ Infrastructure déployée. Étapes suivantes :

    1. Copiez dans votre .env :
       API_BASE_URL=${aws_apigatewayv2_stage.prod.invoke_url}

    2. Confirmez l'abonnement SNS (email envoyé à ${var.alert_email})

    3. Vérifiez l'environnement :
       python infra/check_env.py

    4. Lancez le frontend :
       cd src/frontend && streamlit run app.py
  EOT
}