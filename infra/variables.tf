variable "aws_region" {
  description = "Région AWS où déployer l'infrastructure"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Nom du projet, utilisé comme préfixe pour toutes les ressources"
  type        = string
  default     = "smart-meeting-rag"
}

variable "environment" {
  description = "Environnement de déploiement"
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "L'environnement doit être : dev, staging ou prod."
  }
}

variable "s3_bucket_name" {
  description = "Nom du bucket S3 — doit être globalement unique sur AWS"
  type        = string
}

variable "dynamodb_table_name" {
  description = "Nom de la table DynamoDB pour stocker les feedbacks"
  type        = string
  default     = "FeedbackStore"
}

variable "pinecone_api_key" {
  description = "Clé API Pinecone (sensible — ne jamais commiter)"
  type        = string
  sensitive   = true
}

variable "pinecone_index_name" {
  description = "Nom de l'index Pinecone (1536 dimensions, métrique cosine)"
  type        = string
  default     = "smart-meeting-rag"
}

variable "alert_email" {
  description = "Adresse email pour les alertes CloudWatch via SNS"
  type        = string
}