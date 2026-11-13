variable "project_id" {
  description = "Google Cloud Project ID hosting the Daily Journaling Agent."
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region for Firestore and Cloud Storage."
  type        = string
  default     = "us-central1"
}

variable "medical_export_bucket_name" {
  description = "Globally unique GCS bucket name for clinician review report exports."
  type        = string
}

variable "environment" {
  description = "Deployment environment label (e.g., test, dev)."
  type        = string
  default     = "test"
}
