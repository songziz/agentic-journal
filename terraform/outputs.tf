output "firestore_database_name" {
  description = "Name of the provisioned Cloud Firestore database."
  value       = google_firestore_database.default.name
}

output "medical_exports_bucket_url" {
  description = "GCS URL of the bucket storing clinician review exports."
  value       = google_storage_bucket.medical_exports.url
}

output "export_bucket_secret_id" {
  description = "Secret Manager resource ID storing the medical export bucket name."
  value       = google_secret_manager_secret.export_bucket_secret.secret_id
}
