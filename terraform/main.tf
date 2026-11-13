terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 5.30.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

locals {
  required_services = toset([
    "aiplatform.googleapis.com",
    "cloudtrace.googleapis.com",
    "dlp.googleapis.com",
    "firestore.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
  ])
}

resource "google_project_service" "agent_services" {
  for_each           = local.required_services
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

resource "google_firestore_database" "default" {
  project     = var.project_id
  name        = "(default)"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"

  depends_on = [google_project_service.agent_services]
}

resource "google_storage_bucket" "medical_exports" {
  name                        = var.medical_export_bucket_name
  project                     = var.project_id
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  labels = {
    environment = var.environment
    purpose     = "agentic-journal-test"
  }

  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.agent_services]
}

resource "google_secret_manager_secret" "export_bucket_secret" {
  project   = var.project_id
  secret_id = "journal-agent-export-bucket"

  labels = {
    environment = var.environment
    purpose     = "agentic-journal-test"
  }

  replication {
    auto {}
  }

  depends_on = [google_project_service.agent_services]
}

resource "google_secret_manager_secret_version" "export_bucket_secret_v1" {
  secret      = google_secret_manager_secret.export_bucket_secret.id
  secret_data = google_storage_bucket.medical_exports.name
}
