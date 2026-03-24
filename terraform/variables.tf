variable "resource_group_name" {
  description = "Azure resource group that holds all project resources."
  type        = string
  default     = "spatial-agent-rg"
}

variable "location" {
  description = "Azure region — westeurope is closest to Berlin."
  type        = string
  default     = "westeurope"
}

variable "environment" {
  description = "Deployment environment tag (dev / staging / prod)."
  type        = string
  default     = "dev"
}

variable "container_image" {
  description = "Full ACR image reference, e.g. spatialagentacr.azurecr.io/spatial-agent:latest"
  type        = string
  # No default — must be supplied at apply time:
  #   terraform apply -var="container_image=<acr>.azurecr.io/spatial-agent:<sha>"
}

variable "db_admin_username" {
  description = "PostgreSQL administrator login."
  type        = string
  default     = "spatialadmin"
}

variable "db_admin_password" {
  description = "PostgreSQL administrator password. Keep secret — pass via TF_VAR_db_admin_password."
  type        = string
  sensitive   = true
}
