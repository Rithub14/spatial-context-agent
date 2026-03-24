output "api_url" {
  description = "Public URL of the FastAPI container on Azure Container Instances."
  value       = "http://${azurerm_container_group.api.fqdn}:8000"
}

output "function_url" {
  description = "Base URL of the Azure Function gateway."
  value       = "https://${azurerm_linux_function_app.gateway.default_hostname}"
}

output "acr_login_server" {
  description = "ACR login server — use this when tagging and pushing Docker images."
  value       = azurerm_container_registry.acr.login_server
}

output "db_connection_string" {
  description = "PostgreSQL connection string for the app. Treat as a secret."
  sensitive   = true
  value = (
    "postgresql://${var.db_admin_username}:${var.db_admin_password}"
    "@${azurerm_postgresql_flexible_server.db.fqdn}:5432/spatial_agent"
  )
}
