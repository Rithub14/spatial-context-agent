locals {
  tags = {
    environment = var.environment
    project     = "spatial-context-agent"
  }
  # Unique suffix so globally-scoped names (ACR, storage) don't clash across deployments
  suffix = lower(replace(var.resource_group_name, "-", ""))
}

# ---------------------------------------------------------------------------
# Resource Group
# ---------------------------------------------------------------------------

resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  tags     = local.tags
}

# ---------------------------------------------------------------------------
# Azure Container Registry (Basic SKU — cheapest tier)
# ---------------------------------------------------------------------------

resource "azurerm_container_registry" "acr" {
  name                = "${local.suffix}acr"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = local.tags
}

# ---------------------------------------------------------------------------
# PostgreSQL Flexible Server (Burstable B1ms — free-tier eligible)
# ---------------------------------------------------------------------------

resource "azurerm_postgresql_flexible_server" "db" {
  name                   = "${local.suffix}-psql"
  resource_group_name    = azurerm_resource_group.main.name
  location               = azurerm_resource_group.main.location
  version                = "16"
  administrator_login    = var.db_admin_username
  administrator_password = var.db_admin_password
  storage_mb             = 32768
  sku_name               = "B_Standard_B1ms"

  # Disable high-availability for dev (costs extra)
  high_availability {
    mode = "Disabled"
  }

  tags = local.tags
}

resource "azurerm_postgresql_flexible_server_database" "spatial_agent" {
  name      = "spatial_agent"
  server_id = azurerm_postgresql_flexible_server.db.id
  charset   = "utf8"
  collation = "en_US.utf8"
}

# Allow the ACI container to reach PostgreSQL
resource "azurerm_postgresql_flexible_server_firewall_rule" "allow_azure_services" {
  name      = "allow-azure-services"
  server_id = azurerm_postgresql_flexible_server.db.id
  # Azure magic IP range that allows Azure services through
  start_ip_address = "0.0.0.0"
  end_ip_address   = "0.0.0.0"
}

# ---------------------------------------------------------------------------
# Azure Container Instance — FastAPI app (1 vCPU, 1.5 GB RAM)
# ---------------------------------------------------------------------------

resource "azurerm_container_group" "api" {
  name                = "${var.resource_group_name}-api"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  ip_address_type     = "Public"
  dns_name_label      = "${local.suffix}-api"
  os_type             = "Linux"
  restart_policy      = "Always"

  # Pull from ACR using admin credentials
  image_registry_credential {
    server   = azurerm_container_registry.acr.login_server
    username = azurerm_container_registry.acr.admin_username
    password = azurerm_container_registry.acr.admin_password
  }

  container {
    name   = "spatial-agent"
    image  = var.container_image
    cpu    = "1.0"
    memory = "1.5"

    ports {
      port     = 8000
      protocol = "TCP"
    }

    environment_variables = {
      CLIP_MODEL_NAME    = "ViT-B/32"
      DEVICE             = "cpu"
      API_HOST           = "0.0.0.0"
      API_PORT           = "8000"
      ENABLE_AUTH        = "true"
      ENABLE_RATE_LIMIT  = "true"
      RATE_LIMIT_RPM     = "60"
    }

    secure_environment_variables = {
      DATABASE_URL = (
        "postgresql://${var.db_admin_username}:${var.db_admin_password}"
        "@${azurerm_postgresql_flexible_server.db.fqdn}:5432/spatial_agent"
      )
      API_KEY = "change-me-in-production"
    }
  }

  tags = local.tags
}

# ---------------------------------------------------------------------------
# Storage Account — required by Azure Function
# ---------------------------------------------------------------------------

resource "azurerm_storage_account" "func" {
  name                     = "${local.suffix}funcsa"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  tags                     = local.tags
}

# ---------------------------------------------------------------------------
# Azure Function — consumption plan (serverless, pay-per-use)
# ---------------------------------------------------------------------------

resource "azurerm_service_plan" "func" {
  name                = "${var.resource_group_name}-func-plan"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  os_type             = "Linux"
  sku_name            = "Y1" # Consumption plan
  tags                = local.tags
}

resource "azurerm_linux_function_app" "gateway" {
  name                       = "${local.suffix}-gateway"
  resource_group_name        = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  storage_account_name       = azurerm_storage_account.func.name
  storage_account_access_key = azurerm_storage_account.func.primary_access_key
  service_plan_id            = azurerm_service_plan.func.id

  app_settings = {
    ACI_ENDPOINT                     = "http://${azurerm_container_group.api.fqdn}:8000"
    FUNCTIONS_WORKER_RUNTIME         = "python"
    SCM_DO_BUILD_DURING_DEPLOYMENT   = "true"
  }

  site_config {
    application_stack {
      python_version = "3.11"
    }
  }

  tags = local.tags
}
