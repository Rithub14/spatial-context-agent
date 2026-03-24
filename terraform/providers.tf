terraform {
  required_version = ">= 1.5"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90"
    }
  }

  # Local backend for development.
  # To switch to Azure Storage for team/CI use:
  #
  # backend "azurerm" {
  #   resource_group_name  = "terraform-state-rg"
  #   storage_account_name = "tfstatespatialagent"
  #   container_name       = "tfstate"
  #   key                  = "spatial-agent.tfstate"
  # }
}

provider "azurerm" {
  features {}
}
