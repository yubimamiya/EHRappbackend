targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name which is used to generate a short unique hash for each resource')
// YUBI: hard code name to be the prefix of the resource group
param name string = 'rec-ex-app'

@minLength(1)
@description('Primary location for all resources')
param location string = 'eastus'

@description('Initialize the imageName to keep track of deployments')
param imageName string = ''

@description('Id of the user or app to assign application roles')
param principalId string = ''

@description('Flag to decide where to create OpenAI role for current user')
param createRoleForUser bool = true

@minLength(1)
@description('Location for the OpenAI resource')
// Look for desired models on the availability table:
// https://learn.microsoft.com/azure/ai-services/openai/concepts/models#global-standard-model-availability
@allowed([
  'australiaeast'
  'brazilsouth'
  'canadaeast'
  'eastus'
  'eastus2'
  'francecentral'
  'germanywestcentral'
  'japaneast'
  'koreacentral'
  'northcentralus'
  'norwayeast'
  'polandcentral'
  'spaincentral'
  'southafricanorth'
  'southcentralus'
  'southindia'
  'swedencentral'
  'switzerlandnorth'
  'uksouth'
  'westeurope'
  'westus'
  'westus3'
])
@metadata({
  azd: {
    type: 'location'
  }
})
param openAILocation string = 'eastus'

// These parameters can be customized via azd env variables referenced in main.parameters.json:
param openAiResourceName string = ''
// YUBI: hard-coded resource group name
// EDIT: change when the resource group changes
param openAiResourceGroupName string = 'rec-ex-app-rg'
param openAiApiVersion string = '2025-01-01-preview'
// we do not need key based authentication because it is disabled for this resource group
param disableKeyBasedAuth bool = true
// setting parameters to values in main.parameters.json:
param openAiSkuName string='S0'
param openAiModelName string='gpt-4o'
param openAiModelVersion string='2024-05-13'
param openAiDeploymentName string='gpt-4o'
param openAiDeploymentCapacity int=50
param openAiDeploymentSkuName string='GlobalStandard'

// YUBI: set the flag to FALSE so I can use existing deployment
@description('Flag to decide whether to create Azure OpenAI instance or not')
// param createAzureOpenAi bool = true
param createAzureOpenAi bool = false

// YUBI: using key authentication
@description('Azure OpenAI key to use for authentication. If not provided, managed identity will be used (and is preferred)')
@secure()
// we want this to be empty
param openAiKey string = ''

// YUBI: set OpenAI endpoint to rec-ex-app-rg
@description('Azure OpenAI endpoint to use. If provided, no Azure OpenAI instance will be created.')
// param openAiEndpoint string = ''
param openAiEndpoint string = 'https://2wccj467aelpw-cog.openai.azure.com/'

param acaExists bool = false

// change resourceToken so that it points to correct resource group
var resourceToken = toLower(uniqueString(subscription().id, name, location))
var tags = { 'azd-env-name': name }

resource resourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: '${name}-rg'
  location: location
  tags: tags
}

resource openAiResourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' existing = if (!empty(openAiResourceGroupName)) {
  name: !empty(openAiResourceGroupName) ? openAiResourceGroupName : resourceGroup.name
}

var prefix = '${name}-${resourceToken}'

module openAi 'core/ai/cognitiveservices.bicep' = if (createAzureOpenAi) {
  name: 'openai'
  scope: openAiResourceGroup
  params: {
    name: !empty(openAiResourceName) ? openAiResourceName : '${resourceToken}-cog'
    location: openAILocation
    tags: tags
    disableLocalAuth: disableKeyBasedAuth
    sku: {
      name: openAiSkuName
    }
    deployments: [
      {
        name: openAiDeploymentName
        model: {
          format: 'OpenAI'
          name: openAiModelName
          version: openAiModelVersion
        }
        sku: {
          name: openAiDeploymentSkuName
          capacity: openAiDeploymentCapacity
        }
      }
    ]
  }
}

module logAnalyticsWorkspace 'core/monitor/loganalytics.bicep' = {
  name: 'loganalytics'
  scope: resourceGroup
  params: {
    name: '${prefix}-loganalytics'
    location: location
    tags: tags
  }
}

// Container apps host (including container registry): all can be shared beween multiple container apps
module containerApps 'core/host/container-apps.bicep' = {
  name: 'container-apps'
  scope: resourceGroup
  params: {
    name: 'app'
    location: location
    tags: tags
    containerAppsEnvironmentName: '${prefix}-containerapps-env'
    containerRegistryName: '${replace(prefix, '-', '')}registry'
    logAnalyticsWorkspaceName: logAnalyticsWorkspace.outputs.name
  }
}

// Container app frontend
module aca 'aca.bicep' = {
  name: 'aca'
  scope: resourceGroup
  params: {
    // YUBI: update name of the container app to ca-back
    name: replace('${take(prefix,19)}-ca-back', '--', '-')
    location: location
    tags: tags
    identityName: '${prefix}-id-aca'
    // YUBI: pass imageName to aca.bicep file for DevOps deployment
    imageName: imageName
    containerAppsEnvironmentName: containerApps.outputs.environmentName
    containerRegistryName: containerApps.outputs.registryName
    openAiDeploymentName: openAiDeploymentName
    // fix with safe access operator
    openAiEndpoint: createAzureOpenAi ? openAi.outputs.endpoint : openAiEndpoint
    // openAiEndpoint: createAzureOpenAi ? openAi.outputs?.endpoint : openAiEndpoint
    openAiApiVersion: openAiApiVersion
    // do I need this key? it is empty right now
    openAiKey: openAiKey
    exists: acaExists
  }
}


module openAiRoleUser 'core/security/role.bicep' = if (createRoleForUser && createAzureOpenAi) {
  scope: openAiResourceGroup
  name: 'openai-role-user'
  params: {
    principalId: principalId
    roleDefinitionId: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
    principalType: 'User'
  }
}


module openAiRoleBackend 'core/security/role.bicep' = if (createAzureOpenAi) {
  scope: openAiResourceGroup
  name: 'openai-role-backend'
  params: {
    principalId: aca.outputs.SERVICE_ACA_IDENTITY_PRINCIPAL_ID
    roleDefinitionId: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
    principalType: 'ServicePrincipal'
  }
}

output AZURE_LOCATION string = location
output AZURE_RESOURCE_GROUP string = resourceGroup.name
output AZURE_TENANT_ID string = tenant().tenantId

// output AZURE_OPENAI_RESOURCE_NAME string = openAi.outputs.name
// fix with safe access operator
// output AZURE_OPENAI_RESOURCE_NAME string = openAi.outputs?.name
output AZURE_OPENAI_RESOURCE_NAME string = openAi.outputs.name


output AZURE_OPENAI_DEPLOYMENT string = openAiDeploymentName
output AZURE_OPENAI_API_VERSION string = openAiApiVersion
// fix with safe access operator
output AZURE_OPENAI_ENDPOINT string = createAzureOpenAi ? openAi.outputs.endpoint : openAiEndpoint
// output AZURE_OPENAI_ENDPOINT string = createAzureOpenAi ? openAi.outputs?.endpoint : openAiEndpoint

output SERVICE_ACA_IDENTITY_PRINCIPAL_ID string = aca.outputs.SERVICE_ACA_IDENTITY_PRINCIPAL_ID
output SERVICE_ACA_NAME string = aca.outputs.SERVICE_ACA_NAME
output SERVICE_ACA_URI string = aca.outputs.SERVICE_ACA_URI
output SERVICE_ACA_IMAGE_NAME string = aca.outputs.SERVICE_ACA_IMAGE_NAME

output AZURE_CONTAINER_ENVIRONMENT_NAME string = containerApps.outputs.environmentName
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerApps.outputs.registryLoginServer
output AZURE_CONTAINER_REGISTRY_NAME string = containerApps.outputs.registryName
