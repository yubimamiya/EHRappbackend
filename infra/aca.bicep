param name string
param location string = resourceGroup().location
param tags object = {}

param identityName string
// YUBI: add imageName parameter for DevOps deployment
param imageName string
param containerAppsEnvironmentName string
param containerRegistryName string

// YUBI: change service name to aca-back from aca to prevent conflicts with other services
// param serviceName string = 'aca'
param serviceName string = 'aca-back'

param exists bool
param openAiDeploymentName string = 'gpt-4o'
param openAiEndpoint string = 'https://2wccj467aelpw-cog.openai.azure.com/'
param openAiApiVersion string = '2025-01-01-preview'
@secure()
// this should stay empty
param openAiKey string = ''

resource acaIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

var env = [
  {
    name: 'OPENAI_HOST'
    value: 'azure'
  }
  {
    name: 'OPENAI_MODEL'
    // YUBI: how sure are we that the value for OPENAI_MODEL is the openAIDeploymentName?
    value: openAiDeploymentName
  }
  {
    name: 'AZURE_OPENAI_ENDPOINT'
    value: openAiEndpoint
  }
  {
    name: 'AZURE_OPENAI_API_VERSION'
    value: openAiApiVersion
  }
  {
    name: 'RUNNING_IN_PRODUCTION'
    value: 'true'
  }
  {
    // ManagedIdentityCredential will be passed this environment variable:
    name: 'AZURE_CLIENT_ID'
    value: acaIdentity.properties.clientId
  }
]

// YUBI: I'm a little confused about these keys
var envWithSecret = !empty(openAiKey) ? union(env, [
  {
    name: 'AZURE_OPENAI_KEY_FOR_CHATVISION'
    secretRef: 'azure-openai-key'
  }
]) : env

var secrets = !empty(openAiKey) ? {
  'azure-openai-key': openAiKey
} : {}

module app 'core/host/container-app-upsert.bicep' = {
  name: '${serviceName}-container-app-module'
  params: {
    name: name
    location: location
    tags: union(tags, { 'azd-service-name': serviceName })
    identityName: acaIdentity.name
    // YUBI: pass imageName for DevOps deployment
    imageName: imageName
    exists: exists
    containerAppsEnvironmentName: containerAppsEnvironmentName
    containerRegistryName: containerRegistryName
    env: envWithSecret
    secrets: secrets
    targetPort: 50505
  }
}

output SERVICE_ACA_IDENTITY_PRINCIPAL_ID string = acaIdentity.properties.principalId
output SERVICE_ACA_NAME string = app.outputs.name
output SERVICE_ACA_URI string = app.outputs.uri
output SERVICE_ACA_IMAGE_NAME string = app.outputs.imageName
