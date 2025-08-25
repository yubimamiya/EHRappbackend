metadata description = 'Creates an Azure Cognitive Services instance.'
param name string
param location string = resourceGroup().location
param tags object = {}
@description('The custom subdomain name used to access the API. Defaults to the value of the name parameter.')
param customSubDomainName string = name
param disableLocalAuth bool = false
param deployments array = []
param kind string = 'OpenAI'

@allowed([ 'Enabled', 'Disabled' ])
param publicNetworkAccess string = 'Enabled'
param sku object = {
  name: 'S0'
}

param allowedIpRules array = []
param networkAcls object = empty(allowedIpRules) ? {
  defaultAction: 'Allow'
} : {
  ipRules: allowedIpRules
  defaultAction: 'Deny'
}

resource account 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: name
  location: location
  tags: tags
  kind: kind
  properties: {
    customSubDomainName: customSubDomainName
    publicNetworkAccess: publicNetworkAccess
    networkAcls: networkAcls
    disableLocalAuth: disableLocalAuth
  }
  sku: sku
}

@batchSize(1)
resource deployment 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = [for deployment in deployments: {
  parent: account
  name: deployment.name
  properties: {
    model: deployment.model
    // Try using safe access operator instead, but I'm getting so many errors
    // to use, it says I need to use-safe-access in bicepconfig.json, but I don't see that file, so I switched back tp revious
    // It's only a warning so it shouldn't cause a big error

    // fix with safe access operator
    raiPolicyName: contains(deployment, 'raiPolicyName') ? deployment.raiPolicyName : null
    // raiPolicyName: contains(deployment, 'raiPolicyName') ? deployment?.raiPolicyName : null

    // raiPolicyName: deployment.?'raiPolicyName' ?? null
  }
  // try using safe access operator instead
  // sku: deployment.?'sku' ?? {

  // fix with safe access operator
  // sku: contains(deployment, 'sku') ? deployment?.sku : {
    
  sku: contains(deployment, 'sku') ? deployment.sku : {
    name: 'Standard'
    capacity: 20
  }
}]

output endpoint string = account.properties.endpoint
output endpoints object = account.properties.endpoints
output id string = account.id
output name string = account.name
