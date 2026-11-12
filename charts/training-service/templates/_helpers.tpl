{{/*
Expand the name of the chart.
*/}}
{{- define "training-service.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "training-service.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/* Cluster-wide RBAC must coexist with the same release in another namespace. */}}
{{- define "training-service.rbacName" -}}
{{- printf "%s-%s" (include "training-service.fullname" .) .Release.Namespace -}}
{{- end -}}

{{/* Use the same runtime image for catalog export and training, including legacy values. */}}
{{- define "training-service.runtimeImage" -}}
{{- $runtime := (.Values.trainingRuntime | default dict).image | default "" -}}
{{/* Keep in sync with values.yaml: legacy env settings may override this default. */}}
{{- $defaultRuntime := "quay.io/modh/ray:2.58.0-py312-cu130-th0112" -}}
{{- range .Values.env -}}
{{- if eq .name "TRAINING_RAY_IMAGE" -}}
{{- if and $runtime (ne $runtime $defaultRuntime) -}}
{{- fail "Configure trainingRuntime.image or env TRAINING_RAY_IMAGE, not both; prefer trainingRuntime.image." -}}
{{- end -}}
{{- $runtime = .value | default "" -}}
{{- end -}}
{{- end -}}
{{- required "trainingRuntime.image is required (or legacy env TRAINING_RAY_IMAGE)." $runtime -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "training-service.labels" -}}
helm.sh/chart: {{ include "training-service.chart" . }}
{{ include "training-service.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels.
*/}}
{{- define "training-service.selectorLabels" -}}
app.kubernetes.io/name: {{ include "training-service.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Chart name and version.
*/}}
{{- define "training-service.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create the name of the service account to use.
*/}}
{{- define "training-service.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "training-service.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/* Validate the resolved gateway and legacy migration settings once per render. */}}
{{- define "training-service.validateLogBackend" -}}
{{- $gateway := .Values.logBackend.gatewayURL -}}
{{- $ca := .Values.logBackend.trustedCA -}}
{{- $legacyGateway := false -}}
{{- $legacyCA := false -}}
{{- $legacyTimeout := false -}}
{{- range .Values.env -}}
{{- if eq .name "TRAINING_SERVICE_RAY_GATEWAY_URL" -}}
{{- if or $gateway $legacyGateway -}}{{- fail "Configure logBackend.gatewayURL or env TRAINING_SERVICE_RAY_GATEWAY_URL once, not both." -}}{{- end -}}
{{- $legacyGateway = true -}}
{{- $gateway = .value | default "" -}}
{{- end -}}
{{- if eq .name "TRAINING_SERVICE_RAY_CA_FILE" -}}
{{- if or $ca.configMapName $ca.bundle $legacyCA -}}{{- fail "Configure logBackend.trustedCA or env TRAINING_SERVICE_RAY_CA_FILE once, not both." -}}{{- end -}}
{{- $legacyCA = true -}}
{{- end -}}
{{- if eq .name "TRAINING_SERVICE_RAY_CA_DISCOVERY" -}}{{- fail "Configure logBackend.trustedCA.autoDiscover instead of env TRAINING_SERVICE_RAY_CA_DISCOVERY." -}}{{- end -}}
{{- if eq .name "TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS" -}}
{{- if or (ne (int $.Values.logBackend.streamTimeoutSeconds) 300) $legacyTimeout -}}{{- fail "Configure logBackend.streamTimeoutSeconds or env TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS once, not both." -}}{{- end -}}
{{- if not (regexMatch "^[1-9][0-9]*$" (.value | default "" | toString)) -}}{{- fail "env TRAINING_SERVICE_LOG_STREAM_TIMEOUT_SECONDS must be a positive integer." -}}{{- end -}}
{{- $legacyTimeout = true -}}
{{- end -}}
{{- if or (eq .name "RAY_ADDRESS") (eq .name "RAY_API_SERVER_ADDRESS") -}}{{- fail "Unset RAY_ADDRESS and RAY_API_SERVER_ADDRESS in the API pod." -}}{{- end -}}
{{- end -}}
{{- if and (not $gateway) (not .Values.serviceAccount.automount) -}}{{- fail "serviceAccount.automount must be true for automatic gateway discovery; alternatively provide logBackend.gatewayURL and a configured service identity." -}}{{- end -}}
{{- if and $gateway (not (regexMatch "^https://([a-zA-Z0-9]([a-zA-Z0-9.-]*[a-zA-Z0-9])?|\\[[0-9a-fA-F:]+\\])(:[0-9]{1,5})?/?$" $gateway)) -}}{{- fail "logBackend.gatewayURL must be an HTTPS origin without credentials, a path, query parameters, or a fragment." -}}{{- end -}}
{{- $port := regexFind ":[0-9]+/?$" $gateway | trimPrefix ":" | trimSuffix "/" -}}
{{- if and $port (or (lt (int $port) 1) (gt (int $port) 65535)) -}}{{- fail "logBackend.gatewayURL port must be between 1 and 65535." -}}{{- end -}}
{{- if and $ca.configMapName $ca.bundle -}}{{- fail "Configure logBackend.trustedCA.configMapName or bundle, not both." -}}{{- end -}}
{{- if and (or $ca.configMapName $ca.bundle) (not $ca.key) -}}{{- fail "logBackend.trustedCA.key is required when a CA is configured." -}}{{- end -}}
{{- if and $ca.bundle (or (not (contains "-----BEGIN CERTIFICATE-----" $ca.bundle)) (contains "PRIVATE KEY" $ca.bundle)) -}}{{- fail "logBackend.trustedCA.bundle must contain public PEM certificates only." -}}{{- end -}}
{{- end -}}

{{- define "training-service.logCAName" -}}
{{- if .Values.logBackend.trustedCA.bundle -}}
{{- printf "%s-log-ca" (include "training-service.fullname" . | trunc 56 | trimSuffix "-") -}}
{{- else -}}
{{- .Values.logBackend.trustedCA.configMapName -}}
{{- end -}}
{{- end -}}

{{/* An explicit structured or legacy gateway bypasses platform discovery. */}}
{{- define "training-service.discoverGateway" -}}
{{- $gateway := .Values.logBackend.gatewayURL -}}
{{- range .Values.env -}}
{{- if eq .name "TRAINING_SERVICE_RAY_GATEWAY_URL" -}}
{{- $gateway = .value | default "" -}}
{{- end -}}
{{- end -}}
{{- if not $gateway -}}true{{- end -}}
{{- end -}}

{{/* Platform ingress trust accompanies gateway discovery unless explicitly overridden. */}}
{{- define "training-service.discoverLogCA" -}}
{{- $override := or .Values.logBackend.trustedCA.configMapName .Values.logBackend.trustedCA.bundle -}}
{{- range .Values.env -}}
{{- if has .name (list "TRAINING_SERVICE_RAY_CA_FILE" "REQUESTS_CA_BUNDLE" "CURL_CA_BUNDLE") -}}
{{- $override = true -}}
{{- end -}}
{{- end -}}
{{- if and .Values.logBackend.trustedCA.autoDiscover (include "training-service.discoverGateway" .) (not $override) -}}true{{- end -}}
{{- end -}}
