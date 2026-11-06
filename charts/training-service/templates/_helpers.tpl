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
{{- $runtime -}}
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
