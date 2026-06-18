#!/usr/bin/env bash
set -euo pipefail

S3_BUCKET="s3://models-resources"
BASE_PATH="tiny-cnn-seismicML/models"
PUBLIC_BASE="https://models-resources.concord.org"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <model-id>"
    echo "Example: $0 compact-v1"
    exit 1
fi

MODEL_ID="$1"
MODEL_DIR="models/${MODEL_ID}"
METADATA_FILE="${MODEL_DIR}/metadata.json"
WEIGHTS_FILE="${MODEL_DIR}/weights.json"

# Check that both files exist locally
if [ ! -f "$METADATA_FILE" ]; then
    echo "Error: ${METADATA_FILE} not found"
    exit 1
fi

if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "Error: ${WEIGHTS_FILE} not found"
    exit 1
fi

# Extract schema version from $schema field
SCHEMA_URL=$(python3 -c "import json, sys; print(json.load(open('${METADATA_FILE}'))['\$schema'])")
SCHEMA_VERSION=$(echo "$SCHEMA_URL" | grep -oE 'v[0-9]+' | tail -1)

if [ -z "$SCHEMA_VERSION" ]; then
    echo "Error: Could not extract schema version from \$schema field in ${METADATA_FILE}"
    exit 1
fi

S3_PREFIX="${S3_BUCKET}/${BASE_PATH}/${SCHEMA_VERSION}/${MODEL_ID}"

# Validate metadata against the JSON Schema
echo "Validating ${METADATA_FILE} against schema..."
SCHEMA_FILE=$(mktemp)
trap 'rm -f "$SCHEMA_FILE"' EXIT
if ! curl -sf "$SCHEMA_URL" -o "$SCHEMA_FILE"; then
    echo "Error: Could not download schema from ${SCHEMA_URL}"
    exit 1
fi
python3 scripts/validate-metadata.py "$SCHEMA_FILE" "$METADATA_FILE"

# Check that the model doesn't already exist in S3
if aws s3 ls "${S3_PREFIX}/metadata.json" > /dev/null 2>&1; then
    echo "Error: Model already exists at ${S3_PREFIX}/"
    echo "To prevent accidental overwrites, remove the existing model first."
    exit 1
fi

echo "Deploying model '${MODEL_ID}' (schema ${SCHEMA_VERSION})..."
echo "  ${METADATA_FILE} -> ${S3_PREFIX}/metadata.json"
echo "  ${WEIGHTS_FILE}  -> ${S3_PREFIX}/weights.json"
echo ""

# Upload both files
aws s3 cp "$METADATA_FILE" "${S3_PREFIX}/metadata.json" --content-type application/json
aws s3 cp "$WEIGHTS_FILE" "${S3_PREFIX}/weights.json" --content-type application/json

echo ""
echo "Deployed successfully. Public URLs:"
echo "  ${PUBLIC_BASE}/${BASE_PATH}/${SCHEMA_VERSION}/${MODEL_ID}/metadata.json"
echo "  ${PUBLIC_BASE}/${BASE_PATH}/${SCHEMA_VERSION}/${MODEL_ID}/weights.json"
