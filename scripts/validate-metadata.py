"""Validate a model metadata JSON file against a JSON Schema.

Usage:
    python scripts/validate-metadata.py <schema_file> <metadata_file>

Exits 0 and prints "Schema validation passed." when the metadata satisfies
the schema; prints the validation errors and exits 1 otherwise.
"""

import json
import sys

import jsonschema


def main(schema_path, metadata_path):
    """Validate metadata against schema.

    Args:
        schema_path: Path to the JSON Schema file.
        metadata_path: Path to the metadata JSON file to validate.

    Returns:
        0 if valid, 1 if invalid.
    """
    with open(schema_path) as f:
        schema = json.load(f)
    with open(metadata_path) as f:
        data = json.load(f)

    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        print("Schema validation failed:")
        for err in errors:
            location = "/".join(str(p) for p in err.path) or "(root)"
            print(f"  {location}: {err.message}")
        return 1

    print("Schema validation passed.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/validate-metadata.py <schema_file> <metadata_file>")
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
