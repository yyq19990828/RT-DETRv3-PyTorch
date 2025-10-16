#!/usr/bin/env bash

set -e

# Parse command line arguments
JSON_MODE=false
TARGET_PATH=""

for arg in "$@"; do
    case "$arg" in
        --json)
            JSON_MODE=true
            ;;
        --help|-h)
            echo "Usage: $0 [path] [--json]"
            echo "  path      Target directory containing paper PDF and code (default: current directory)"
            echo "  --json    Output results in JSON format"
            echo "  --help    Show this help message"
            exit 0
            ;;
        *)
            TARGET_PATH="$arg"
            ;;
    esac
done

# Use current directory if no path specified
if [[ -z "$TARGET_PATH" ]]; then
    TARGET_PATH="."
fi

# Convert to absolute path
TARGET_PATH="$(cd "$TARGET_PATH" && pwd)"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Define paths
TEMPLATE="$REPO_ROOT/.claude/templates/tech-report-template.md"
REPORT_PATH="$TARGET_PATH/tech-report.md"
PDF_PATH=""
HAS_PDF=false

# Search for PDF files
mapfile -t PDF_FILES < <(find "$TARGET_PATH" -maxdepth 1 -name "*.pdf" -type f)
PDF_COUNT=${#PDF_FILES[@]}

if [[ $PDF_COUNT -eq 1 ]]; then
    PDF_PATH="${PDF_FILES[0]}"
    HAS_PDF=true
elif [[ $PDF_COUNT -gt 1 ]]; then
    HAS_PDF=multiple
fi

# Check if source code exists
HAS_CODE=false
if [[ -n "$(find "$TARGET_PATH" -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.cpp" -o -name "*.c" -o -name "*.java" -o -name "*.go" -o -name "*.rs" | head -1)" ]]; then
    HAS_CODE=true
fi

# Copy template if it exists and report doesn't exist
if [[ -f "$TEMPLATE" ]] && [[ ! -f "$REPORT_PATH" ]]; then
    cp "$TEMPLATE" "$REPORT_PATH"
    if ! $JSON_MODE; then
        echo "Copied tech-report template to $REPORT_PATH"
    fi
fi

# Output results
if $JSON_MODE; then
    printf '{"TARGET_PATH":"%s","REPORT_PATH":"%s","PDF_PATH":"%s","PDF_COUNT":%d,"HAS_CODE":"%s","TEMPLATE":"%s"}\n' \
        "$TARGET_PATH" "$REPORT_PATH" "$PDF_PATH" "$PDF_COUNT" "$HAS_CODE" "$TEMPLATE"
else
    echo "=== Tech Report Setup ==="
    echo "TARGET_PATH: $TARGET_PATH"
    echo "REPORT_PATH: $REPORT_PATH"
    if [[ $HAS_PDF == true ]]; then
        echo "PDF_PATH: $PDF_PATH"
    elif [[ $HAS_PDF == multiple ]]; then
        echo "PDF_COUNT: $PDF_COUNT (multiple PDFs found)"
        echo "PDF_FILES:"
        for pdf in "${PDF_FILES[@]}"; do
            echo "  - $(basename "$pdf")"
        done
    else
        echo "PDF_PATH: (not found)"
    fi
    echo "HAS_CODE: $HAS_CODE"
    echo "TEMPLATE: $TEMPLATE"
fi

exit 0
