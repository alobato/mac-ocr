# macOS OCR

CLI tool for document OCR using macOS Vision framework.

Extracts text from document images (converted PDFs, photos, scans, etc.) using the macOS Vision library via PyObjC.

## Installation

Install using pipx:

```bash
pipx install git+https://github.com/alobato/mac-ocr.git
```

Or install using pip:

```bash
pip install git+https://github.com/alobato/mac-ocr.git
```

## Requirements

- macOS (uses Vision framework)
- Python 3.8+

## Usage

```bash
# Extract text from an image
mac-ocr document.jpg

# Save to text file
mac-ocr document.jpg --output result.txt

# JSON output with details
mac-ocr document.jpg --format json

# Multiple files
mac-ocr *.jpg --output-dir results/

# Adjust minimum confidence and languages
mac-ocr document.jpg --min-confidence 0.5 --languages pt-BR,en-US

# Fast mode (less accurate)
mac-ocr document.jpg --level fast
```

## Options

- `--output`, `-o`: Output file (text only). If not specified, prints to stdout
- `--output-dir`: Directory to save results (creates .txt file for each image)
- `--format`: Output format: `text` (default) or `json`
- `--min-confidence`: Minimum confidence to include results (0.0-1.0, default: 0.0)
- `--languages`: Languages separated by comma (e.g., pt-BR,pt-PT,en-US). Default: pt-BR,pt-PT,en-US
- `--level`: Accuracy level: `fast` (quick) or `accurate` (precise, default)
- `--scale`: Scale factor to improve OCR (e.g., 2.0 to double size). Default: 1.0
- `--no-preserve-layout`: Do not try to preserve layout/line breaks (just concatenate text)

## License

MIT
