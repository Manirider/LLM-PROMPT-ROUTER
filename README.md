# LLM-PROMPT-ROUTER

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![License](https://img.shields.io/github/license/Manirider/LLM-PROMPT-ROUTER?style=flat-square) ![Last Commit](https://img.shields.io/github/last-commit/Manirider/LLM-PROMPT-ROUTER?style=flat-square) ![Issues](https://img.shields.io/github/issues/Manirider/LLM-PROMPT-ROUTER?style=flat-square)

`portfolio-project`

## Project Overview

An intelligent routing service that classifies user queries and routes them to the most cost-effective model, optimizing prompt performance and system latency.

## Core Features

- Intent classifier routing queries based on difficulty and required domain expertise.
- Multi-model integration supporting routes to OpenAI, Anthropic, and local models.
- Dynamic latency and cost tracking logs for model calls.
- Fallback paths redirecting failed API calls to backup endpoints.
- JSON configuration schemas managing routing rules and model thresholds.

## Technical Flow & Execution

An incoming prompt is analyzed by the intent classifier. If the query requires complex reasoning, it routes to a high-tier model; simpler requests are sent to a faster, lower-cost model.

## Getting Started

### Requirements

- Python 3.10 or higher
- Pip package manager

### Environment Configuration

```bash
# Clone this repository
git clone https://github.com/Manirider/LLM-PROMPT-ROUTER.git
cd LLM-PROMPT-ROUTER

# Create a virtual environment to manage dependencies locally
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install required library dependencies
pip install -r requirements.txt
```

### Execution

```bash
python main.py
```

## Directory Layout

```
LLM-PROMPT-ROUTER/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
└── (source files)
```

## Contributing to the Project

I welcome issues and pull requests to make this project better. Please see the detailed guidelines in the [Contributing Guide](CONTRIBUTING.md).

## Project License

This repository is distributed under the MIT License. For complete terms, see the [LICENSE](LICENSE) file.

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)
