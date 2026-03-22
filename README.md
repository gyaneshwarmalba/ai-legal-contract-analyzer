# AI Legal Contract Analyzer

> AI-powered RAG Pipeline for Legal Document Analysis using Salesforce + Databricks + AWS Bedrock

## Architecture

![Architecture Diagram](architecture/architecture-diagram.png)

<!-- TODO: Replace with actual architecture diagram from excalidraw.com -->

## Overview

An end-to-end Retrieval-Augmented Generation (RAG) pipeline that automates legal contract analysis. The system ingests contracts from Salesforce, processes them through Databricks, and leverages AWS Bedrock foundation models for intelligent clause extraction, risk scoring, and compliance checking.

### Key Features

- **Document Ingestion**: Automated contract ingestion from Salesforce via Change Data Capture
- **Vector Embeddings**: Contract chunking and embedding using Amazon Titan on Databricks
- **RAG Pipeline**: Context-aware Q&A over legal documents using AWS Bedrock (Claude)
- **Risk Scoring**: Automated clause-level risk assessment with explainability
- **Salesforce Integration**: Results surfaced back to Salesforce as structured records

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | AWS Bedrock (Claude 3.5 Sonnet) |
| Embeddings | Amazon Titan Embeddings |
| Vector Store | FAISS / Databricks Vector Search |
| Orchestration | LangChain |
| Data Platform | Databricks |
| CRM | Salesforce |
| Backend | FastAPI |
| Frontend | Streamlit |

## Setup

### Prerequisites

- Python 3.11+
- AWS account with Bedrock access
- Databricks workspace
- Salesforce developer org

### Installation

```bash
git clone https://github.com/gyaneshwarmalba/ai-legal-contract-analyzer.git
cd ai-legal-contract-analyzer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Fill in your credentials
```

### Running

```bash
uvicorn app.main:app --reload
```

## Demo

<!-- TODO: Add demo GIF/video -->

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
