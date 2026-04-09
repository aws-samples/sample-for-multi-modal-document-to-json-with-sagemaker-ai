# Product Overview

This project provides an end-to-end pipeline for converting multi-page documents (PDFs/images) into structured JSON using Vision LLMs on Amazon SageMaker AI and Amazon Bedrock.

## Purpose
- Fine-tune Vision Language Models (VLMs) for document understanding tasks
- Extract structured data from invoices, tax forms, and other documents
- Achieve high accuracy with small models (3B-11B parameters) and minimal training data

## Key Capabilities
- Dataset preparation and conversion to training formats (SWIFT, Amazon Nova)
- Model fine-tuning using SageMaker and Bedrock
- Batch inference for large document sets
- Comprehensive model evaluation with field-level metrics
- Endpoint deployment for production use

## Supported Models
- Qwen2.5 VL (3B, 7B)
- Llama 3.2 Vision (11B)
- Amazon Nova (via Bedrock fine-tuning)

## Datasets Used
- Fatura2: Multi-layout invoice dataset (CC BY 4.0)
- W2 US Tax Forms: Synthetic tax document dataset
