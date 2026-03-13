
# CausalRE

Honours research project investigating **causal relation extraction (CRE)** from natural language text using transformer-based information extraction models.

---

## Overview

This project explores how **span-based information extraction models** (similar to NER/RE architectures such as SpERT) can be adapted to extract **causal relationships between longer spans of text**.

The core goal is to convert unstructured text into structured **causal triplets**:

Cause → Effect

Example:

"The earthquake caused destruction in the city"

Extracted causal pair:

earthquake → destruction in the city

The work investigates the limitations of discriminative span-based models when applied to **long-span causal relationships**, which are significantly harder than standard entity-relation extraction tasks.

---

## Research Motivation

Causal relation extraction is important for:

- Knowledge graph construction
- Event reasoning
- Scientific document analysis
- Geological report analysis (original project motivation)

Unlike standard entity extraction tasks, causal spans can be **long, complex, and loosely defined**, which introduces challenges such as:

- ambiguous span boundaries
- long-range dependencies
- increased linguistic noise
- limited annotated datasets

---

## Approach

This project adapts a **discriminative transformer-based information extraction architecture** inspired by models such as:

- SpERT
- DYGIE++
- GraphER

The model performs:

1. Transformer encoding of input text
2. Span representation generation
3. Span classification
4. Span-pair relation classification
5. Extraction of causal triplets

Two model variants were explored:

### Short-span model

Uses brute-force span enumeration:

num_spans ≈ sequence_length × max_span_width

Includes binary filtering layers to reduce candidate spans.

### Long-span model

Uses a token tagging approach to avoid span explosion and reduce memory usage.

This variant is more scalable for longer documents but showed some performance trade-offs.

---

## Key Findings

Main observations from the research:

- Discriminative NER/RE style models **can be adapted** for causal extraction
- Performance drops significantly with **longer spans and implicit causality**
- Models perform best on **shorter explicit causal relations**
- A major bottleneck is the **lack of high-quality annotated causal datasets**

Experiments also suggested that **LLMs may be useful as a preprocessing stage** to simplify complex sentences before extraction.

---

## Repository Contents

modules/
    training pipeline
    model layers
    data preparation
    evaluation tools

config.yaml
    model and training configuration

train.py
    entry point for training and inference

Core components implemented:

- data ingestion and validation
- span generation and filtering
- relation classification
- custom evaluation metrics
- configurable training pipeline

---

## Requirements

Training requires a GPU with significant memory due to span enumeration.

Typical configurations require:

16–40 GB GPU RAM

Libraries used:

- PyTorch
- HuggingFace Transformers
- NumPy / Python data stack

---

## Running the Model

Example:

python train.py --config config.yaml

The configuration file controls:

- model architecture
- span width limits
- dataset paths
- training parameters

---

## Author

Nathan Scott  
MSc Data Science (UWA)  
Honours Research Project – 2025

Focus areas:

- Information extraction
- Machine learning
- causal reasoning from text
