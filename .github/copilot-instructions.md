# ChatEV AI Agent Instructions

This document provides essential knowledge for AI agents working with the ChatEV codebase - a framework for EV charging demand prediction using large language models.

## Project Architecture

### Key Components
- `code/` - Core implementation with full training pipeline
  - `data_interface.py` - Handles data loading and preprocessing for EV charging data
  - `model_interface.py` - Main LLM model implementation using PyTorch Lightning
  - `prompts.py` - Defines templates for model inputs and outputs
- `data/` - Contains CSV files for:
  - Charging data (`*.csv`)
  - Infrastructure data (`inf.csv`) 
  - Pricing data (`e_price.csv`, `s_price.csv`)
  - Weather data (`weather_*.csv`)
  - Zone data (`zone_dist.csv`, `adj_filter.csv`)
- `simple.py` - Inference-only implementation
- `finetune.py` - Basic finetuning implementation

### Data Flow
1. Data is loaded from CSVs via `MyDataModule` in `data_interface.py`
2. Training data combines:
   - Local charging history
   - Neighbor zone charging patterns
   - Current prices and weather
   - Zone infrastructure metadata
3. Prompts are generated using templates in `prompts.py`
4. Model predicts future charging demand using LLM

## Critical Workflows

### Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Hugging Face token:
- Get token from huggingface.co/settings/tokens
- Set in `utils.py` for simple version or `model_interface.py` for full version

### Training Modes
- Basic finetuning: `python finetune.py`
- Full training: 
```bash
cd code
python main.py
```

### Advanced Training Options
- Resume from checkpoint: `--ckpt --ckpt_name='last'`
- Few-shot learning: `--few_shot --few_shot_ratio=0.2`
- Meta-learning: `--meta_learning`

## Project Conventions

### Data Processing
- All numerical data is rounded to 4 decimal places for consistency
- Zone relationships use adjacency matrices for neighbor identification
- Weather data is centrally recorded and shared across zones

### Model Input Format
- Historical data uses sliding windows of `seq_len` hours
- Predictions are made for `pre_len` hours ahead
- Inputs combine:
  - Zone characterization (location, infrastructure)
  - Historical charging patterns
  - Current conditions (price, weather)
  - Future price predictions

### Output Format
All predictions should be wrapped in angle brackets `<value>` as per `prompts.py`

## Key Integration Points

### External Dependencies
- Hugging Face model hub for LLaMA models
- PyTorch Lightning for training infrastructure
- CSV data sources for charging/weather/pricing data

### Cross-Component Communication
- Data loading → Prompt generation → Model prediction
- Zone-to-zone relationships for neighbor analysis
- Weather and pricing data integration with charging patterns