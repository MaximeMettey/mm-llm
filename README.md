# Mini LLM

A lightweight transformer-based language model implementation built with PyTorch, designed for educational purposes and experimentation.

## Overview

This project implements a mini version of a Large Language Model (LLM) using transformer architecture. It includes a complete pipeline for training, testing, and text generation with a custom tokenizer and model architecture.

## Features

- **Transformer Architecture**: Multi-head attention mechanism with feed-forward networks
- **Custom Tokenizer**: Character-level tokenization for text processing
- **Training Pipeline**: Complete training loop with loss monitoring and visualization
- **Text Generation**: Inference capabilities with temperature-controlled sampling
- **Code Assistant Training**: Specialized training for code-related tasks
- **Model Persistence**: Save and load trained models

## Project Structure

```
mmllm/
├── model.py                    # Core transformer model implementation
├── tokenizer.py               # Simple character-level tokenizer
├── train.py                   # Main training script
├── test_model.py              # Model testing and text generation
├── data_loader.py             # Data loading utilities
├── train_code_assistant.py    # Code assistant specific training
├── code_assistant_data.py     # Code assistant data preparation
├── purpose_specific_training.py # Specialized training methods
├── train_improved.py          # Enhanced training implementation
├── improved_test.py           # Improved testing utilities
├── training_analysis.py       # Training performance analysis
├── requirements.txt           # Python dependencies
├── mini_llm.pth              # Trained model weights
└── venv/                     # Virtual environment
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mmllm
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training a Model

```bash
python train.py
```

This will train a mini LLM on sample text data and save the model weights to `mini_llm.pth`.

### Testing and Text Generation

```bash
python test_model.py
```

Generate text using the trained model with customizable prompts and parameters.

### Code Assistant Training

```bash
python train_code_assistant.py
```

Train a specialized version for code-related tasks.

## Model Architecture

The model implements a standard transformer architecture with:

- **Multi-Head Attention**: Parallel attention mechanisms for better representation learning
- **Feed-Forward Networks**: Position-wise fully connected layers
- **Layer Normalization**: Stabilizes training and improves convergence
- **Positional Encoding**: Enables the model to understand sequence order

### Key Components

- `MultiHeadAttention`: Implements scaled dot-product attention
- `FeedForward`: Position-wise feed-forward network
- `TransformerBlock`: Complete transformer layer with attention and feed-forward
- `MiniLLM`: Full model combining embedding, transformer blocks, and output layers

## Configuration

Default model parameters:
- Vocabulary size: Based on input text character set
- Model dimension: 128
- Number of heads: 8
- Number of layers: 4
- Feed-forward dimension: 512
- Sequence length: 64

## Requirements

- Python 3.7+
- PyTorch 2.0.0+
- NumPy 1.21.0+
- tqdm 4.62.0+
- matplotlib 3.5.0+

## License

This project is for educational purposes. Please ensure compliance with relevant licenses when using or distributing.

## Contributing

This is an educational project. Feel free to experiment and extend the functionality for learning purposes.