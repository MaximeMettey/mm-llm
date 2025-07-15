# Mini LLM from Scratch

A lightweight transformer-based language model built from scratch with PyTorch, designed for code assistance and conversational AI.

## 🚀 Features

- **Multi-language code support**: PHP, React/React Native, Python, Symfony, API Platform, JavaScript
- **Conversational abilities**: Understands natural language questions about code
- **Configurable training**: Flexible parameters for different use cases
- **Efficient architecture**: Optimized for small-scale deployment
- **Training analysis**: Tools to decide when to train more vs add more data

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

## 🔧 Usage

### Basic Training

Train with default settings (40 epochs):
```bash
python3 train_code_assistant.py
```

### Custom Training Parameters

```bash
# Quick training (10 epochs)
python3 train_code_assistant.py --epochs 10

# High-performance training
python3 train_code_assistant.py --epochs 50 --lr 1e-3 --batch_size 8

# Memory-efficient training
python3 train_code_assistant.py --epochs 20 --seq_len 64 --batch_size 2

# Custom checkpoint frequency
python3 train_code_assistant.py --epochs 30 --save_every 5
```

### Available Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 40 | Number of training epochs |
| `--lr` | 5e-4 | Learning rate |
| `--batch_size` | 4 | Training batch size |
| `--seq_len` | 128 | Input sequence length |
| `--save_every` | 10 | Save checkpoint every N epochs |

### Testing Your Model

```bash
# Test with various code prompts
python3 test_model.py

# Improved testing with better sampling
python3 improved_test.py
```

## 📊 Training Strategy

### When to Train More vs Add More Data

Run the analysis tool to understand your training progress:

```bash
python3 training_analysis.py
```

**Guidelines:**
- **Loss still decreasing** → Continue training
- **Loss plateaued** → Add more data
- **Early epochs (<20)** → Keep training
- **Overfitting signs** → Stop training, add data

### Data Expansion Options

1. **Synthetic Generation** (Easy, 10-20x data)
   ```bash
   python3 expand_data.py
   ```

2. **Documentation Mining** (Medium effort, 5-10x data)
   - Official framework documentation
   - Tutorial examples

3. **Web Scraping** (High effort, 50-100x data)
   - GitHub repositories
   - Code examples from blogs

## 🎯 Model Architecture

- **Type**: Decoder-only Transformer
- **Attention**: Multi-head self-attention with causal masking
- **Layers**: 8 transformer blocks
- **Embedding**: 384 dimensions
- **Vocabulary**: Character-level tokenization
- **Context**: Up to 512 tokens

**Model Specifications:**
```python
MiniLLM(
    vocab_size=tokenizer.vocab_size,
    d_model=384,      # Embedding dimension
    n_heads=12,       # Attention heads
    n_layers=8,       # Transformer layers
    d_ff=1536,        # Feed-forward dimension
    max_seq_len=512,  # Maximum sequence length
    dropout=0.1       # Dropout rate
)
```

## 🔄 Training Process

1. **Data Preparation**: Combines conversational patterns with code examples
2. **Tokenization**: Character-level encoding for multi-language support
3. **Training**: AdamW optimizer with cosine learning rate scheduling
4. **Checkpointing**: Regular model saves for resuming training
5. **Generation**: Top-k sampling for coherent code completion

## 💡 Example Usage

### Code Completion
```python
# Input: "function calculateTotal("
# Output: "function calculateTotal(items) { return items.reduce((sum, item) => sum + item.price, 0); }"
```

### Conversational Code Help
```python
# Input: "How do I create a PHP class?"
# Output: "You can create a PHP class using the class keyword followed by the class name..."
```

## 📈 Performance Tips

### Training Optimization
- **GPU**: Use CUDA if available for faster training
- **Batch Size**: Increase for faster training (if memory allows)
- **Sequence Length**: Shorter sequences train faster
- **Learning Rate**: Start with 5e-4, adjust based on loss curve

### Memory Optimization
- Reduce `batch_size` if running out of memory
- Use shorter `seq_len` for efficiency
- Enable gradient checkpointing for large models

### Quality Improvement
- Train longer when loss is still decreasing
- Add more diverse training data when loss plateaus
- Use lower temperature (0.5-0.7) for more focused generation
- Implement top-k sampling for better code quality

## 🔍 Monitoring Training

Watch for these patterns:
- **Healthy**: Steady loss decrease, stable learning
- **Overfitting**: Loss starts increasing or oscillating
- **Plateaued**: Loss stops decreasing (time for more data)
- **Underfitting**: Very high loss, needs more training

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory**:
```bash
python3 train_code_assistant.py --batch_size 2 --seq_len 64
```

**Training Too Slow**:
```bash
python3 train_code_assistant.py --batch_size 8 --epochs 20
```

**Poor Generation Quality**:
1. Train longer if loss is decreasing
2. Add more training data if loss plateaued
3. Adjust temperature in generation (0.5-0.8)

### File Locations

After training, you'll find:
- `code_assistant_model.pth` - Trained model weights
- `code_assistant_tokenizer.pth` - Tokenizer for encoding/decoding
- `code_assistant_training_loss.png` - Loss curve visualization
- `code_assistant_checkpoint_epoch_*.pth` - Training checkpoints

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