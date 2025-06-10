# 🎉 all_kinds_of_test 🎉

Welcome to **all_kinds_of_test**! This repository contains various experimental projects and code snippets, including a custom GPT-2 model training implementation! 🚀

## Project Structure 📁

### baka_gpt/ 🤖
A custom implementation of GPT-2 model training and text generation.

- `train.py` - Train a custom GPT-2 model on your data
- `gen.py` - Generate text using the trained model
- `chat.py` - Interactive chat interface with the model
- `data/` - Training data directory
- `tiny-gpt/` - Trained model checkpoints and configurations

#### Usage:
1. Install dependencies:
```bash
pip install -r baka_gpt/requirements.txt
```

2. Train the model:
```bash
python baka_gpt/train.py
```

3. Generate text:
```bash
python baka_gpt/gen.py
```

4. Chat with the model:
```bash
python baka_gpt/chat.py
```

### baka/ 🧪
Collection of experimental Python scripts:
- `main.py` - Text generation using distilgpt2
- `main2.py`, `main3.py`, `main4.py` - Various experimental implementations

## Features ✨

- Custom GPT-2 model training
- Text generation capabilities
- Interactive chat interface
- Experimental implementations
- Support for custom training data

## Technical Details 🔧

The GPT-2 implementation includes:
- Model architecture: GPT-2 (small)
- Training parameters:
  - Epochs: 5
  - Batch size: 2
  - Sequence length: 128
  - Embedding dimension: 128
  - Layers: 2
  - Attention heads: 2

## How to Contribute 🤝

1. **Identify worthy projects:** Spot a gem that deserves its own repository
2. **Suggest improvements:** Have ideas for better implementations?
3. **Add features:** Want to extend the functionality?

---

Happy coding! 🌟