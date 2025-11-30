# Building LLMs from Scratch 🚀

This repository contains the implementation code and notes from the course **"Building LLMs from scratch"** by **Vizuara**.

The project takes a deep dive into the architecture of Large Language Models, specifically GPT, implementing every component step-by-step using **PyTorch**. It covers the entire pipeline: from data tokenization to coding the Transformer architecture, loading pretrained GPT-2 weights, and fine-tuning the model for specific tasks like classification and instruction following.

## 📚 Project Overview

This notebook covers the following core concepts:

### 1. Data Preparation & Tokenization
- **Text Tokenization:** Building a simple tokenizer using Regex and transitioning to Byte Pair Encoding (BPE) using `tiktoken`.
- **Data Loading:** Creating custom PyTorch Datasets (`GPTDatasetV1`) and DataLoaders to handle sliding windows for pretraining.
- **Embeddings:** Implementing Token Embeddings and Positional Embeddings.

### 2. The Transformer Architecture
- **Self-Attention:** Implementing the attention mechanism from scratch (with and without trainable weights).
- **Causal Attention:** Implementing masking to ensure the model only attends to past tokens (crucial for auto-regressive generation).
- **Multi-Head Attention:** Building the `MultiHeadAttention` class to process information in parallel subspaces.
- **Feed Forward Networks:** Implementing GELU activation and the MLP layers.
- **Layer Normalization:** Building `LayerNorm` from scratch.

### 3. The GPT Model
- Assembling the full **GPT Architecture** (`GPTModel`) by connecting Transformer blocks.
- Implementing **Weight Tying** for the output layer.
- Calculating model parameter counts and memory requirements.

### 4. Text Generation (Inference)
- **Decoding Strategies:**
  - Greedy Decoding.
  - **Temperature Scaling:** To control randomness/creativity.
  - **Top-k Sampling:** To improve coherence by filtering low-probability tokens.
- Loading **Pretrained Weights**: Scripts to download and load official OpenAI **GPT-2 (124M and 355M)** weights into the custom model architecture.

### 5. Fine-Tuning
- **Classification Fine-Tuning:** Converting the generative model into a Spam Classifier (using the SMS Spam Collection dataset).
- **Instruction Fine-Tuning:**
  - Preparing an instruction dataset (Alpaca format).
  - Custom Collate functions for variable-length padding.
  - Training loop with training/validation loss tracking.
  - **Automated Evaluation:** Using **Ollama (Llama 3)** as a judge to score the fine-tuned model's responses.

## 🛠️ Tech Stack

* **Python 3.x**
* **PyTorch** (Neural Network primitives)
* **Tiktoken** (OpenAI's BPE tokenizer)
* **Pandas** (Data manipulation)
* **Matplotlib** (Visualizing loss curves and activation functions)
* **TensorFlow** (Used strictly for extracting original OpenAI weights)
* **Ollama** (For local LLM evaluation)

## ⚡ Getting Started

### Prerequisites

Ensure you have Python installed. You can install the necessary dependencies using:

```bash
pip install torch pandas matplotlib tiktoken tensorflow tqdm
(Optional) For the evaluation step, you will need to install and run Ollama.

Running the Code
The entire implementation is contained within the Jupyter Notebook. You can run it sequentially to observe the build process from scratch.

Tokenization: Observe how raw text is converted to integers.

Model Build: See the shapes of tensors as they pass through Attention and FeedForward layers.

Training: The notebook includes a training loop. Note: Training is set to a low epoch count for demonstration purposes. For better results, increase epochs or run on a GPU.

🧠 Key Implementations
Multi-Head Attention
The core of the Transformer, implemented with nn.Module:

Python

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # ... implementation details ...
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # ...
GPT Model
The assembly of the full architecture:

Python

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
📊 Results
The repository includes visualization functions to plot:

Training vs Validation Loss to check for overfitting.

Classification Accuracy for the spam detection task.
