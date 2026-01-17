# LLM Lab 101

Hands-on notebooks covering core LLM concepts and building blocks.

## Notebooks

- `001- Byte_Pair_Encoding.ipynb` - Byte pair encoding basics and examples.
- `002- Positional_Embeddings.ipynb` - Positional embeddings and intuition.
- `003- Self_&_Multihead_Attention.ipynb` - Self-attention and multi-head attention.
- `004- Transformers_&_QKV.ipynb` - Transformer components and QKV flow.
- `005- Sampling_Parameters.ipynb` - Sampling parameters and decoding behavior.

## What You Build

Tokenization & Embeddings
- build byte-pair encoder + train your own subword vocab
- write a token visualizer to map words/chunks to IDs
- one-hot vs learned-embedding: plot cosine distances

Positional Embeddings
- classic sinusoidal vs learned vs RoPE vs ALiBi: demo all four
- animate a toy sequence being position-encoded in 3D
- ablate positions and watch attention collapse

Self-Attention & Multihead Attention
- hand-wire dot-product attention for one token
- scale to multi-head and plot per-head weight heatmaps
- mask out future tokens and verify the causal property

Transformers, QKV, & Stacking
- stack attention with LayerNorm + residuals into one block
- generalize to an n-block mini-former on toy data
- dissect Q, K, V: swap them and observe failure modes

Sampling Parameters
- code a sampler dashboard to tune temp/top-k/top-p
- plot entropy vs output diversity as you sweep params
- set temp=0 (argmax) and observe repetition

## Usage

Open the notebooks with Jupyter or VS Code and run cells top to bottom.

## Requirements

- Python 3.10+ recommended
- Jupyter Lab or VS Code with the Jupyter extension

## Getting Started

```bash
# optional virtual environment
python -m venv .venv
.venv/Scripts/activate

# install essentials
pip install jupyter numpy matplotlib
```

## Notes

- The notebooks are designed to be read in order.
- If you use a different environment, ensure basic scientific packages are available.
