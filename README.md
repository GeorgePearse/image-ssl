# Image Self-Supervised Learning Playground

A repository for exploring and experimenting with image Self-Supervised Learning (SSL) techniques.

## Overview

This repository contains implementations and experiments with various SSL methods for computer vision:

- **SimCLR** - Simple Framework for Contrastive Learning of Visual Representations
- **MoCo** - Momentum Contrast for Unsupervised Visual Representation Learning
- **BYOL** - Bootstrap Your Own Latent
- **MAE** - Masked Autoencoders
- **SimSiam** - Exploring Simple Siamese Representation Learning

## Project Structure

```
image-ssl/
├── techniques/          # Implementation of various SSL techniques
│   ├── simclr/
│   ├── moco/
│   ├── byol/
│   ├── mae/
│   └── simsiam/
├── data/               # Data loading and augmentation utilities
├── models/             # Backbone architectures (ResNet, ViT, etc.)
├── utils/              # Helper functions and utilities
├── experiments/        # Experiment scripts and notebooks
└── README.md
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

TBD - Examples and experiment scripts coming soon.

## References

- [SimCLR Paper](https://arxiv.org/abs/2002.05709)
- [MoCo Paper](https://arxiv.org/abs/1911.05722)
- [BYOL Paper](https://arxiv.org/abs/2006.07733)
- [MAE Paper](https://arxiv.org/abs/2111.06377)
