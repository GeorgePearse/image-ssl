# Image Self-Supervised Learning Playground

A repository for exploring and experimenting with image Self-Supervised Learning (SSL) techniques.

## Overview

This repository contains implementations and experiments with various SSL methods for computer vision:

### Contrastive Learning Methods
- **SimCLR** - Simple Framework for Contrastive Learning of Visual Representations
- **MoCo** - Momentum Contrast for Unsupervised Visual Representation Learning
- **BYOL** - Bootstrap Your Own Latent
- **SimSiam** - Exploring Simple Siamese Representation Learning

### Self-Distillation Methods
- **DINOv2** - Self-Distillation with No Labels (v2)
  - A powerful self-supervised learning method from Meta AI that uses self-distillation with Vision Transformers
  - Uses a student-teacher framework where the student learns from the teacher's features
  - No contrastive loss needed - relies on cross-entropy between student and teacher outputs
  - Produces exceptional features that work well across many downstream tasks
  - DINOv2 improves upon DINO with better data curation, longer training, and architectural improvements

- **DINOv3** - Next Generation Self-Distillation
  - Latest iteration building on DINOv2's foundation
  - Enhanced training strategies and data efficiency
  - Improved feature quality and downstream task performance
  - State-of-the-art results on various vision benchmarks

### Masked Image Modeling
- **MAE** - Masked Autoencoders for self-supervised learning

## Project Structure

```
image-ssl/
├── techniques/          # Implementation of various SSL techniques
│   ├── simclr/         # Contrastive learning
│   ├── moco/           # Momentum contrastive learning
│   ├── byol/           # Bootstrap your own latent
│   ├── dino/           # DINO, DINOv2, DINOv3
│   ├── mae/            # Masked autoencoders
│   └── simsiam/        # Simple Siamese networks
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

### Contrastive Learning
- [SimCLR Paper](https://arxiv.org/abs/2002.05709) - A Simple Framework for Contrastive Learning
- [MoCo Paper](https://arxiv.org/abs/1911.05722) - Momentum Contrast for Unsupervised Visual Representation Learning
- [BYOL Paper](https://arxiv.org/abs/2006.07733) - Bootstrap Your Own Latent

### Self-Distillation
- [DINO Paper](https://arxiv.org/abs/2104.14294) - Emerging Properties in Self-Supervised Vision Transformers
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193) - Learning Robust Visual Features without Supervision
- [DINOv2 GitHub](https://github.com/facebookresearch/dinov2) - Official Meta AI implementation

### Masked Image Modeling
- [MAE Paper](https://arxiv.org/abs/2111.06377) - Masked Autoencoders Are Scalable Vision Learners
