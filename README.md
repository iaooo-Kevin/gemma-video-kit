# Gemma Video Kit

A lightweight, modular Python package for working with **Gemma 3 / 3n models** on video-based tasks.  
Extract frames, format prompts, and run local inference — all in one place. In the future, we plan to add support for training, finetuning and even Reinforcement Learning.

---

## Installation

## It is advised to create a new virtual environment for this project.
```bash
python -m venv .venv
source .venv/bin/activate
```

Alternatively, if you prefer using conda
```bash
conda create -n gemma-video-kit python=3.10
conda activate gemma-video-kit
```

```bash
git clone git@github.com:iaooo-Kevin/gemma-video-kit.git
cd gemma-video-kit
pip install -e .
