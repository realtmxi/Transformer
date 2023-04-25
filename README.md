# CSC311 project: Estimating the student's quiz correctness using transformer

This repository contains the implementation of a custom Transformer model for predicting student-question interactions. The model captures the relationships between students and questions based on their historical performance data.

## Overview

The Custom Transformer model is designed to predict whether a student will answer a question correctly or not. The model incorporates students' positive and negative answer histories and question embeddings to make predictions.

## Model Architecture
![Architecture.jpeg](Architecture.jpeg)
**Figure 1:** Model architecture

The Custom Transformer model includes the following components:

- Student Embedding: Converts 1774-dimensional student vectors into a lower-dimensional representation.
- Question Embedding: Converts 768-dimensional question vectors into a lower-dimensional representation.
- Multihead Attention Layers: A stack of MultiheadAttention layers for capturing the relationships between students and questions.
- Feedforward Network: A two-layer feedforward network for generating binary classification outputs.

The model uses the question as the query and the student and question embeddings as key and value pairs in the MultiheadAttention layers. The number of layers in the attention mechanism can be adjusted as needed.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Transformers

## Setup

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Clone this repository: 

```
git clone https://github.com/yourusername/student-question-interaction-prediction.git
```

3. Head into `part_b` and use `Transformer_Deep_Attention.ipynb`

# License
MIT License

# Acknowledgements
```
This project is inspired by the paper "Wang, Zichao, et al. Diagnostic Questions: The NeurIPS 2020 Education Challenge. arXiv preprint arXiv:2007.12061 (2020)".
```