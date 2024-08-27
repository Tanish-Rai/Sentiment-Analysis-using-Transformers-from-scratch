# Sentiment-Analysis-using-Transformers-from-scratch

# Amazon Reviews Sentiment Analysis with Transformers

This project implements a Transformer-based model for sentiment analysis on Amazon product reviews. It classifies reviews as positive or negative based on the review text.

## Features

- Data preprocessing and cleaning of Amazon review dataset
- Custom implementation of Transformer architecture using PyTorch
- Sentiment classification (positive/negative)
- Training, validation, and testing pipeline

## Requirements

- Python 3.7+
- PyTorch
- Pandas
- Scikit-learn
- Transformers library
- CUDA-capable GPU (recommended for faster training)

## Installation

1. Clone this repository:git clone https://github.com/yourusername/amazon-reviews-transformer.git
cd amazon-reviews-transformer
2. Install the required packages:## Usage

 a) Prepare your data:
- Ensure you have a CSV file named `AmazonReviews.csv` with columns 'Score' and 'Text'
- Place this file in the appropriate directory (adjust `file_path` in the script if needed)
 b) Run the script

3. 3. The script will:
- Load and preprocess the data
- Build the vocabulary
- Split the data into train, validation, and test sets
- Train the Transformer model
- Evaluate the model on the validation set

## Model Architecture

The model uses a Transformer architecture with the following components:

- Embedding Layer
- Positional Encoding
- Multi-Head Attention
- Feed-Forward Networks
- Layer Normalization

## Customization

You can adjust various hyperparameters in the script:

- `MAX_LEN`: Maximum sequence length
- `d_model`: Embedding dimension
- `num_heads`: Number of attention heads
- `d_ff`: Dimension of feed-forward network
- `num_layers`: Number of encoder layers
- `dropout`: Dropout rate
- `epochs`: Number of training epochs
- `lr`: Learning rate
