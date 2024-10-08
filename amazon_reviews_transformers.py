# -*- coding: utf-8 -*-
"""Amazon Reviews Transformers.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10-2g3ohGJjVULIbi-JK48njZ6BBjvhEL
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Install necessary libraries
!pip install transformers
!pip install torch
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from google.colab import drive
drive.mount('/content/drive')

import csv

file_path = '/content/drive/My Drive/AmazonReviews.csv'  # Adjust the path to your file location

# Try reading the CSV file with different options to handle the error
try:
    data = pd.read_csv(file_path, engine='python', quoting=csv.QUOTE_NONE, on_bad_lines='skip')
    print("File loaded successfully with python engine and quoting=csv.QUOTE_NONE.")
except Exception as e:
    print(f"Error loading file: {e}")

data = data.sample(frac=0.01, random_state=42)
# Display the first few rows to confirm
data.head()

# Only utilize review Text and Score for this analysis
data = data[['Score', 'Text']]

# Remove 3-star reviews (Score 3)
data = data[data['Score'] != 3]

# Categorize reviews with score 1 or 2 as negative sentiment (0)
# and those with score 4 or 5 as positive sentiment (1)
data['Sentiment'] = data['Score'].apply(lambda x: 1 if x > 3 else 0)

# Drop the Score column as it's no longer needed
data = data[['Text', 'Sentiment']]

# Display the first few rows of the cleaned dataset
data.head()

# Split the data into training (75%), validation (10%), and test (15%) sets
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.6, random_state=42)

print(f"Training set: {train_data.shape}")
print(f"Validation set: {val_data.shape}")
print(f"Test set: {test_data.shape}")

import re
import unicodedata
import numpy as np
from torch.utils.data import Dataset, DataLoader

def clean_text(text):
    """
    Clean the input text by removing special characters and lowercasing.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

def tokenize(sentence):
    """
    Tokenize a sentence into words.
    """
    sentence = clean_text(sentence)
    return sentence.split()

def build_vocab(sentences):
    """
    Build a vocabulary from a list of sentences.
    """
    vocab = {}
    for sentence in sentences:
        cleaned_sentence = clean_text(sentence)
        for word in cleaned_sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab) + 1  # Start indexing from 1
    return vocab

def encode_sentence(sentence, vocab):
    """
    Encode a sentence into a list of integers using the vocabulary.
    """
    tokens = tokenize(sentence)
    return [vocab.get(token, 0) for token in tokens]  # Use 0 for unknown tokens

# Example usage
sentences = data['Text'].tolist()
vocab = build_vocab(sentences)

data['Token_Indices'] = data['Text'].apply(lambda x: encode_sentence(x, vocab))

# Pad sequences to a fixed length
MAX_LEN = 100  # Example maximum length
data['Token_Indices'] = data['Token_Indices'].apply(lambda x: x[:MAX_LEN] + [0]*(MAX_LEN - len(x)))

# Prepare data for training and evaluation
X = np.array(data['Token_Indices'].tolist())
y = np.array(data['Sentiment'].tolist())

# Split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

# Create PyTorch Datasets and DataLoaders
class ReviewDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.long)

train_dataset = ReviewDataset(X_train, y_train)
val_dataset = ReviewDataset(X_val, y_val)
test_dataset = ReviewDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model) with positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Example usage of PositionalEncoding
d_model = 512  # Embedding dimension
pos_encoder = PositionalEncoding(d_model)

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)

# Example usage of EmbeddingLayer
vocab_size = len(vocab) + 1  # Plus one for padding
embedding_layer = EmbeddingLayer(vocab_size, d_model)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_model // num_heads  # Dimension of the key vectors
        self.num_heads = num_heads

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(d_model, num_heads)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        query = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        scores, attn_weights = self.attention(query, key, value, mask)

        # Concatenate heads
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear layer
        output = self.out_linear(concat)

        return output, attn_weights

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feedforward
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(d_model)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, num_classes, max_len, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.encoder = Encoder(vocab_size, d_model, num_heads, d_ff, num_layers, max_len, dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        x = self.encoder(x, mask)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# Hyperparameters
vocab_size = len(vocab) + 1
d_model = 512
num_heads = 1
d_ff = 2048
num_layers = 6
num_classes = 2
max_len = MAX_LEN
dropout = 0.1

# Instantiate the model
model = TransformerClassifier(vocab_size, d_model, num_heads, d_ff, num_layers, num_classes, max_len, dropout)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Training loop
def train_model(model, train_loader, val_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_model(model, val_loader)
        print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader)}, Validation Loss: {val_loss}')

def evaluate_model(model, val_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(val_loader)

# Training the model
epochs = 1
lr = 1e-1
train_model(model, train_loader, val_loader, epochs, lr)