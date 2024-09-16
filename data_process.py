import numpy as np
from sklearn.model_selection import train_test_split
import re

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    return data

# Function to preprocess text: lowercase, tokenize, build vocabulary, and create sequences
def preprocess_text(data, seq_length=100, tokenize_by='char'):
    data = data.lower()  # Convert text to lowercase

    if tokenize_by == 'char':
        # Tokenize by character
        chars = sorted(list(set(data)))
        vocab_size = len(chars)

        # Create character to index and index to character mappings
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        idx_to_char = {i: ch for i, ch in enumerate(chars)}

        # Convert text to integer indices
        data_indices = [char_to_idx[ch] for ch in data]

    elif tokenize_by == 'word':
        # Tokenize by words using regular expressions
        words = re.findall(r'\b\w+\b', data)  # Tokenize by words (ignoring punctuation)
        vocab = set(words)
        vocab_size = len(vocab)

        # Create word to index and index to word mappings
        word_to_idx = {word: i for i, word in enumerate(vocab)}
        idx_to_word = {i: word for i, word in enumerate(vocab)}

        # Convert text to integer indices
        data_indices = [word_to_idx[word] for word in words]

    else:
        raise ValueError("tokenize_by must be either 'char' or 'word'")

    # Create sequences and targets
    sequences = []
    targets = []
    for i in range(0, len(data_indices) - seq_length):
        sequences.append(data_indices[i:i + seq_length])
        targets.append(data_indices[i + seq_length])

    print(f"Total sequences created: {len(sequences)}")
    
    return np.array(sequences), np.array(targets), vocab_size

def split_data(sequences, targets, test_size=0.1):
    train_sequences, val_sequences, train_targets, val_targets = train_test_split(
        sequences, targets, test_size=test_size, random_state=42)

    print(f"Training sequences: {len(train_sequences)}, Validation sequences: {len(val_sequences)}")
    return train_sequences, val_sequences, train_targets, val_targets


def prepare_data(file_path, seq_length=100, tokenize_by='char', test_size=0.1):
    data = load_data(file_path)
    sequences, targets, vocab_size = preprocess_text(data, seq_length, tokenize_by)
    train_sequences, val_sequences, train_targets, val_targets = split_data(sequences, targets, test_size)
    
    return train_sequences, val_sequences, train_targets, val_targets, vocab_size