#Building transformer with numpy
import numpy as np

def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** ((i + 1) / d_model)))
    return PE

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.shape[-1]
    scores = np.matmul(query, key.T) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask, -np.inf, scores)
    
    attention_weight = softmax(scores)

    return np.matmul(attention_weight, value), attention_weight


class MultiHeadAttention:
    def __init__(self, num_heads, d_model):
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.Wq = np.random.randn(d_model, d_model)
        self.Wk = np.random.randn(d_model, d_model)
        self.Wv = np.random.randn(d_model, d_model)
        self.Wo = np.random.randn(d_model, d_model)
    
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.depth).transpose(0, 2, 1, 3)
    
    def combine_heads(self, x):
        batch_size, num_heads, seq_len, depth = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.num_heads * depth)
    
    def forward(self, query, key, value, mask=None):
        query = np.matmul(query, self.Wq)
        key = np.matmul(key, self.Wk)
        value = np.matmul(value, self.Wv)

        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        attention_out, _ = scaled_dot_product_attention(query, key, value, mask)

        attention_out = self.combine_heads(attention_out)
        return np.matmul(attention_out, self.Wo)
    
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff)
        self.W2 = np.random.randn(d_ff, d_model)
    
    def forward(self, x):
        return np.matmul(np.maximum(0, np.matmul(x, self.W1)), self.W2)  # ReLU
    
class LayerNormalization:
    def __init__(self, d_model, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta

class TransformerLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.multi_head_attention = MultiHeadAttention(num_heads, d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = LayerNormalization(d_model)
        self.layer_norm2 = LayerNormalization(d_model)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer normalization
        attn_output = self.multi_head_attention.forward(x, x, x, mask)
        x = self.layer_norm1.forward(x + attn_output)

        # Feed-forward network with residual connection and layer normalization
        ff_output = self.feed_forward.forward(x)
        return self.layer_norm2.forward(x + ff_output)

class Transformer:
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, seq_len):
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = np.random.randn(vocab_size, d_model)
        self.positional_encoding = positional_encoding(seq_len, d_model)
        self.layers = [TransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
    
    def forward(self, x, mask=None):
        # Add embedding and positional encoding
        x = self.embedding[x] + self.positional_encoding[:x.shape[1]]

        # Pass through each Transformer layer
        for layer in self.layers:
            x = layer.forward(x, mask)
        
        return x


