import tensorflow as tf
import numpy as np

# =============================================================================
# 1. DATA (The "Dataset")
# =============================================================================
TEXT = (
    "To be, or not to be, that is the question:\n"
    "Whether tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles\n"
    "And by opposing end them. To die to sleep,\n"
    "No more; and by a sleep to say we end\n"
    "The heart-ache and the thousand natural shocks\n"
    "That flesh is heir to: tis a consummation\n"
    "Devoutly to be wished. To die, to sleep;\n"
    "To sleep, perchance to dream, ay, there is the rub:\n"
    "For in that sleep of death what dreams may come,\n"
    "When we have shuffled off this mortal coil,\n"
    "Must give us pause. There is the respect\n"
    "That makes calamity of so long life.\n"
    "To be, or not to be, that is the question:\n"
    "Whether tis nobler in the mind to suffer.\n"
)

# Character-level tokenization (simplest possible)
chars = sorted(set(TEXT))  # All unique characters
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}  # char -> int
itos = {i: c for i, c in enumerate(chars)}  # int -> char
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert text to tensor
data = tf.constant(encode(TEXT), dtype=tf.int32)

# =============================================================================
# 2. MODEL (Tiny Transformer - 2 blocks, 64 dims)
# =============================================================================
class TinyBlock(tf.keras.layers.Layer):
    """Minimal transformer block: Attention -> Add/Norm -> MLP -> Add/Norm"""
    def __init__(self, dim, heads=2):
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim//heads)
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(dim*4, activation='gelu'),  # Expand
            tf.keras.layers.Dense(dim)  # Contract
        ])
        
    def call(self, x):
        # Causal mask is handled automatically by MultiHeadAttention with use_causal_mask=True
        attn_out = self.attn(self.norm1(x), self.norm1(x), use_causal_mask=True)
        x = x + attn_out  # Residual
        x = x + self.mlp(self.norm2(x))  # Residual
        return x

class TinyGPT(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(vocab_size, 64)  # Token embeddings
        self.pos = tf.keras.layers.Embedding(32, 64)  # Positional embeddings (max 32 chars)
        self.blocks = [TinyBlock(64) for _ in range(2)]  # Just 2 transformer blocks!
        self.norm = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(vocab_size)  # Projects to vocabulary
        
    def call(self, idx):
        b, t = idx.shape
        # Add token + positional embeddings
        x = self.emb(idx) + self.pos(tf.range(t)[None, :])
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))  # Logits for next char prediction

# =============================================================================
# 3. TRAINING (Overfit to the soliloquy)
# =============================================================================
model = TinyGPT()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Prepare training data: sliding window of length 16, predict next char
seq_len = 16
X, Y = [], []
for i in range(len(data) - seq_len):
    X.append(data[i:i+seq_len])
    Y.append(data[i+1:i+seq_len+1])  # Shifted by 1
X = tf.stack(X)
Y = tf.stack(Y)

print(f"Training on {len(X)} sequences...")

# Training loop (1000 steps is plenty for this tiny data)
for step in range(1000):
    with tf.GradientTape() as tape:
        logits = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(Y, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    if step % 200 == 0:
        print(f"Step {step}: loss = {loss:.4f}")

# =============================================================================
# 4. GENERATION (Autocomplete the soliloquy)
# =============================================================================
def generate(prompt="", max_chars=200):
    """Generate text character by character"""
    context = tf.constant([encode(prompt)], dtype=tf.int32)
    print(prompt, end='', flush=True)
    
    for _ in range(max_chars):
        # Crop to last 32 chars if context gets too long
        context_input = context[:, -32:]
        logits = model(context_input)
        # Take last position's prediction
        next_char_logits = logits[:, -1, :]
        # Sample (temperature 0.8 for coherence)
        probs = tf.nn.softmax(next_char_logits / 0.8)
        next_char = tf.random.categorical(tf.math.log(probs), 1)[0, 0]
        # Append and print
        context = tf.concat([context, [[next_char]]], axis=1)
        print(decode([next_char.numpy()]), end='', flush=True)
    
    print()

# Test it
print("\n" + "="*50)
print("GENERATING FROM 'To be':")
generate("To be", max_chars=100)

print("\nGENERATING FROM 'Whether':")
generate("Whether", max_chars=100)


print("\nGENERATING FROM 'Whether':")
generate("to be wished.", max_chars=100)
