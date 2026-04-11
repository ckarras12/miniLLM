"""
============================================================
  Mini GPT — Character-Level Language Model
  Inspired by Andrej Karpathy's nanoGPT
  Implementation: TensorFlow / Keras
============================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, losses


# ============================================================
# SECTION 1 — HYPERPARAMETERS
# All knobs in one place. Change these to experiment.
# ============================================================

SEQ_LEN    = 40     # how many characters the model sees at once (context window)
EMBED_DIM  = 64     # size of the vector representing each character
NUM_HEADS  = 4      # number of attention heads (EMBED_DIM must be divisible by this)
FF_DIM     = 128    # hidden size inside the feed-forward network
NUM_LAYERS = 2      # how many transformer blocks to stack
DROPOUT    = 0.1    # fraction of neurons randomly dropped during training
BATCH_SIZE = 32     # number of sequences processed together in one update step
EPOCHS     = 150    # how many full passes through the training data
LR         = 3e-3   # learning rate for the Adam optimizer


# ============================================================
# SECTION 2 — DATASET
# The entire "world" the model learns from.
# Every unique character here becomes part of the vocabulary.
# ============================================================

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


# ============================================================
# SECTION 3 — TOKENIZER
# Build a character-level vocabulary from the text.
# Map every unique character to an integer and back.
# ============================================================

# Collect every unique character in sorted order
chars = sorted(set(TEXT))
VOCAB = len(chars)                            # vocabulary size

# Character <-> integer lookup tables
ch2id = {c: i for i, c in enumerate(chars)}  # char  -> int
id2ch = {i: c for i, c in enumerate(chars)}  # int   -> char

# Encode the full text as a list of integers
encoded = [ch2id[c] for c in TEXT]

print(f"[Tokenizer] Text length    : {len(TEXT)} characters")
print(f"[Tokenizer] Vocabulary size: {VOCAB} unique characters")
print(f"[Tokenizer] Characters     : {chars}")


# ============================================================
# SECTION 4 — TRAINING SEQUENCES
# Slide a window of SEQ_LEN across the encoded text.
# Input  = window[0 : SEQ_LEN]
# Target = window[1 : SEQ_LEN + 1]  (shifted by one)
# The model must predict the next character at every position.
# ============================================================

X_list, y_list = [], []

for i in range(len(encoded) - SEQ_LEN):
    X_list.append(encoded[i     : i + SEQ_LEN])       # input window
    y_list.append(encoded[i + 1 : i + SEQ_LEN + 1])   # target = shifted by 1

X = np.array(X_list, dtype=np.int32)   # shape: (num_sequences, SEQ_LEN)
y = np.array(y_list, dtype=np.int32)   # shape: (num_sequences, SEQ_LEN)

print(f"\n[Sequences] Total training sequences: {len(X)}")
print(f"[Sequences] X shape: {X.shape}")
print(f"[Sequences] y shape: {y.shape}")


# ============================================================
# SECTION 5 — MODEL DEFINITION
#
# TransformerBlock:
#   - Multi-Head Self-Attention (causal — no peeking at future chars)
#   - Feed-Forward Network (expand then contract)
#   - Residual connections + LayerNorm after each sub-layer
#
# MiniGPT:
#   - Token Embedding  (what is this character?)
#   - Position Embedding (where is this character in the sequence?)
#   - N x TransformerBlock
#   - Dense output layer -> logits over vocabulary
# ============================================================

class TransformerBlock(layers.Layer):
    """One transformer block: attention + feed-forward + residuals."""

    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()

        # Causal self-attention: each position only attends to past positions
        self.attn  = layers.MultiHeadAttention(
                         num_heads=num_heads,
                         key_dim=embed_dim // num_heads,
                         dropout=DROPOUT)

        # Two-layer feed-forward network applied independently at each position
        self.ff1   = layers.Dense(ff_dim, activation="relu")
        self.ff2   = layers.Dense(embed_dim)

        # LayerNorm stabilises training by normalising each vector
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop  = layers.Dropout(DROPOUT)

    def call(self, x, training=False):
        # --- Self-Attention ---
        # use_causal_mask=True ensures position i cannot attend to position j > i
        attn_out = self.attn(x, x, use_causal_mask=True, training=training)
        x = self.norm1(x + attn_out)           # residual: add input back, then normalise

        # --- Feed-Forward ---
        ff_out = self.ff2(self.ff1(x))          # expand to ff_dim, contract back to embed_dim
        ff_out = self.drop(ff_out, training=training)
        x = self.norm2(x + ff_out)             # residual: add input back, then normalise

        return x                               # shape unchanged: (batch, seq_len, embed_dim)


class MiniGPT(Model):
    """Character-level GPT: embedding -> transformer blocks -> output logits."""

    def __init__(self):
        super().__init__()

        # Learns a 64-dim vector for each character in the vocabulary
        self.token_emb = layers.Embedding(VOCAB,   EMBED_DIM)

        # Learns a 64-dim vector for each position 0..SEQ_LEN-1
        self.pos_emb   = layers.Embedding(SEQ_LEN, EMBED_DIM)

        # Stack of transformer blocks — each refines the representations
        self.blocks    = [TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM)
                          for _ in range(NUM_LAYERS)]

        # Final projection: 64-dim vector -> score for each vocabulary character
        self.out       = layers.Dense(VOCAB)

    def call(self, x, training=False):
        # x shape: (batch, seq_len) — integer token IDs

        # Build a [0, 1, 2, ..., seq_len-1] positions tensor
        positions = tf.range(tf.shape(x)[1])

        # Add token meaning + position meaning together
        x = self.token_emb(x) + self.pos_emb(positions)  # (batch, seq_len, embed_dim)

        # Pass through each transformer block in sequence
        for block in self.blocks:
            x = block(x, training=training)

        # Project to vocabulary: one score per character, at every position
        return self.out(x)    # (batch, seq_len, vocab_size)


# Build the model (run a dummy forward pass to initialise all weights)
model = MiniGPT()
_ = model(tf.zeros((1, SEQ_LEN), dtype=tf.int32))
model.summary()


# ============================================================
# SECTION 6 — COMPILE & TRAIN
#
# Loss: SparseCategoricalCrossentropy
#   - targets are integer IDs (not one-hot vectors)
#   - from_logits=True because our model outputs raw scores, not probabilities
#
# Optimizer: Adam with a slightly high learning rate (3e-3) to learn
#   fast on this small dataset.
# ============================================================

model.compile(
    optimizer=optimizers.Adam(learning_rate=LR),
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

print("\n[Training] Starting training...\n")

history = model.fit(
    X, y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)

print("\n[Training] Done.")
print(f"[Training] Final loss    : {history.history['loss'][-1]:.4f}")
print(f"[Training] Final accuracy: {history.history['accuracy'][-1]:.4f}")


# ============================================================
# SECTION 7 — INFERENCE (TEXT GENERATION)
#
# Given a seed string, generate one character at a time:
#   1. Encode the current context (last SEQ_LEN characters)
#   2. Forward pass -> logits for the last position
#   3. Apply temperature scaling to control randomness
#   4. Softmax -> probability distribution over vocabulary
#   5. Sample one character from the distribution
#   6. Append it and repeat
#
# Temperature:
#   < 1.0  ->  more focused / repetitive output
#   > 1.0  ->  more creative / random output
#   = 1.0  ->  raw model probabilities, no adjustment
# ============================================================

def generate(seed_text, num_chars=200, temperature=0.8):
    """Generate text character by character from a seed string."""

    result = list(seed_text)

    for _ in range(num_chars):

        # Encode the current context; keep only the last SEQ_LEN characters
        context = [ch2id.get(c, 0) for c in result[-SEQ_LEN:]]

        # If we have fewer than SEQ_LEN chars, pad the left side with zeros
        if len(context) < SEQ_LEN:
            context = [0] * (SEQ_LEN - len(context)) + context

        x_in = np.array([context], dtype=np.int32)    # shape: (1, SEQ_LEN)

        # Forward pass: get logits for every position
        logits = model(x_in, training=False)           # (1, SEQ_LEN, VOCAB)

        # We only care about the LAST position — that is the "next character" prediction
        next_logits = logits[0, -1, :].numpy()         # (VOCAB,)

        # Temperature scaling: divide logits before softmax
        next_logits = next_logits / temperature

        # Softmax (with numerical stability trick: subtract max before exp)
        probs = np.exp(next_logits - next_logits.max())
        probs = probs / probs.sum()

        # Sample one character index from the probability distribution
        next_id = np.random.choice(len(probs), p=probs)

        # Append the decoded character to our growing result
        result.append(id2ch[next_id])

    return "".join(result)


# ============================================================
# SECTION 8 — RUN GENERATION
# We test two temperatures to show the effect of the parameter.
# ============================================================

print("\n" + "=" * 55)
print("GENERATED TEXT  |  temperature = 0.8  (balanced)")
print("=" * 55)
print(generate("To be", num_chars=250, temperature=0.8))

print("\n" + "=" * 55)
print("GENERATED TEXT  |  temperature = 0.4  (more focused)")
print("=" * 55)
print(generate("To be", num_chars=250, temperature=0.4))

print("\n" + "=" * 55)
print("GENERATED TEXT  |  temperature = 1.2  (more creative)")
print("=" * 55)
print(generate("to be wished.", num_chars=250, temperature=1.2))