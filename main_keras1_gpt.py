# tiny_gpt_keras.py
#
# A tiny decoder-only transformer language model in TensorFlow + Keras.
# It follows the same broad idea as Karpathy's tiny GPT examples:
# predict the next token from previous tokens only.
#
# This is NOT a real large language model.
# It is a small teaching model with a coherent hardcoded dataset.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------
# Step 0: Reproducibility
# ------------------------------------------------------------
tf.keras.utils.set_random_seed(42)
np.random.seed(42)

# ------------------------------------------------------------
# Step 1: A small but sensible hardcoded dataset
# ------------------------------------------------------------
# Why this dataset?
# - It is coherent: customer support / operations language.
# - A tiny model learns better from a narrow topic than from random text.
# - It stays fully self-contained: no external downloads needed.

sentences = [
    "the support team answers customer questions by email",
    "the support team checks the order number first",
    "the support agent sends a clear and polite reply",
    "the customer forgot the account password",
    "the customer wants to change the delivery address",
    "the package left the warehouse this morning",
    "shipping usually takes three business days",
    "the refund appears after five business days",
    "the billing page shows the latest invoice",
    "the customer asked for a copy of the receipt",
    "the order status changed to ready for pickup",
    "the warehouse printed the shipping label",
    "the customer received the wrong item",
    "the support team created a replacement order",
    "the support team approved the refund request",
    "the customer asked when the package will arrive",
    "the package arrived later than expected",
    "the customer wants to cancel the subscription",
    "the customer upgraded the monthly plan",
    "the dashboard shows the current account balance",
    "the support team verified the email address",
    "the customer asked for faster delivery",
    "the invoice includes tax and shipping fees",
    "the account was locked after many failed logins",
    "the support agent reset the login link",
    "the customer asked how to track the package",
    "the tracking page shows the latest location",
    "the package is waiting at the local depot",
    "the support team reopened the old ticket",
    "the customer added a second payment method",
    "the subscription renews on the first day of the month",
    "the customer changed the password successfully",
    "the support team apologized for the delay",
    "the help center explains the return policy",
    "the return label was sent by email",
    "the customer asked whether the item is in stock",
    "the store updated the product quantity",
    "the customer requested a call from support",
    "the support team scheduled a phone call",
    "the customer confirmed the shipping address",
    "the billing system recorded the payment",
    "the order was packed and sealed carefully",
    "the customer reported a damaged box",
    "the support team requested a photo of the item",
    "the replacement package was shipped today",
    "the customer asked for an update on the case",
    "the support team sent a short status message",
    "the customer thanked the team for the help",
]

# Special token used as an end-of-sentence marker.
EOS_TOKEN = "endofsentence"

# We append the EOS token to each sentence so the model can learn when a thought ends.
training_texts = [s + f" {EOS_TOKEN}" for s in sentences]

# Join everything into one long stream of tokens.
# This is very similar to the "next token over a text stream" idea used in tiny GPT examples.
full_corpus = " ".join(training_texts)

# ------------------------------------------------------------
# Step 2: Hyperparameters
# ------------------------------------------------------------
# SEQ_LEN is Karpathy's "block_size" idea in simpler words:
# how many previous tokens the model can look at.
SEQ_LEN = 12
VOCAB_LIMIT = 1000

EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 3e-4

# ------------------------------------------------------------
# Step 3: Tokenization with TextVectorization
# ------------------------------------------------------------
# This layer:
# - lowercases the text
# - removes punctuation
# - splits on whitespace
# - maps words to integer ids
vectorizer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    split="whitespace",
    output_mode="int",
    max_tokens=VOCAB_LIMIT,
)

# Build the vocabulary from our training texts.
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(training_texts).batch(16))

# Convert the whole corpus into token ids.
token_ids = vectorizer(tf.constant([full_corpus]))[0].numpy()

# Vocabulary helpers for decoding later.
vocab = vectorizer.get_vocabulary()
vocab_size = len(vocab)
word_to_id = {word: idx for idx, word in enumerate(vocab)}
id_to_word = np.array(vocab)

eos_id = word_to_id[EOS_TOKEN]

print("Vocabulary size:", vocab_size)
print("Total tokens in corpus:", len(token_ids))

# ------------------------------------------------------------
# Step 4: Build training pairs for next-token prediction
# ------------------------------------------------------------
# For each window of length T+1:
# input  = first T tokens
# target = next T tokens
#
# Example:
# input  = [the, support, team]
# target = [support, team, answers]
#
# This teaches the model to predict the next token at every position.

def make_next_token_data(all_token_ids, seq_len):
    x_data = []
    y_data = []

    for start in range(0, len(all_token_ids) - seq_len):
        window = all_token_ids[start:start + seq_len + 1]
        x_data.append(window[:-1])
        y_data.append(window[1:])

    x_data = np.array(x_data, dtype=np.int32)
    y_data = np.array(y_data, dtype=np.int32)
    return x_data, y_data

x_all, y_all = make_next_token_data(token_ids, SEQ_LEN)

# Shuffle once before splitting.
indices = np.random.permutation(len(x_all))
x_all = x_all[indices]
y_all = y_all[indices]

split = int(0.9 * len(x_all))
x_train, y_train = x_all[:split], y_all[:split]
x_val, y_val = x_all[split:], y_all[split:]

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(len(x_train))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("Training samples:", len(x_train))
print("Validation samples:", len(x_val))

# ------------------------------------------------------------
# Step 5: Token + position embedding layer
# ------------------------------------------------------------
# Token embeddings turn token ids into vectors.
# Position embeddings tell the model where each token sits in the sequence.
# Both are added together.

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        length = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        position_vectors = self.pos_emb(positions)
        token_vectors = self.token_emb(x)
        return token_vectors + position_vectors

# ------------------------------------------------------------
# Step 6: One GPT-style transformer block
# ------------------------------------------------------------
# Structure:
# 1. LayerNorm
# 2. Causal self-attention
# 3. Residual connection
# 4. LayerNorm
# 5. Feed-forward network
# 6. Residual connection

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
        )
        self.drop1 = layers.Dropout(dropout_rate)

        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.drop2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Pre-norm attention block
        x_norm = self.norm1(x)

        # use_causal_mask=True is the key decoder behavior:
        # token i cannot look at tokens after i
        attn_out = self.attn(
            query=x_norm,
            value=x_norm,
            key=x_norm,
            use_causal_mask=True,
            training=training,
        )
        attn_out = self.drop1(attn_out, training=training)
        x = x + attn_out

        # Pre-norm feed-forward block
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm, training=training)
        ffn_out = self.drop2(ffn_out, training=training)
        x = x + ffn_out

        return x

# ------------------------------------------------------------
# Step 7: Build the full decoder-only model
# ------------------------------------------------------------
inputs = keras.Input(shape=(SEQ_LEN,), dtype=tf.int32)

x = TokenAndPositionEmbedding(SEQ_LEN, vocab_size, EMBED_DIM)(inputs)

for _ in range(NUM_LAYERS):
    x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT)(x)

x = layers.LayerNormalization(epsilon=1e-6)(x)

# One vocabulary logit vector per position
outputs = layers.Dense(vocab_size)(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# We predict the next token id at each position.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=loss_fn,
    metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)

model.summary()

# ------------------------------------------------------------
# Step 8: Train
# ------------------------------------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=2,
)

print()
print("Final train loss:", round(history.history["loss"][-1], 4))
print("Final val loss:  ", round(history.history["val_loss"][-1], 4))

# ------------------------------------------------------------
# Step 9: Inference helpers
# ------------------------------------------------------------

def encode_text(text):
    """Convert a prompt string into token ids."""
    ids = vectorizer(tf.constant([text]))[0].numpy().tolist()
    # Remove zeros just in case
    ids = [int(i) for i in ids if i != 0]
    return ids

def decode_tokens(token_list):
    """Convert token ids back into readable text."""
    words = []
    for token_id in token_list:
        if token_id == 0:
            continue  # skip padding
        word = id_to_word[token_id]

        if word == EOS_TOKEN:
            words.append(".")
        elif word == "[UNK]":
            # Ignore unknown token in final display
            continue
        else:
            words.append(word)

    text = " ".join(words)
    text = text.replace(" .", ".")
    return text.strip()

def sample_next_token(logits, temperature=1.0, top_k=10):
    """
    Sample one token from logits.
    - temperature < 1.0 makes output more conservative
    - top_k limits sampling to the best k tokens
    """
    logits = np.asarray(logits).astype("float64")

    # Never sample padding or [UNK] during generation
    logits[0] = -1e9
    if len(logits) > 1:
        logits[1] = -1e9

    temperature = max(temperature, 1e-6)
    logits = logits / temperature

    if top_k is not None and 0 < top_k < len(logits):
        top_ids = np.argpartition(logits, -top_k)[-top_k:]
        top_logits = logits[top_ids]
        probs = tf.nn.softmax(top_logits).numpy()
        next_id = np.random.choice(top_ids, p=probs)
    else:
        probs = tf.nn.softmax(logits).numpy()
        next_id = np.random.choice(len(logits), p=probs)

    return int(next_id)

def generate_text(prompt, max_new_tokens=20, temperature=0.8, top_k=10):
    """
    Autoregressive generation:
    1. Encode the prompt
    2. Feed the latest context window to the model
    3. Sample one next token
    4. Append it and repeat
    """
    token_list = encode_text(prompt)

    if len(token_list) == 0:
        raise ValueError("Prompt became empty after tokenization.")

    for _ in range(max_new_tokens):
        context = token_list[-SEQ_LEN:]

        # Right-pad to fixed sequence length
        padded = context + [0] * (SEQ_LEN - len(context))
        x = np.array([padded], dtype=np.int32)

        # Model output shape: (1, SEQ_LEN, vocab_size)
        logits = model.predict(x, verbose=0)

        # We want the next token after the last real token
        last_real_position = len(context) - 1
        next_token_logits = logits[0, last_real_position]

        next_id = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
        )

        token_list.append(next_id)

        if next_id == eos_id:
            break

    return decode_tokens(token_list)

# ------------------------------------------------------------
# Step 10: Try a few prompts
# ------------------------------------------------------------
prompts = [
    "the support team",
    "the customer asked",
    "the package",
    "the billing system",
]

print("\nGenerated samples:\n")
for p in prompts:
    generated = generate_text(
        prompt=p,
        max_new_tokens=18,
        temperature=0.8,
        top_k=8,
    )
    print(f"Prompt:    {p}")
    print(f"Generated: {generated}")
    print("-" * 60)