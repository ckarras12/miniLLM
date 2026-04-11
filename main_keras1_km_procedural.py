import tensorflow as tf
import numpy as np

# =============================================================================
# STEP 1: FROM TEXT TO NUMBERS
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
    "To sleep, perchance to dream, ay3, there is the rub:\n"
    "For in that sleep of death what dreams may come,\n"
    "When we have shuffled off this mortal coil,\n"
    "Must give us pause. There is the respect\n"
    "That makes calamity of so long life.\n"
    "To be, or not to be, that is the question:\n"
    "Whether tis nobler in the mind to suffer.\n"
    "To sleep, perchance to dream, ay2, there is the rub:\n"
    "For in that sleep of death what dreams may come,\n"
    "To sleep, perchance to dream, ay2, there is the rub:\n"
    "For in that sleep of death what dreams may come,\n"
    "To sleep, perchance to dream, ay3, there is the rub:\n"
    "For in that sleep of death what dreams may come,\n"
    "To sleep, perchance to dream, ay3, there is the rub:\n"
    "For in that sleep of death what dreams may come,\n"
)

chars = sorted(list(set(TEXT)))
vocab_size = len(chars)
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(string):
    return [char_to_id[c] for c in string]

def decode(ids):
    return ''.join([id_to_char[i] for i in ids])

data = encode(TEXT)
data = tf.constant(data, dtype=tf.int32)

print(f"Text length: {len(TEXT)} characters")
print(f"Vocabulary size: {vocab_size} unique characters")

# =============================================================================
# STEP 2: CONFIGURATION
# =============================================================================

DIM = 64
SEQ_LEN = 16
HEADS = 2
BLOCKS = 2
LR = 0.01
STEPS = 1000

# =============================================================================
# STEP 3: CREATE ALL THE LAYERS
# =============================================================================

token_embedding = tf.keras.layers.Embedding(vocab_size, DIM)
position_embedding = tf.keras.layers.Embedding(64, DIM)

# Block 1
norm1_block1 = tf.keras.layers.LayerNormalization()
norm2_block1 = tf.keras.layers.LayerNormalization()
attention_block1 = tf.keras.layers.MultiHeadAttention(num_heads=HEADS, key_dim=DIM//HEADS)
mlp_hidden_block1 = tf.keras.layers.Dense(DIM * 4, activation='gelu')
mlp_output_block1 = tf.keras.layers.Dense(DIM)

# Block 2
norm1_block2 = tf.keras.layers.LayerNormalization()
norm2_block2 = tf.keras.layers.LayerNormalization()
attention_block2 = tf.keras.layers.MultiHeadAttention(num_heads=HEADS, key_dim=DIM//HEADS)
mlp_hidden_block2 = tf.keras.layers.Dense(DIM * 4, activation='gelu')
mlp_output_block2 = tf.keras.layers.Dense(DIM)

# Output
final_norm = tf.keras.layers.LayerNormalization()
output_projection = tf.keras.layers.Dense(vocab_size)

# =============================================================================
# STEP 4: DEFINE THE FORWARD PASS (Fixed - no tf.shape issues)
# =============================================================================

def forward_pass(input_ids):
    """
    input_ids shape: (batch_size, seq_length)
    output shape: (batch_size, seq_length, vocab_size)
    """
    # Get actual shape values from the tensor shape attribute (not tf.shape)
    batch_size = tf.shape(input_ids)[0]
    seq_length = tf.shape(input_ids)[1]
    
    # Embeddings: token content + position
    token_vectors = token_embedding(input_ids)
    
    # Create position indices [0, 1, 2, ..., seq_length-1] for each batch item
    positions = tf.range(seq_length)[None, :]  # Shape (1, seq_length)
    position_vectors = position_embedding(positions)  # Broadcasting adds batch dim
    
    hidden_state = token_vectors + position_vectors
    
    # Block 1
    # Attention with residual
    normalized = norm1_block1(hidden_state)
    attention_output = attention_block1(normalized, normalized, use_causal_mask=True)
    hidden_state = hidden_state + attention_output
    
    # MLP with residual
    normalized = norm2_block1(hidden_state)
    mlp_out = mlp_output_block1(mlp_hidden_block1(normalized))
    hidden_state = hidden_state + mlp_out
    
    # Block 2
    normalized = norm1_block2(hidden_state)
    attention_output = attention_block2(normalized, normalized, use_causal_mask=True)
    hidden_state = hidden_state + attention_output
    
    normalized = norm2_block2(hidden_state)
    mlp_out = mlp_output_block2(mlp_hidden_block2(normalized))
    hidden_state = hidden_state + mlp_out
    
    # Output
    hidden_state = final_norm(hidden_state)
    logits = output_projection(hidden_state)
    
    return logits

# =============================================================================
# STEP 5: COLLECT ALL TRAINABLE VARIABLES
# =============================================================================

all_layers = [
    token_embedding, position_embedding,
    norm1_block1, norm2_block1, attention_block1, mlp_hidden_block1, mlp_output_block1,
    norm1_block2, norm2_block2, attention_block2, mlp_hidden_block2, mlp_output_block2,
    final_norm, output_projection
]

# Build the variables by calling the model once with dummy data
dummy_input = tf.zeros((1, SEQ_LEN), dtype=tf.int32)
_ = forward_pass(dummy_input)

# Now collect all trainable variables
trainable_vars = []
for layer in all_layers:
    if hasattr(layer, 'trainable_variables'):
        trainable_vars.extend(layer.trainable_variables)

print(f"Total trainable variables: {len(trainable_vars)}")

# =============================================================================
# STEP 6: PREPARE TRAINING DATA
# =============================================================================

X_train = []
Y_train = []

for i in range(len(data) - SEQ_LEN):
    X_train.append(data[i : i + SEQ_LEN])
    Y_train.append(data[i + 1 : i + SEQ_LEN + 1])

X_train = tf.stack(X_train)
Y_train = tf.stack(Y_train)

print(f"Training examples: {len(X_train)}")

# =============================================================================
# STEP 7: TRAINING LOOP (Using forward_pass directly, not model())
# =============================================================================

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

print("\nStarting training...")

for step in range(STEPS):
    with tf.GradientTape() as tape:
        # Call forward_pass directly with concrete tensors
        logits = forward_pass(X_train)
        
        # Calculate loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(Y_train, logits, from_logits=True)
        loss = tf.reduce_mean(loss)
    
    # Compute gradients and update
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    
    if step % 200 == 0:
        print(f"Step {step:4d}: Loss = {loss:.4f}")

print("Training complete!")

# =============================================================================
# STEP 8: GENERATION
# =============================================================================

def generate(prompt="", max_new_chars=200, temperature=0.8):
    context = tf.constant([encode(prompt)], dtype=tf.int32)
    print(prompt, end='', flush=True)
    
    for _ in range(max_new_chars):
        input_seq = context[:, -SEQ_LEN:]
        
        # Call forward_pass directly (not model())
        logits = forward_pass(input_seq)
        
        next_char_logits = logits[:, -1, :]
        scaled_logits = next_char_logits / temperature
        probs = tf.nn.softmax(scaled_logits)
        
        next_char_id = tf.random.categorical(tf.math.log(probs), num_samples=1)[0, 0]
        
        context = tf.concat([context, [[next_char_id]]], axis=1)
        print(decode([next_char_id.numpy()]), end='', flush=True)
    
    print()

# =============================================================================
# STEP 9: TEST
# =============================================================================

print("\n" + "="*50)
print("GENERATION")
print("="*50)

print("\nPrompt: 'be, that is'")
generate("be, that is", max_new_chars=100)

print("\nPrompt: 'perchance to dream'")
generate("perchance to dream", max_new_chars=100)