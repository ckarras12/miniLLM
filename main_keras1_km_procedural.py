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
    "To sleep, perchance to dream, ay2, there is the rub:\n"
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
    # print(X_train)
    # print('------------')
    # print(Y_train)

X_train = tf.stack(X_train)
Y_train = tf.stack(Y_train)
print('+++++++++++++++++++++++++++++++++++++++++')
print(X_train)
print('------------')
print(Y_train)


print(f"Training examples: {len(X_train)}")

# =============================================================================
# STEP 7: TRAINING LOOP (Using forward_pass directly, not model())
# =============================================================================

optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

print("\nStarting training...")

#simple training loop
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
# STEP 7: TRAINING LOOP (Modified to show logits for first 3 rows)
# =============================================================================

# #analytical
# for step in range(STEPS):
#     with tf.GradientTape() as tape:
#         logits = forward_pass(X_train)  # Shape: (1083, 16, 40)
        
#         # Every 200 steps, inspect the first 3 rows in detail
#         if step % 200 == 0:
#             print(f"\n{'='*60}")
#             print(f"STEP {step} - Inspecting first 3 training examples:")
#             print(f"{'='*60}")
            
#             # Show the input text (what the model sees)
#             print("\n1. INPUT SEQUENCES (what the model receives):")
#             for i in range(3):
#                 input_tokens = X_train[i].numpy()
#                 input_text = decode(input_tokens)
#                 print(f"   Row {i}: '{input_text}'")
#                 print(f"         Token IDs: {input_tokens}")
            
#             # Show the target text (what we want the model to predict)
#             print("\n2. TARGET SEQUENCES (what should come next):")
#             for i in range(3):
#                 target_tokens = Y_train[i].numpy()
#                 target_text = decode(target_tokens)
#                 print(f"   Row {i}: '{target_text}'")
#                 print(f"         Token IDs: {target_tokens}")
            
#             # Show the raw logits shape and statistics
#             print("\n3. LOGITS (raw neural network outputs):")
#             print(f"   Shape for first 3 rows: {logits[:3].shape}")  # (3, 16, 40)
            
#             # For each of first 3 rows, show logits for the first position only
#             # (showing all 16*40=640 numbers would be overwhelming)
#             print("\n   Logits for POSITION 0 of each row (first 10 vocab entries shown):")
#             for i in range(3):
#                 pos0_logits = logits[i, 0, :10].numpy()  # First 10 of 40 vocab scores
#                 print(f"   Row {i}, Pos 0: {pos0_logits}")
#                 print(f"         Min: {tf.reduce_min(logits[i, 0]).numpy():.2f}, "
#                       f"Max: {tf.reduce_max(logits[i, 0]).numpy():.2f}")
            
#             # Show what the model currently predicts (argmax of logits)
#             print("\n4. CURRENT PREDICTIONS (argmax of logits):")
#             predictions = tf.argmax(logits[:3], axis=-1).numpy()  # (3, 16)
#             for i in range(3):
#                 pred_tokens = predictions[i]
#                 pred_text = decode(pred_tokens)
#                 print(f"   Row {i}: '{pred_text}'")
#                 print(f"         Token IDs: {pred_tokens}")
#                 # Show if predictions match targets
#                 match = "✓ MATCH" if np.array_equal(pred_tokens, Y_train[i].numpy()) else "✗ DIFF"
#                 print(f"         {match}")
            
#             print(f"\n{'='*60}")
        
#         # Calculate loss as usual
#         loss = tf.keras.losses.sparse_categorical_crossentropy(Y_train, logits, from_logits=True)
#         loss = tf.reduce_mean(loss)
    
#     # Backpropagation and update
#     gradients = tape.gradient(loss, trainable_vars)
#     optimizer.apply_gradients(zip(gradients, trainable_vars))
    
#     # Print loss every 200 steps
#     if step % 200 == 0:
#         print(f"Step {step:4d}: Loss = {loss:.4f}")

# print("\nTraining complete!")

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