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

X_train = [] #-->tokens
Y_train = [] #-->tokens

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

###################-----------NOTES-----------###################
# the
# attention_block1 = tf.keras.layers.MultiHeadAttention(num_heads=HEADS, key_dim=DIM//HEADS)
# is
# # Single head version (no W_O, no multi-head split)
# Q = X @ W_Q
# K = X @ W_K  
# V = X @ W_V
# scores = Q @ K.T / np.sqrt(dim)
# attention_weights = softmax(scores)
# output = attention_weights @ V (output is X with "meaning". it is also the "attention output")

#############################################################################################################
# X (what goes IN)
# =================================================
# X = (5 × 768)    →    tokens × embedding_dims

#          dim0 ... dim767
# 'the'  [ 0.0497 ... 0.0412 ]
# 'cat'  [ 0.0333 ... 0.0227 ]
# 'sat'  [ 0.0093 ... -0.0356]
# 'on'   [ 0.0769 ... 0.0341 ]
# 'mat'  [ 0.0122 ... 0.0178 ]

# Logits (what comes OUT)
# =================================================
# logits = (5 × vocab_size)    →    tokens × vocabulary

#          'the' 'cat' 'sat' 'on' 'mat' 'dog' 'log' ...all vocab words
# 'the'  [  2.1   4.7   0.3   1.2   0.8   0.1   0.5  ...]
# 'cat'  [  1.8   0.9   5.2   0.3   1.1   0.4   0.2  ...]
# 'sat'  [  0.4   1.3   0.7   4.9   0.6   0.2   0.8  ...]
# 'on'   [  0.9   0.4   1.1   0.8   5.3   0.1   0.3  ...]
# 'mat'  [  1.2   0.7   0.3   0.5   0.9   3.8   0.4  ...]


#############################################################################################################
# logits  = (5 × vocab_size)     Y_train = (5,)
#            ↓                              ↓
# 'the' → [2.1, 4.7, 0.3, ...]    correct answer → 52  ("cat")
# 'cat' → [1.8, 0.9, 5.2, ...]    correct answer →  3  ("sat")
# 'sat' → [0.4, 1.3, 0.7, ...]    correct answer → 56  ("on")
# 'on'  → [0.9, 0.4, 1.1, ...]    correct answer → 77  ("mat")
# 'mat' → [1.2, 0.7, 0.3, ...]    correct answer → 89  (".")
#
# and action to
# Row 0: logits = [2.1, 4.7, 0.3, ...]
#        correct answer index = 52
#        score AT index 52 = 4.7   ← was this the highest? loss measures that
# ######note######
# "sparse"    → Y_train gives just the INDEX  → [52, 3, 56, 77, 89]
# "not sparse"→ Y_train gives full one-hot    → [[0,0,...,1,...,0],
#                                                [0,0,1,...,0,...,0],
#                                                ...]  ← (5 × vocab_size)

# #########################------VISUALISED STEPS (TRAINING)------#########################
# 1 — corpus
# Raw text. Our entire training corpus is one tiny sentence.
# corpus = "the cat sat"
# ↓
# unique characters / words → thecatsat

# 2 — token IDs (vocab)
# Each unique word gets an integer ID. Sorted alphabetically.
# word	token ID
# cat	0
# sat	1
# the	2
# "the cat sat" encodes to → 201

# 3 — X_train & Y_train (SEQ_LEN=2)
# We slide a window of size 2 across the token IDs. X_train = input, Y_train = next token (shifted by 1). Still just integers — no embeddings yet.
# token sequence: [ 2, 0, 1 ] → the, cat, sat
# sequence	X_train (input)	Y_train (target)
# window 0	[ 2, 0 ] → "the cat"	[ 0, 1 ] → "cat sat"
# window 1	[ 0, 1 ] → "cat sat"	[ 1, ? ] → "sat ???"
# shape of X_train = (2 sequences × 2 tokens)  |  still plain integers

# 4 — embedding lookup → X matrix
# INSIDE forward_pass: each token ID is looked up in the embedding table. Each token becomes a vector of size 3. These are random at first — they get trained later.
# embedding table (randomly initialised, size = vocab × 3)
# token	ID	dim0	dim1	dim2
# cat	0	0.21	0.45	-0.12
# sat	1	0.09	0.61	0.72
# the	2	0.44	0.22	0.11
# ↓ lookup for window 0: X_train=[2,0] → "the","cat"
# X matrix = (2 tokens × 3 dims)
# token	dim0	dim1	dim2
# the (id=2)	0.44	0.22	0.11
# cat (id=0)	0.21	0.45	-0.12
# THIS is the X in Q = X @ W_Q, K = X @ W_K, V = X @ W_V

# 5 — Q, K, V via W matrices
# X is projected into 3 different views. W_Q, W_K, W_V are also random at first. Shape stays (2×3) — same tokens, different perspective.
# X (2×3) @ W_Q (3×3) = Q (2×3)
# dim0	dim1	dim2
# Q "the"	0.31	0.19	0.08
# Q "cat"	0.14	0.38	-0.09
# X (2×3) @ W_K (3×3) = K (2×3)
# dim0	dim1	dim2
# K "the"	0.22	0.41	0.17
# K "cat"	0.09	0.28	-0.14
# X (2×3) @ W_V (3×3) = V (2×3)
# dim0	dim1	dim2
# V "the"	0.38	0.11	0.24
# V "cat"	0.17	0.52	-0.08

# 6 — attention → context-enriched output → logits
# Attention blends V vectors using Q·Kᵀ scores. Output is still (2×3). Then output_projection squashes 3 dims → vocab_size scores (logits). Each row = scores for every possible next word.
# attention output (2×3) — "the" now influenced by "cat" and vice versa
# dim0	dim1	dim2
# "the" (enriched)	0.29	0.34	0.12
# "cat" (enriched)	0.22	0.41	0.06
# ↓ output_projection: (2×3) → (2×vocab_size=3)
# logits (2 tokens × 3 vocab scores)
# predicting next after...	score "cat"(0)	score "sat"(1)	score "the"(2)
# "the" →	1.2	0.3	0.8
# "cat" →	0.4	2.1	0.6
# highest score per row = model's current best guess

# 7 — compare logits vs Y_train → loss
# Y_train gives the correct next token ID. Loss checks: how high did we score the correct answer? Low score for correct answer = high loss.
# position	logits row	Y_train (correct ID)	correct word	score at correct	result
# after "the"	[1.2, 0.3, 0.8]	0	cat	1.2 ✓ (highest)	low loss
# after "cat"	[0.4, 2.1, 0.6]	1	sat	2.1 ✓ (highest)	low loss
# loss = average of -log(softmax score at correct index) across all positions
# if the model had guessed wrong → score at correct ID would be LOW → loss HIGH → bigger gradient nudge

# 8 — backprop & repeat
# Gradients flow back through every layer. Both the embedding table rows AND the W_Q/K/V matrices get nudged. Then we go back to step 3 with the next batch.
# what gets updated	what it learns
# embedding table row "the"	"what am I?"
# embedding table row "cat"	"what am I?"
# embedding table row "sat"	"what am I?"
# W_Q, W_K, W_V	"how should tokens relate?"
# output_projection	"how to map context → vocab scores?"
# ↓
# loop: step 3 → 4 → 5 → 6 → 7 → 8 → repeat
# iteration 1: loss ≈ 1.09 (random)
# iteration 100: loss ↓
# iteration 1000: loss ↓↓ (meaningful)

# ##########################--VISUALISED STEPS (GENERATION) --##########################
# 0 — key difference vs training
# During training we had X_train AND Y_train. During inference we have ONLY the prompt. There is no Y_train, no loss, no backprop. Weights are FROZEN. We just keep asking: what comes next?
# training	inference
# input	X_train (batches)	only the prompt
# target	Y_train (correct next tokens)	none — we ARE producing it
# loss	computed, backprop runs	not computed
# weights	updated every step	frozen — never change
# output	gradients	generated text

# 1 — encode the prompt
# prompt = "the cat" → encode() → integer token IDs. Same vocab as training: cat=0, sat=1, the=2.
# prompt = "the cat"
# ↓ encode()
# word	token ID
# the	2
# cat	0
# context = [ 2, 0 ] ← shape (1, 2) → 1 batch, 2 tokens
# this is tf.constant([encode(prompt)]) in the code

# 2 — crop to SEQ_LEN window
# context[:, -SEQ_LEN:] → we only ever feed the last SEQ_LEN tokens into forward_pass. If the context is shorter than SEQ_LEN that is fine — we just use what we have.
# SEQ_LEN = 4 in our tiny example
# iteration	full context so far	input_seq (last 4)
# 1st call[ 2, 0 ]	[ 2, 0 ] (shorter than 4, use all)
# 2nd call[ 2, 0, 1 ]	[ 2, 0, 1 ]
# 3rd call[ 2, 0, 1, 2 ]	[ 2, 0, 1, 2 ]
# 4th call[ 2, 0, 1, 2, 0 ]	[ 0, 1, 2, 0 ] ← oldest dropped
# the model never sees more than SEQ_LEN tokens at once — this is its "memory"

# 3 — forward_pass → logits
# Exactly the same forward_pass as training: embed → Q,K,V → attention → MLP → output_projection. BUT weights are frozen. Output = logits shape (1, seq_len, vocab_size).
# input_seq = [ 2, 0 ] → "the", "cat"
# ↓ embed → Q,K,V → attention → output_projection
# logits shape = (1 batch, 2 tokens, 3 vocab)
# position	score "cat"(0)	score "sat"(1)	score "the"(2)
# pos 0 — "the"	1.2	0.3	0.8
# pos 1 — "cat"	0.4	2.1	0.6
# next_char_logits = logits[:, -1, :] → we take ONLY the LAST position row
# ↓
# score "cat"(0)	score "sat"(1)	score "the"(2)
# 0.4	2.1	0.6
# only the last row matters — it is the prediction for what comes AFTER the full prompt

# 4 — temperature scaling
# scaled_logits = next_char_logits / temperature. Temperature controls how sharp or flat the probability distribution is. Lower = more confident (picks top word almost always). Higher = more random.
# raw logits = [ 0.4, 2.1, 0.6 ] (cat, sat, the)
# temperature	scaled logits	effect
# 0.2 (cold)	[ 2.0, 10.5, 3.0 ]	very sharp — "sat" wins almost always
# 0.8 (used here)	[ 0.5, 2.6, 0.75 ]	balanced — some randomness
# 2.0 (hot)	[ 0.2, 1.05, 0.3 ]	flat — all words get similar chance
# dividing by a small number makes differences BIGGER → sharper
# dividing by a big number makes differences SMALLER → flatter

# 5 — softmax → probabilities
# probs = softmax(scaled_logits). Converts raw scores to proper probabilities that sum to 1.0. Now we can sample from them.
# scaled_logits (temp=0.8) = [ 0.5, 2.6, 0.75 ]
# ↓ softmax
# word	scaled logit	exp(logit)	probability
# "cat" (0)	0.50	1.65	0.13
# "sat" (1)	2.60	13.46	0.75
# "the" (2)	0.75	2.12	0.15
# sum		17.23	1.00
# "sat" has 75% chance, "the" 15%, "cat" 13% — but it is NOT guaranteed to pick "sat"

# 6 — sample next token
# tf.random.categorical samples from the probability distribution. It does NOT always pick the highest probability word — it rolls a weighted dice. This is what makes the output creative rather than always identical.
# probs = [ 0.13, 0.75, 0.15 ] → weighted dice roll
# word	prob	wins if random number falls in...
# "cat" (0)	0.13	[ 0.00 → 0.13 ]
# "sat" (1)	0.75	[ 0.13 → 0.88 ] ← largest slice
# "the" (2)	0.15	[ 0.88 → 1.00 ]
# ↓ result this roll: "sat" (id=1)
# next_char_id = 1 → decode([1]) = "sat" → print it

# 7 — append & loop
# The new token is appended to context. Then we go back to step 2 and repeat. The context grows by 1 token each iteration until max_new_chars is reached.
# iteration	context before	token picked	context after	printed so far
# start[ 2, 0 ]	—[ 2, 0 ]	"the cat"
# 1[ 2, 0 ]	1 "sat"[ 2, 0, 1 ]	"the cat sat"
# 2[ 2, 0, 1 ]	2 "the"[ 2, 0, 1, 2 ]	"the cat sat the"
# 3[ 2, 0, 1, 2 ]	0 "cat"[ 2, 0, 1, 2, 0 ]	"the cat sat the cat"
# ...	repeats until max_new_chars reached	...
# no Y_train, no loss, no weight update — just forward_pass → sample → append → repeat


