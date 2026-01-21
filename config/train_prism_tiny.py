# Train a PRISM model on TinyStories (BPE)

out_dir = 'out-prism-tiny'
dataset = 'tinystories'
gradient_accumulation_steps = 2
batch_size = 32
block_size = 512

# Architecture (deeper for logic)
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1

# PRISM Physics
prism_expansion_ratio = 2
prism_freq_multiplier = 3.1415926535  # Pi
prism_lambda_start = 0.01
prism_lambda_end = 0.1

learning_rate = 6e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 6e-5
warmup_iters = 1000

device = 'cuda'
compile = True

# TensorBoard
tensorboard_log = True

