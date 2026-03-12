method = 'exprecast'

# SEVIR VIL canonical shape: [T, C, H, W]
in_shape = [13, 1, 128, 128]
pre_seq_length = 13
aft_seq_length = 12
total_length = 25

# Eval defaults (CLI can override)
batch_size = 4
val_batch_size = 4
