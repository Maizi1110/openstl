method = 'stldm'
model_type = 'stldm'

in_shape = [13, 1, 128, 128]
pre_seq_length = 13
aft_seq_length = 12
total_length = 25

batch_size = 1
val_batch_size = 1
epoch = 100
lr = 1e-4

data_name = 'vil'
metrics = ['mse', 'mae', 'pod', 'sucr', 'csi', 'lpips']

