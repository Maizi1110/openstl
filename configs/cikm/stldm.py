method = 'stldm'
model_type = 'stldm'

in_shape = [5, 4, 112, 112]
pre_seq_length = 5
aft_seq_length = 10
total_length = 15

batch_size = 4
val_batch_size = 4
epoch = 100
lr = 1e-4

metrics = ['mse', 'mae']
