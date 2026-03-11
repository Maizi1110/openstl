method = 'exprecast'
# 🚀 必须在这里显式声明 in_shape，否则框架会自动降级为 [7, 4, 112, 112]
in_shape = [5, 4, 112, 112]

pre_seq_length = 5
aft_seq_length = 10
total_length = 15
metrics = ['mse', 'mae', 'rmse', 'csi','pod','hss','far','ssim','crps']
hid_S = 64
hid_T = 256
N_S = 4
N_T = 8
batch_size = 8
lr = 0.001
epoch = 100

