import torch

batch_size = 1 # 批处理大小
seq_len = 3 # 序列长度
input_size = 4 # 输入维度
hidden_size = 2 # 隐藏层维度

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# (seq, batch, features)
dataset = torch.randn(seq_len, batch_size, input_size)
print(dataset)
hidden = torch.zeros(batch_size, hidden_size)
print(hidden)

for idx, input in enumerate(dataset):
    print( '=' * 20, idx, '=' * 20)
    print( 'Input size: ', input.shape)
    hidden = cell(input, hidden)
    print( 'outputs size: ', hidden.shape)
    print(hidden)
