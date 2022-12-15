import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

emb_size = 128
hidden_size = 256
num_layers = 2
dropout = 0.5
batch_size = 16
epochs = 20
lr = 1e-3
max_grad_norm = 4
sample = -1

train_data_path = os.path.join('dataset', 'train_data.json')
test_data_path = os.path.join('dataset', 'test_data.json')
model_save_path = os.path.join('model', 'model.pt')
optim_save_path = os.path.join('model', 'optim.pt')
vocab_save_path = os.path.join('model', 'vocab.pt')

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
BOS_TOKEN = '<BOS>'
EOS_TOKEN = '<EOS>'
