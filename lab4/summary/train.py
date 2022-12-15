import os.path

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import Vocab, SummaryDataset, load_data
from model import PGN

global vocab


def train(model, train_dataloader, loss_fn, optimizer, epochs):
    model = model.to(config.device)
    model.train()
    train_loss = []
    for _ in range(epochs):
        total_loss = 0
        with tqdm(total=len(train_dataloader) * config.batch_size) as pbar:
            for i, (text, text_len, title_in, title_out, oovs, len_oovs) in enumerate(train_dataloader):
                text = text.to(config.device)
                title_in = title_in.to(config.device)
                title_out = title_out.to(config.device)
                optimizer.zero_grad()
                title_pred, attention_weights, coverage_vector = model(text, title_in, text_len, len_oovs)
                loss = loss_fn(
                    title_pred.transpose(1, 2).to(config.device),
                    title_out
                ) + torch.mean(torch.sum(torch.min(attention_weights, coverage_vector), dim=1))
                assert not torch.isnan(loss)
                cur_loss = loss.item()
                loss.backward()
                clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                total_loss += cur_loss
                pbar.update(config.batch_size)
                pbar.set_postfix(avg_loss=total_loss / (i + 1), cur_loss=cur_loss)
        train_loss.append(total_loss / len(train_dataloader))
        torch.save(model.state_dict(), config.model_save_path)
        torch.save(optimizer.state_dict(), config.optim_save_path)
    return train_loss


def collate_fn(batch):
    return pad_sequence([torch.tensor(bat['text_ids']) for bat in batch], batch_first=True,
                        padding_value=vocab[config.PAD_TOKEN]), \
        torch.tensor([len(bat['text_ids']) for bat in batch]), \
        pad_sequence([torch.tensor(bat['title_ids'][:-1]) for bat in batch], batch_first=True,
                     padding_value=vocab[config.PAD_TOKEN]), \
        pad_sequence([torch.tensor(bat['title_ids'][1:]) for bat in batch], batch_first=True,
                     padding_value=vocab[config.PAD_TOKEN]), \
        [bat['oovs'] for bat in batch], \
        [bat['len_oovs'] for bat in batch]


def main():
    global vocab
    train_text, train_title = load_data(config.train_data_path)
    vocab = Vocab(train_text + train_title)
    train_dataset = SummaryDataset(vocab, train_text, train_title)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)

    model = PGN(vocab)

    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab[config.PAD_TOKEN])
    optimizer = torch.optim.Adam(model.parameters(recurse=True), lr=config.lr)

    if os.path.isfile(config.model_save_path):
        yes = input('load model to continue training? [y/n] ')
        if yes.lower() == 'y':
            model.load_state_dict((torch.load(config.model_save_path)))
            optimizer.load_state_dict((torch.load(config.optim_save_path)))
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(config.device)
    train(model, train_dataloader, loss_fn, optimizer, config.epochs)
    torch.save(vocab, config.vocab_save_path)


if __name__ == '__main__':
    main()
