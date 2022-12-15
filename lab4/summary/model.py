import random

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
from dataset import replace_oovs


class Encoder(nn.Module):
    def __init__(self, vocab):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(len(vocab), config.emb_size, padding_idx=vocab[config.PAD_TOKEN])
        self.gru = nn.GRU(config.emb_size, config.hidden_size, num_layers=config.num_layers, batch_first=True,
                          dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()

    def forward(self, enc_input, text_lengths):
        output, hidden = self.gru(pack_padded_sequence(self.dropout(self.embedding(enc_input)), text_lengths,
                                                       batch_first=True, enforce_sorted=False))
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.relu(self.linear(output))
        return output, hidden[-1].detach()


class Decoder(nn.Module):
    def __init__(self, vocab, attention):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(len(vocab), config.emb_size, padding_idx=vocab[config.PAD_TOKEN])
        self.attention = attention
        self.gru = nn.GRU(config.emb_size + config.hidden_size, config.hidden_size, batch_first=True)
        self.linear = nn.Linear(config.emb_size + 2 * config.hidden_size, len(vocab))
        self.w_gen = nn.Linear(config.hidden_size * 2 + config.emb_size, 1)

    def forward(self, dec_input, prev_hidden, enc_output, text_lengths, coverage_vector):
        dec_input = dec_input.unsqueeze(1)
        embedded = self.embedding(dec_input)
        attention_weights, coverage_vector = self.attention(embedded, enc_output, text_lengths, coverage_vector)
        attention_weights = attention_weights.unsqueeze(1)
        c = torch.bmm(attention_weights, enc_output)
        dec_output, dec_hidden = self.gru(torch.cat([embedded, c], dim=2), prev_hidden.unsqueeze(0))
        dec_output = self.linear(torch.cat((dec_output.squeeze(1), c.squeeze(1), embedded.squeeze(1)), dim=1))
        dec_hidden = dec_hidden.squeeze(0)
        p_gen = torch.sigmoid(self.w_gen(torch.cat([dec_hidden, c.squeeze(1), embedded.squeeze(1)], dim=1)))
        return dec_output, dec_hidden, attention_weights.squeeze(1), p_gen, coverage_vector


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.linear = nn.Linear(config.hidden_size * 2 + config.emb_size, config.hidden_size)
        self.v = nn.Linear(config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, dec_input, enc_output, text_lengths, coverage_vector):
        attention = self.v(torch.tanh(self.linear(torch.cat([
            enc_output,
            (dec_input.repeat(1, enc_output.shape[1], 1)),
            (coverage_vector.unsqueeze(2).repeat(1, 1, enc_output.shape[-1]))
        ], dim=2)))).squeeze(-1)
        max_len = enc_output.shape[1]
        mask = torch.arange(max_len).expand(text_lengths.shape[0], max_len) >= text_lengths.unsqueeze(1)
        attention.masked_fill_(mask.to(config.device), float('-inf'))
        attention_weights = self.softmax(attention)
        return attention_weights, coverage_vector + attention_weights


class PGN(nn.Module):
    @staticmethod
    def get_final_distribution(x, p_gen, p_vocab, attention_weights, max_oov):
        p_gen = torch.clamp(p_gen, 0.001, 0.999)
        return torch.cat([
            (p_gen * p_vocab),
            (torch.zeros(((x.shape[0]), max_oov), dtype=torch.float).to(config.device))
        ], dim=-1).scatter_add_(dim=1, index=x, src=((1 - p_gen) * attention_weights))

    def __init__(self, vocab):
        super(PGN, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.encoder = Encoder(vocab)
        self.decoder = Decoder(vocab, Attention())

    def forward(self, src, tgt, src_lengths, len_oovs, teacher_forcing_ratio=0.5):
        tgt_len = tgt.shape[1]
        enc_output, prev_hidden = self.encoder(replace_oovs(src, self.vocab), src_lengths)
        dec_input = tgt[:, 0]
        dec_outputs = torch.zeros(tgt.shape[0], tgt_len, self.vocab_size + max(len_oovs))
        coverage_vector = torch.zeros_like(src, dtype=torch.float32).to(config.device)
        for t in range(tgt_len - 1):
            dec_input = replace_oovs(dec_input, self.vocab)
            dec_output, prev_hidden, attention_weights, p_gen, coverage_vector = self.decoder(dec_input, prev_hidden,
                                                                                              enc_output, src_lengths,
                                                                                              coverage_vector)
            final_distribution = PGN.get_final_distribution(src, p_gen, dec_output, attention_weights, max(len_oovs))
            teacher_force = random.random() < teacher_forcing_ratio
            dec_outputs[:, t, :] = final_distribution
            top1 = final_distribution.argmax(1)
            dec_input = tgt[:, t] if teacher_force else top1
        return dec_outputs, attention_weights, coverage_vector
