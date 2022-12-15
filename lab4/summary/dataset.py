import json
import re
from collections import defaultdict

import jieba
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import config


class Vocab:
    def __init__(self, sentences):
        self.id2tk = list()
        self.tk2id = {}
        tk_freq = defaultdict(int)
        with tqdm(total=len(sentences)) as pbar:
            for sentence in sentences:
                for token in sentence:
                    tk_freq[token] += 1
                pbar.update(1)
        unique_tokens = [config.PAD_TOKEN, config.BOS_TOKEN, config.EOS_TOKEN] + [token for token, freq in
                                                                                  tk_freq.items() if freq >= 1]
        if config.UNK_TOKEN not in unique_tokens:
            unique_tokens.append(config.UNK_TOKEN)
        for token in unique_tokens:
            self.id2tk.append(token)
            self.tk2id[token] = len(self.id2tk) - 1
        self.unk = self.tk2id[config.UNK_TOKEN]

    def __len__(self):
        return len(self.id2tk)

    def __getitem__(self, token):
        return self.tk2id.get(token, self.unk)

    def tokens2ids(self, tokens):
        return [self[token] for token in tokens]

    def ids2tokens(self, ids):
        return [self.id2tk[idx] for idx in ids]

    def text2ids(self, text_tokens):
        ids = []
        oovs = []
        oovs_set = set()
        unk_id = self.unk
        for token in text_tokens:
            i = self[token]
            if i == unk_id:
                if token not in oovs_set:
                    oovs.append(token)
                    oovs_set.add(token)
                oov_idx = oovs.index(token)
                ids.append(oov_idx + len(self))
            else:
                ids.append(i)
        return ids, oovs, oovs_set

    def title2ids(self, title_tokens, oovs, oovs_set):
        ids = []
        unk_id = self.unk
        for token in title_tokens:
            i = self[token]
            if i == unk_id:
                if token in oovs_set:
                    token_idx = oovs.index(token) + len(self)
                    ids.append(token_idx)
                else:
                    ids.append(unk_id)
            else:
                ids.append(i)
        return ids


class SummaryDataset(Dataset):
    def __init__(self, vocab, text, title):
        self.vocab = vocab
        self.text = text
        self.title = title

    def __getitem__(self, i):
        text_ids, oovs, oovs_set = self.vocab.text2ids(self.text[i])
        return {
            'text_ids': text_ids,
            'oovs': oovs,
            'len_oovs': len(oovs),
            'title_ids': ([self.vocab[config.BOS_TOKEN]] + self.vocab.title2ids(self.title[i], oovs, oovs_set) + [
                self.vocab[config.EOS_TOKEN]])
        }

    def __len__(self):
        return len(self.text)


def load_data(path, sample=config.sample):
    with open(path, 'r') as f:
        data = json.load(f)
    contents, titles = [], []
    if sample > 0:
        data = data[:sample]
    else:
        data = data[:len(data) // -sample]
    with tqdm(total=len(data)) as pbar:
        for item in data:
            content = re.sub(r'\s+', ' ', item['content'])
            title = re.sub(r'\s+', ' ', item['title'])
            contents.append(list(jieba.cut(content)))
            titles.append(list(jieba.cut(title)))
            pbar.update(1)
    return contents, titles


def replace_oovs(word, vocab):
    return torch.where((word > len(vocab) - 1).to(config.device),
                       torch.full(word.shape, vocab.unk, dtype=torch.long).to(config.device),
                       word.to(config.device))
