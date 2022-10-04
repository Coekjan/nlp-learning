import json
import math
from typing import Dict, List, Set

import numpy as np
from seqeval.metrics import classification_report

tag_text: Dict[str, str] = {
    'address': 'ADDRESS',
    'book': 'BOOK',
    'company': 'COM',
    'game': 'GAME',
    'government': 'GOV',
    'movie': 'MOVIE',
    'name': 'NAME',
    'organization': 'ORG',
    'position': 'POS',
    'scene': 'SCENE'
}

tag_dict: Dict[str, int] = {
    'O': 0,
    'B-ADDRESS': 1,
    'I-ADDRESS': 2,
    'B-BOOK': 3,
    'I-BOOK': 4,
    'B-COM': 5,
    'I-COM': 6,
    'B-GAME': 7,
    'I-GAME': 8,
    'B-GOV': 9,
    'I-GOV': 10,
    'B-MOVIE': 11,
    'I-MOVIE': 12,
    'B-NAME': 13,
    'I-NAME': 14,
    'B-ORG': 15,
    'I-ORG': 16,
    'B-POS': 17,
    'I-POS': 18,
    'B-SCENE': 19,
    'I-SCENE': 20
}

word_list: List[str] = []
init_prob: np.core.multiarray = None
transition_prob: np.core.multiarray = None
emission_prob: np.core.multiarray = None


def log(x):
    assert x >= 0
    if x == 0:
        return -10
    else:
        return math.log10(x)


def read_data(path: str) -> List[Dict[str, str | List[int] | Dict[str, List[List[int]]]]]:
    with open(path, encoding='utf-8') as data_file:
        data_json = [json.loads(line) for line in data_file.readlines()]
    data: List[Dict[str, str | List[int] | Dict[str, List[List[int]]]]] = []
    word_set: Set[str] = set(word_list)
    for item in data_json:
        text: List[str] = item['text']
        label: Dict[str, Dict[str, List[List[int]]]] = item['label']
        tags: List[str] = ['O'] * len(text)
        words: Dict[str, List[List[int]]] = {}
        for char in text:
            if char not in word_set:
                word_list.append(char)
                word_set.add(char)
        for lab, content in label.items():
            for word, content_range in content.items():
                words[word] = content_range
                for rg in content_range:
                    tags[rg[0]] = 'B-' + tag_text[lab]
                    if rg[1] > rg[0]:
                        tags[rg[0] + 1:rg[1] + 1] = ['I-' + tag_text[lab]] * (rg[1] - rg[0])
        data.append({'text': text, 'tags': [tag_dict[tag] for tag in tags], 'words': words})
    return data


def init_hmm(data: List[Dict[str, str | List[int] | Dict[str, List[List[int]]]]]) -> None:
    global init_prob, transition_prob, emission_prob
    # init init_prob
    init_prob = np.zeros(shape=len(tag_dict))
    head_tag_counts = [0] * len(tag_dict)
    for dat in data:
        head_tag_counts[dat['tags'][0]] += 1
    for i in range(len(tag_dict)):
        init_prob[i] = head_tag_counts[i] / len(data)
    # init transition_prob
    transition_prob = np.zeros(shape=(len(tag_dict), len(tag_dict)))
    trans_counts = np.zeros(shape=(len(tag_dict), len(tag_dict)), dtype=int)
    for dat in data:
        for i in range(len(dat['tags']) - 1):
            trans_counts[dat['tags'][i], dat['tags'][i + 1]] += 1
    trans_counts_row_sum = np.sum(trans_counts, axis=1)
    for i in range(len(tag_dict)):
        for j in range(len(tag_dict)):
            transition_prob[i, j] = trans_counts[i, j] / trans_counts_row_sum[i]
    # init emission_prob
    emission_prob = np.zeros(shape=(len(tag_dict), len(word_list)))
    emission_counts: Dict[int, Dict[str, int]] = {}
    for dat in data:
        for i in range(len(dat['text'])):
            tag: int = dat['tags'][i]
            if tag not in emission_counts.keys():
                emission_counts[tag] = {}
            if dat['text'][i] not in emission_counts[tag].keys():
                emission_counts[tag][dat['text'][i]] = 0
            emission_counts[tag][dat['text'][i]] += 1
    for j in range(0, len(tag_dict)):
        tot = 0
        for k in range(len(word_list)):
            if word_list[k] in emission_counts[j].keys():
                tot += emission_counts[j][word_list[k]]
        for k in range(len(word_list)):
            if word_list[k] in emission_counts[j].keys():
                emission_prob[j, k] = emission_counts[j][word_list[k]] / tot
            else:
                emission_prob[j, k] = 0


def viterbi(line: List[int],
            pi: np.core.multiarray,
            a: np.core.multiarray,
            b: np.core.multiarray) -> (np.float64, List[int]):
    # init
    delta = np.zeros(shape=(len(line), len(tag_dict)))
    psi = np.zeros(shape=(len(line), len(tag_dict)), dtype=int)
    for i in range(len(tag_dict)):
        delta[0, i] = log(pi[i]) + log(b[i, line[0]])
        psi[0, i] = 0
    # iter
    for t in range(1, len(line)):
        for j in range(len(tag_dict)):
            max_prob = -1000000.0
            max_i = -1
            for i in range(len(tag_dict)):
                prob = delta[t - 1, i] + log(a[i, j])
                if prob >= max_prob:
                    max_i = i
                    max_prob = prob
            delta[t, j] = max_prob + log(b[j, line[t]])
            psi[t, j] = max_i
    # path
    res_prob: np.float64 = np.float64(-1000000.0)
    res_path: List[int] = [-1] * len(line)
    for i in range(len(tag_dict)):
        if delta[len(line) - 1, i] >= res_prob:
            res_prob = delta[len(line) - 1, i]
            res_path[len(line) - 1] = i
    for t in range(len(line) - 2, -1, -1):
        res_path[t] = psi[t + 1, res_path[t + 1]]
    return res_prob, res_path


if __name__ == '__main__':
    __train_data = read_data('train.json')
    init_hmm(__train_data)
    __dev_data = read_data('dev.json')
    __cur_word_list_len = len(word_list)
    __fixed_emission_prob = np.concatenate(
        (emission_prob, np.zeros(shape=(len(tag_dict), __cur_word_list_len - np.shape(emission_prob)[1]))),
        axis=1
    )
    __back_tag = {}
    for t, i in tag_dict.items():
        __back_tag[i] = t
    __paths = []
    __trues = []
    for __data in __dev_data:
        __text_index = list(map(lambda c: word_list.index(c), __data['text']))
        # noinspection PyTypeChecker
        __prob, __path = viterbi(__text_index, init_prob, transition_prob, __fixed_emission_prob)
        __paths.append(list(map(lambda v: __back_tag[v], __path)))
        __trues.append(list(map(lambda v: __back_tag[v], __data['tags'])))
    print(classification_report(__trues, __paths))
