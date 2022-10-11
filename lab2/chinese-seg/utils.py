from typing import List, Set

from wordcloud import WordCloud
import matplotlib.pyplot as plt


def read_data(path: str) -> List[str]:
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            lines.append(line.strip('\n'))
    return lines


def read_dict(path: str) -> Set[str]:
    words = set()
    with open(path, 'r', encoding='utf-8') as f:
        for word in f.readlines():
            words.add(word.strip('\n'))
    return words


def read_gold(path: str) -> List[List[str]]:
    gold = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            gold.append(line.strip('\n').split())
    return gold


def make_word_cloud(path: str, stopwords: Set[str], segs: List[str]) -> None:
    sentences = ' '.join(filter(lambda w: w not in stopwords, segs))
    wc = WordCloud(background_color='white', font_path='style/SimSun.ttc', width=2000, height=2000).generate(sentences)
    plt.imshow(wc)
    plt.axis('off')
    plt.savefig(path)
    plt.show()
