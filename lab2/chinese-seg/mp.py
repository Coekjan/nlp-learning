import math
from queue import PriorityQueue
from typing import List, Tuple

import utils
from seg_model import Model


class MP(Model):
    @staticmethod
    def __dijkstra(graph: List[List[Tuple[int, str]]], source: int) -> List[int]:
        pq = PriorityQueue()
        dist = [math.inf for _ in range(len(graph))]
        prev = [-1 for _ in range(len(graph))]
        visit = set()
        dist[source] = 0
        pq.put((dist[source], source))
        while not pq.empty():
            _, u = pq.get()
            if u not in visit:
                for v, _ in graph[u]:
                    alt = dist[u] + 1
                    if alt < dist[v]:
                        dist[v] = alt
                        prev[v] = u
                    pq.put((dist[v], v))
                visit.add(u)
        return prev

    def do_seg(self, text: str) -> List[str]:
        output = []
        graph = [[(i + 1, text[i])] for i in range(len(text))] + [[]]
        i, j = 0, min(self.word_max_len, len(text))
        while i < j - 1:
            if self.is_word(text[i:j]):
                graph[i].append((j, text[i:j]))
            if j >= len(text):
                j = j - i - 1
                i = 0
            else:
                i += 1
                j += 1
        prev = MP.__dijkstra(graph, source=0)
        target = len(graph) - 1
        while prev[target] != -1:
            out = graph[prev[target]]
            for t, s in out:
                if t == target:
                    output.append(s)
                    break
            target = prev[target]
        output.reverse()
        return output


if __name__ == '__main__':
    __mp = MP('pku_training_words.utf8')
    # noinspection DuplicatedCode
    __text = utils.read_data('corpus.txt')
    __gold = utils.read_gold('gold.txt')
    __seg = []
    for __i in range(len(__text)):
        __seg_res = __mp.do_seg(__text[__i])
        __seg.append(__seg_res)
    assert len(__seg) == len(__gold)
    print('P   = %.5f\n'
          'R   = %.5f\n'
          'F_1 = %.5f' % __mp.evaluate(__seg, __gold))
    __stopwords = utils.read_dict('stopwords.txt')
    __flatten_seg = [__word for __sublist in __seg for __word in __sublist]
    utils.make_word_cloud('corpus_mp.jpg', __stopwords, __flatten_seg)
