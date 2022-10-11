from abc import abstractmethod
from typing import List, Tuple, Set

import utils


class Model:
    def __init__(self, dict_path: str):
        self.word_dict = utils.read_dict(dict_path)
        self.word_max_len = max(map(len, self.word_dict))

    def is_word(self, text: str) -> bool:
        return text in self.word_dict

    # noinspection PyPep8Naming
    @staticmethod
    def evaluate(res: List[List[str]], gold: List[List[str]]) -> Tuple[float, float, float]:
        n: int = 0
        N: int = sum(len(r) for r in res)
        M: int = sum(len(g) for g in gold)
        for r, g in zip(res, gold):
            if len(g) == 0:
                continue
            l: List[int] = [0]
            s: Set[Tuple[str, int]] = {(g[0], 0)}
            for i in range(1, len(g)):
                l.append(l[i - 1] + len(g[i - 1]))
                s.add((g[i], l[i]))
            l = [0]
            if (r[0], 0) in s:
                n += 1
            for i in range(1, len(r)):
                l.append(l[i - 1] + len(r[i - 1]))
                if (r[i], l[i]) in s:
                    n += 1

        P = n / N
        R = n / M
        F_1 = 2 * P * R / (P + R)
        return P, R, F_1

    @abstractmethod
    def do_seg(self, text: str) -> List[str]:
        pass
