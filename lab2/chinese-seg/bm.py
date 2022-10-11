from typing import List

import utils
from seg_model import Model


class BM(Model):
    def __do_fmm(self, text: str) -> List[str]:
        output = []
        i = 0
        max_len = min(self.word_max_len, len(text))
        while i < len(text):
            j = min(i + max_len, len(text))
            while j > i:
                if self.is_word(text[i:j]):
                    output.append(text[i:j])
                    i = j
                    break
                j -= 1
            else:
                output.append(text[i])
                i += 1
        return output

    def __do_bmm(self, text: str) -> List[str]:
        output = []
        max_len = min(self.word_max_len, len(text))
        i = len(text)
        while i > 0:
            j = max(i - max_len, 0)
            while i > j:
                if self.is_word(text[j:i]):
                    output.append(text[j:i])
                    i = j
                    break
                j += 1
            else:
                output.append(text[j - 1])
                i -= 1
        output.reverse()
        return output

    def do_seg(self, text: str) -> List[str]:
        fmm_res = self.__do_fmm(text)
        bmm_res = self.__do_bmm(text)
        if fmm_res == bmm_res:
            return fmm_res
        if len(fmm_res) != len(bmm_res):
            return fmm_res if len(fmm_res) < len(bmm_res) else bmm_res
        fmm_single, bmm_single = 0, 0
        for i in range(len(fmm_res)):
            if len(fmm_res[i]) == 1:
                fmm_single += 1
            if len(bmm_res[i]) == 1:
                bmm_single += 1
        return fmm_res if fmm_single < bmm_single else bmm_res


if __name__ == '__main__':
    __bm = BM('pku_training_words.utf8')
    # noinspection DuplicatedCode
    __text = utils.read_data('corpus.txt')
    __gold = utils.read_gold('gold.txt')
    __seg = []
    for __i in range(len(__text)):
        __seg_res = __bm.do_seg(__text[__i])
        print(__seg_res)
        __seg.append(__seg_res)
    assert len(__seg) == len(__gold)
    print('P   = %.5f\n'
          'R   = %.5f\n'
          'F_1 = %.5f' % __bm.evaluate(__seg, __gold))
    __stopwords = utils.read_dict('stopwords.txt')
    __flatten_seg = [__word for __sublist in __seg for __word in __sublist]
    utils.make_word_cloud('../wordcloud/corpus_bm.jpg', __stopwords, __flatten_seg)
