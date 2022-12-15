from tqdm import trange

from dataset import *
from model import PGN

import jieba
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def predict(model, vocab, text, max_len=20):
    model = model.to(config.device)
    model.eval()
    dec_words = []
    with torch.no_grad():
        src, oovs, _ = vocab.text2ids(text)
        src_lengths = torch.tensor([len(src)])
        src: torch.Tensor = torch.tensor(src).reshape(1, -1).to(config.device)
        enc_output, prev_hidden = model.encoder(replace_oovs(src, vocab), src_lengths)
        dec_input = torch.tensor([vocab[config.BOS_TOKEN]]).to(config.device)
        for t in range(max_len):
            coverage_vector = torch.zeros_like(src, dtype=torch.float32).to(config.device)
            dec_output, prev_hidden, attention_weights, p_gen, coverage_vector = model.decoder(dec_input, prev_hidden,
                                                                                               enc_output, src_lengths,
                                                                                               coverage_vector)
            dec_output = PGN.get_final_distribution(src, p_gen, dec_output, attention_weights, len(oovs)).argmax(-1)
            token_id = dec_output.item()
            if dec_output.item() == vocab[config.EOS_TOKEN]:
                dec_words.append(config.EOS_TOKEN)
                break
            elif token_id < len(vocab):
                dec_words.append(vocab.id2tk[token_id])
            elif token_id < len(vocab) + len(oovs):
                dec_words.append(oovs[token_id - len(vocab)])
            else:
                dec_words.append(config.UNK_TOKEN)
            dec_input = replace_oovs(dec_output, vocab)
    return dec_words


class GetEvalIndex:
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1

    def get_rouge(self, source, target, unit='char'):
        if unit == 'word':
            source = jieba.cut(source)
            target = jieba.cut(target)
        source, target = ' '.join(source), ' '.join(target)
        score = self.rouge.get_scores(hyps=target, refs=source)
        return score[0]['rouge-1']['f'], score[0]['rouge-2']['f'], score[0]['rouge-l']['f']

    def get_bleu(self, source, target, unit='char'):
        if unit == 'word':
            source = jieba.cut(source)
            target = jieba.cut(target)
        source, target = ' '.join(source), ' '.join(target)
        bleu = sentence_bleu(
            references=source.split(' '),
            hypothesis=target.split(' '),
            smoothing_function=self.smooth
        )
        return bleu


def main():
    test_text, test_title = load_data(config.test_data_path, -1)
    vocab = torch.load(config.vocab_save_path)
    model = PGN(vocab)
    model.load_state_dict((torch.load(config.model_save_path, map_location=config.device)))
    print(model)

    matrix = GetEvalIndex()
    rouge_1 = 0
    rouge_2 = 0
    rouge_l = 0
    bleu = 0
    for i in trange(len(test_text)):
        title_pred = ''.join(predict(model, vocab, test_text[i]))
        title_real = ''.join(test_title[i])
        _rouge_1, _rouge_2, _rouge_l = matrix.get_rouge(title_real, title_pred)
        rouge_1 += _rouge_1
        rouge_2 += _rouge_2
        rouge_l += _rouge_l
        bleu += matrix.get_bleu(title_real, title_pred)
    rouge_1 /= len(test_title)
    rouge_2 /= len(test_title)
    rouge_l /= len(test_title)
    bleu /= len(test_title)
    print("avg rouge_1: ", rouge_1)
    print("avg rouge_2: ", rouge_2)
    print("avg rouge_l: ", rouge_l)
    print("avg bleu: ", bleu)


if __name__ == '__main__':
    main()
