from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=256):
        self.comment_text = df.text
        self.__data: list[dict] = [{}] * len(self.comment_text)
        for i in range(len(self.comment_text)):
            encode_dict = tokenizer.encode_plus(self.comment_text[i], add_special_tokens=True, truncation=True,
                                                max_length=max_len, padding='max_length',
                                                return_attention_mask=True, return_tensors='pt')
            if 'label' in df:
                self.__data[i] = {
                    'ids': encode_dict['input_ids'].squeeze(0),
                    'mask': encode_dict['attention_mask'].squeeze(0),
                    'target': df.label[i],
                }
            else:
                self.__data[i] = {
                    'ids': encode_dict['input_ids'].squeeze(0),
                    'mask': encode_dict['attention_mask'].squeeze(0),
                }
            print('dataset encoding: %.2f%%' % (i / len(self.comment_text) * 100), end='\r')
        print('\nencoding done')

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, i):
        return self.__data[i]
