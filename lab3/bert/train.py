import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from classifier import Classifier
from dataset import CustomDataset


def main():
    train_set = pd.read_csv('./train_set.csv', sep='\t').sample(frac=1).reset_index(drop=True)
    test_set = pd.read_csv('./test_a.csv', sep='\t')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CustomDataset(train_set, tokenizer)
    test_dataset = CustomDataset(test_set, tokenizer)

    hparams = dict(batch_size=8, lr=3e-5, gamma=0.8)
    model = Classifier(hparams, train_dataset, test_dataset)
    checkpoint_callback = ModelCheckpoint(monitor='train_f1', dirpath='./', mode='max', filename='best')
    trainer = pl.Trainer(gpus=1, max_epochs=3, log_every_n_steps=1000, callbacks=[checkpoint_callback])
    trainer.fit(model)

    model = Classifier.load_from_checkpoint('./best.ckpt', hparams=hparams, train_dataset=train_dataset,
                                            test_dataset=test_dataset)
    predictions = trainer.predict(model, model.test_dataloader())

    submission = pd.read_csv('./test_a_sample_submit.csv')
    submission['label'] = [p for bat in predictions for p in bat.argmax(dim=1).numpy()]
    submission.to_csv('./submission.csv', index=False)


if __name__ == '__main__':
    main()
