import multiprocessing

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchmetrics.functional import f1_score
from transformers import BertForSequenceClassification


class Classifier(pl.LightningModule):
    def __init__(self, hparams, train_dataset, test_dataset):
        super(Classifier, self).__init__()

        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=14,
                                                                   output_attentions=False, output_hidden_states=False)
        self.batch_size = hparams['batch_size']
        self.learning_rate = hparams['lr']
        self.gamma = hparams['gamma']
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def forward(self, batch):
        if len(batch) == 3:
            input_ids, attention_masks, labels = batch['ids'], batch['mask'], batch['target']
            loss, logits = self.model(input_ids, attention_mask=attention_masks, labels=labels, token_type_ids=None,
                                      return_dict=False)
            return loss, logits
        else:
            input_ids, attention_mask = batch['ids'], batch['mask']
            logits = self.model(input_ids, token_type_ids=None, attention_mask=attention_mask)
            return logits[0]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss, logits = self.forward(batch)
        f1 = f1_score(torch.argmax(logits, dim=1).flatten(), batch['target'].flatten(), average="macro", num_classes=14)
        self.log("loss", loss)
        self.log("train_f1", f1)
        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=multiprocessing.cpu_count())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=multiprocessing.cpu_count())
