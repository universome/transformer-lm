import os
import math
import random
import pickle
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext import data
from torchtext.data import Field, Example, Dataset
from firelab import BaseTrainer

from src.models.transformer_lm import TransformerLM
from src.utils import itos_many
from src.optims.triangle_adam import TriangleAdam
from src.inference import InferenceState


class LMTrainer(BaseTrainer):
    def __init__(self, config):
        super(LMTrainer, self).__init__(config)

    def init_dataloaders(self):
        print('Initializing dataloaders')

        project_path = self.config.firelab.project_path
        data_path_train = os.path.join(project_path, self.config.data.train)
        data_path_val = os.path.join(project_path, self.config.data.val)

        data_train = open(data_path_train).read().splitlines()
        data_val = open(data_path_val).read().splitlines()[:self.config.get('val_set_size')]

        field = Field(init_token='<bos>', eos_token='<eos>', batch_first=True)

        train_examples = [Example.fromlist([x], [('text', field)]) for x in data_train]
        val_examples = [Example.fromlist([x], [('text', field)]) for x in data_val]

        train_ds = Dataset(train_examples, [('text', field)])
        val_ds = Dataset(val_examples, [('text', field)])
        field.build_vocab(train_ds)

        self.vocab = field.vocab
        self.train_dataloader = data.BucketIterator(train_ds, self.config.hp.batch_size, repeat=False)
        self.val_dataloader = data.BucketIterator(val_ds, batch_size=self.config.hp.batch_size, shuffle=False, repeat=False)

        print('Dataloaders initialized!')

    def init_models(self):
        self.model = TransformerLM(self.config.hp.transformer, self.vocab)
        self.model = self.model.to(self.config.firelab.device_name)

    def init_criterions(self):
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.config.firelab.device_name)

    def init_optimizers(self):
        # self.optim = TriangleAdam(self.model.parameters(), self.config.hp.get('optim', {}))
        self.optim = torch.optim.Adam(self.model.parameters())

    def train_on_batch(self, batch):
        loss = self.loss_on_batch(batch)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.writer.add_scalar('Train loss', loss.item(), self.num_iters_done)

    def loss_on_batch(self, batch):
        sents = batch.text.to(self.config.firelab.device_name)

        inputs = sents[:, :-1] # feed everything, except the last <eos> (or <pad>) tokens
        targets = sents[:, 1:].contiguous().view(-1)

        preds = self.model(inputs)
        preds = preds.view(-1, len(self.vocab))

        loss = self.criterion(preds, targets)

        return loss

    def validate(self):
        sources = []
        generated = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                input = batch.text.to(self.config.firelab.device_name)
                _, input_state = self.model(input, return_state=True)

                results = InferenceState({
                    "model": self.model.cached_forward,
                    "inputs": input_state,
                    "vocab": self.vocab,
                    "device": self.config.firelab.device_name,
                    "max_len": 55,
                    "is_inputs_update_enabled": True,
                    "inputs_batch_dim": 1,
                }).inference()

                results = [x.cpu().numpy().tolist() for x in results]
                results = itos_many(results, self.vocab, sep='')

                generated.extend(results)
                sources.extend(itos_many(batch.text, self.vocab, sep=''))

        texts = ['`{} => {}`'.format(s,g) for s,g in zip(sources, generated)]
        text = '\n\n'.join(texts)

        self.writer.add_text('Samples', text, self.num_iters_done)
