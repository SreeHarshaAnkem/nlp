import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
import re
import spacy
import warnings

warnings.filterwarnings("ignore")
import logging

logging.basicConfig(level=logging.DEBUG)
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print("running on gpu!!!")
else:
    print("cpu :(")


class ModelJob:
    def __init__(
        self,
        model,
        dataloaders,
        model_save_path,
        criterion=None,
        optimizer=None,
        n_epochs=None,
        phases=[],
        model_save_name=None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.dataloaders = dataloaders
        self.model_save_path = model_save_path
        self.phases = phases
        self.loss = {"train": [], "test": [], "validation": []}
        self.accuracy = {"train": [], "test": [], "validation": []}
        self.best_epoch = 0
        self.model_save_name = model_save_name

    def train_step(self):
        for epoch in range(1, self.n_epochs + 1):
            print(f"EPOCH: {epoch} out of {self.n_epochs}")
            epoch_loss = 0
            epoch_accuracy = 0
            best_val_accuracy = 0

            for mode in self.phases:
                if mode == "train":
                    self.model.train()
                else:
                    self.model.eval()
                for n_batches, batch in enumerate(self.dataloaders[mode]):
                    if n_batches % 20 == 0:
                        print("|", end="")
                    batch_X, batch_len, batch_y = batch
                    # forward propogation
                    y_pred, state, lengths = self.model((batch_X, batch_len))
                    # compute loss
                    loss_value = self.criterion(y_pred, batch_y)
                    if mode == "train":
                        # zero grad
                        self.optimizer.zero_grad()
                        # back propagation
                        loss_value.backward()
                        # update weights
                        self.optimizer.step()
                    epoch_loss += loss_value
                    n_correct = (batch_y == torch.argmax(y_pred, dim=1)).sum()
                    epoch_accuracy += n_correct / batch_y.shape[0]
                epoch_loss = epoch_loss / (n_batches + 1)
                epoch_accuracy = epoch_accuracy / (n_batches + 1)
                print()
                print(
                    f"\tMODE: {mode} : LOSS: {epoch_loss} : ACCURACY: {epoch_accuracy}"
                )
                self.loss[mode].append(epoch_loss.item())
                self.accuracy[mode].append(epoch_accuracy.item())
                if mode == "test":
                    self.save_model()

    def predict_step(self, dataloader=None, h_t_1=None, if_y=False, mode="predict"):
        self.model.eval()
        for n_batches, batch in enumerate(dataloader):
            if if_y:
                batch_X, batch_len, batch_y = batch
            else:
                batch_X, batch_len = batch
            # forward propogation
            y_pred, state, lens = self.model((batch_X, batch_len), h_t_1, mode)
            if n_batches == 0:
                predictions = y_pred
                if if_y:
                    all_y = batch_y
            else:
                predictions = torch.cat((predictions, y_pred))
                if if_y:
                    all_y = torch.cat((all_y, batch_y))

        if if_y:
            return all_y, predictions,state, lens
        else:
            return predictions,state, lens

    def save_model(self):
        if self.best_epoch != np.argmin(self.loss["test"]) or self.best_epoch == 0:
            self.best_epoch = np.argmin(self.loss["test"])
            print(f"Best epoch : {self.best_epoch+1}")
            print(f"Saving model : {self.model_save_name} at {self.model_save_path}")
            file_name = os.path.join(self.model_save_path, self.model_save_name)
            torch.save(self.model.state_dict(), f=file_name)

    def load_model(self, model_name):
        file_name = os.path.join(self.model_save_path, model_name)
        model_state_dict = torch.load(file_name)
        self.model.load_state_dict(model_state_dict)


class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.w_h = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.v = nn.Linear(in_features=input_dim, out_features=1)
        self.mask_impute = torch.tensor(-9999.0).to(device)

    def forward(self, x):
        (seq, mask) = x
        transformed_x = self.w_h(
            seq.contiguous().view(seq.shape[0] * seq.shape[1], seq.shape[-1])
        )
        context_vector = self.v(F.tanh(transformed_x))
        context_vector = context_vector.view(seq.shape[0], seq.shape[1])
        masked_context = torch.where(mask == 0.0, self.mask_impute, context_vector)
        alpha = F.softmax(masked_context, dim=1)
        alpha_repeated = alpha.unsqueeze(dim=2).repeat(1, 1, self.input_dim)
        effective_x = seq * alpha_repeated
        return torch.sum(effective_x, dim=1), alpha
    
class Vocab:
    def __init__(self, df, min_frequency, vocab_size, field):
        self.df = df
        self.min_frequency = min_frequency
        self.vocab_size = vocab_size
        self.field = field
        self.vocab = self.build_vocab()
        self.vocab["<UNK>"] = self.vocab_size
        
    def build_vocab(self):
        freq_dict = dict()
        for index, row in self.df.iterrows():
            for token in row[self.field]:
                freq_dict[token] = freq_dict.get(token, 0)+1
        freq_dict = [(word, frequency) for word, frequency in freq_dict.items()
                    if frequency >= self.min_frequency]
        freq_dict = sorted(freq_dict, key = lambda x: x[1], reverse=True)
        freq_dict = freq_dict[:self.vocab_size-1]
        freq_dict = dict(freq_dict)
        vocab = {token : idx+1 for idx, (token, _) in enumerate(freq_dict.items())}
        return vocab
