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
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.dataloaders = dataloaders
        self.model_save_path = model_save_path
        self.phases = ["train", "test"]
        self.loss = {"train": [], "test": [], "validation": []}
        self.accuracy = {"train": [], "test": [], "validation": []}
        self.best_epoch = 0

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
                    print("|", end="")
                    batch_X, batch_len, batch_y = batch
                    # forward propogation
                    y_pred = self.model((batch_X, batch_len))
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
                    self.save_model(model_name="ag_news_gru")
        
    def predict_step(self, dataloader=None, if_y=False):
        self.model.eval()
        for n_batches, batch in enumerate(dataloader):
            batch_X, batch_len, batch_y = batch
            # forward propogation
            y_pred = self.model((batch_X, batch_len))
            if n_batches == 0:
                predictions = y_pred
                if if_y:
                    all_y = batch_y
            else:
                predictions = torch.cat((predictions, y_pred))
                if if_y:
                    all_y = torch.cat((all_y, batch_y))

        if if_y:
            return all_y, predictions
        else:
            return predictions

    def save_model(self, model_name):
        if self.best_epoch != np.argmin(self.loss["test"]):
            self.best_epoch = np.argmin(self.loss["test"])
            print(f"Best epoch : {self.best_epoch}")
            print(f"Saving model : {self.model_name} at {self.model_save_path}")
            file_name = os.path.join(self.model_save_path, model_name)
            torch.save(model.state_dict(), f=file_name)

    def load_model(self, model_name):
        file_name = os.path.join(self.model_save_path, model_name)
        model_state_dict = torch.load(file_name)
        self.model.load_state_dict(model_state_dict)


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.w_h = nn.Linear(in_features=input_dim, out_features=input_dim)
        self.v = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        alpha = F.softmax(self.v(F.tanh(x)), dim=1)
        alpha_repeated = alpha.repeat(1, 1, self.input_dim)
        effective_x = x * alpha_repeated
        return effective_x, alpha
