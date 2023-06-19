"""This module implements neural network models and associated helper
functions/classes using Pytorch"""
import os

from sklearn.model_selection import train_test_split

import torch.nn.functional as F
import torch.nn as nn
import torch

import sklearn.metrics
import numpy as np
import tqdm


class PD_Dataset(torch.utils.data.Dataset):
    """Pytorch dataset for persistence diagrams

    Args:
        data (torch.Tensor): torch.Tensor of dataset
        labels (torch.Tensor): torch.Tensor of labels
    """

    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data = data
        self.labels = labels

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_pt = self.data[idx, :]
        label = self.labels[idx]
        return data_pt, label


class NN_Trainer:
    """Class which implements training of machine learning models and saving
    best metrics and model weights

    Args:
        model (nn.Module): Pytorch nn.Module which implements a neural network
    """

    def __init__(self, model: nn.Module):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")

        self.model_save_dir = os.path.join(
            "./machine_learning",
            "models",
            "best_nn_model",
        )
        self.model_fname = os.path.join(self.model_save_dir, "best_model.pth")

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.test_criterion = F.cross_entropy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.batch_size = 1024
        self.epochs = 1024
        self.max_ba = -1

    def fit_split(self, train_data: torch.Tensor, train_label: torch.Tensor) -> None:
        """Fits a neural network model on a single split of a k-fold split"""
        X_train, X_test, y_train, y_test = train_test_split(
            train_data,
            train_label,
            test_size=0.25,
        )
        self.fit(X_train, y_train, X_test, y_test)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
    ) -> None:
        """Fits a neural network on the entire dataset"""
        train_dataset = PD_Dataset(X_train, y_train)
        test_dataset = PD_Dataset(X_test, y_test)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.pbar = tqdm.tqdm(range(self.epochs), desc="Training")
        for epoch in self.pbar:
            self.train_epoch(train_loader, epoch)
            self.test_epoch(test_loader, epoch)

        print(f"Best Balanced Accuracy: {self.max_ba:.3f}")

    def train_epoch(self, loader: torch.utils.data.DataLoader, epoch_num: int) -> None:
        """Trains a single epoch of a neural network"""
        #  pbar = tqdm.tqdm(loader, desc=f"[{epoch_num}/{self.epochs}]")
        self.model.train()
        for data, label in loader:
            data = data.to(self.device).float()
            label = label.to(self.device).long()

            pred = self.model(data)
            loss = self.criterion(pred, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #  pbar.set_postfix({"loss": f"{loss.item():.3f}"})

    def test_epoch(self, loader: torch.utils.data.DataLoader, epoch_num: int) -> None:
        """Tests a a neural network on the test set after an epoch of training"""
        with torch.no_grad():
            self.model.eval()
            #  pbar = tqdm.tqdm(loader, desc=f"Test")
            loss = []
            pred_labels = []
            true_labels = []
            for data, label in loader:
                data = data.to(self.device).float()
                label = label.to(self.device).long()

                pred_logits = self.model(data)
                pred_label = F.sigmoid(pred_logits).argmax(-1)

                pred_labels.append(pred_label.cpu().numpy())
                true_labels.append(label.cpu().numpy())

                loss.append(
                    self.test_criterion(pred_logits, label, reduce=False).cpu().numpy()
                )
            loss = np.concatenate(loss)
            true_labels = np.concatenate(true_labels)
            pred_labels = np.concatenate(pred_labels)

            test_cmat = sklearn.metrics.confusion_matrix(
                y_true=true_labels,
                y_pred=pred_labels,
            )
            test_ba = np.mean(np.diag(test_cmat) / np.sum(test_cmat, axis=1))
            self.pbar.set_postfix(
                {
                    "loss": f"{np.mean(loss):.3f}",
                    "ba": f"{test_ba:.3f}",
                }
            )

            if test_ba > self.max_ba:
                self.max_ba = test_ba
                self.save_model()

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """Runs prediction of the neural network model trained by the NN_Trainer
        class on a torch.Tensor of data. Returns torch.Tensor of predictions
        made by the model

        Args:
            data (torch.Tensor): Tensor of data for predictions

        Returns:
            preds (torch.Tensor): Tensor of predictions made by neural network

        """
        self.model.eval()
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        data = data.to(self.device).float()
        preds = F.sigmoid(self.model(data)).argmax(-1).cpu().numpy()
        return preds

    def save_model(self):
        """Helper function to save model weights"""
        torch.save(self.model.state_dict(), self.model_fname)

    def load_best_model(self):
        """Helper function to load best model weights"""
        self.model.load_state_dict(torch.load(self.model_fname))


class FCNN(nn.Module):
    """Class that implements a neural network model in PyTorch

    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
    """

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.fc_layers = nn.Sequential(
            nn.Linear(self.num_features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(32, 8),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
        )

        self.output_layer = nn.Linear(8, self.num_classes)

    def forward(self, x: torch.Tensor):
        """Forward function for prediction"""
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x


class FCNN_deep(nn.Module):
    """Class that implements a deeper neural network model in PyTorch

    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
    """

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        start_size = 1024
        end_size = 8

        self.fc_layers = [nn.Linear(self.num_features, start_size)]

        while start_size > end_size:
            self.fc_layers.append(nn.Linear(start_size, start_size // 2))
            start_size = start_size // 2

        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.output_layer = nn.Linear(start_size, self.num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor):
        """Forward function for prediction"""
        for layer in self.fc_layers:
            x = F.leaky_relu(layer(x))
            x = self.dropout(x)

        x = self.output_layer(x)
        return x
