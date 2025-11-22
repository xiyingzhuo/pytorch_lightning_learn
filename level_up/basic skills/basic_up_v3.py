"""加载预训练模型"""

import torch
import pytorch_lightning as pl
from torch import nn

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.l1(x)
    
class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28)
        )

    def forward(self, x):
        return self.l1(x)

class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.cirterion = nn.functional.mse_loss()

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        x_hat = self.forward(x)
        train_loss = self.cirterion(x_hat, x)
        self.log("train loss", train_loss)
        return train_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self, PATH=None):
        super().__init__()
        input_dims = 32*32*3
        if PATH:
            # init the pretarined LightningModule
            self.feature_extractor = AutoEncoder.load_from_checkpoint(PATH).encoder
            self.feature_extractor.freeze()
            feature_size = 3
        else:
            self.feature_extractor = Encoder(input_dims, latent_dim=128)
            feature_size=128
        

        # the autoencoder outputs a 28*28-dim representation and CIFAR-10 has 10 classes
        self.classifier = nn.Linear(feature_size, 10)

    def forward(self, x):
        x = x.view(x.view(0), -1)
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)