import torch
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F

from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader

"""模型统计相关"""
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        z = self.l1(x)
        return z
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        z = self.l2(x)
        return z
    
class LightningModule(pl.LightningModule):
    def __init__(self, encoder, decoder, learning_rate=1e-3, class_weight=None, num_class=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_class = num_class

        self.classifier = None
        self._classifier_input_dim = None

        self.learning_rate = learning_rate 
        if class_weight is  not None:
            self.loss = torch.nn.CrossEntropyLoss(weight=class_weight)
        else:
            self.loss = torch.nn.CrossEntropyLoss()

    def _build_classifier(self, x):
        x = x.view(x.size(0), -1)
        self._classifier_input_dim = x.size(1)

        self.classifier = nn.Sequential(
            nn.Linear(self._classifier_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_class)
        )


    def training_step(self, batch, batch_idx):
        x, y = batch
        x = self.encoder(x)
        x = self.decoder(x)
        if self.classifier is None:
            self._build_classifier(x)
        x_flat = x.view(x.size(0), -1)
        y_hat = self.classifier(x_flat)
        train_loss = self.loss(y_hat, y)
        self.log("train_loss", train_loss)
        return train_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        return optimizer
    
class MNISTDataloader(pl.LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    # 这里创建数据调用的时候，需要设置stage参数，后面训练的时候，pytorch-lightning在训练不同阶段会自动选择相应的数据调用
    def setup(self, stage=None):
        dataset = MNIST(self.data_dir, download=True, transform=transforms.ToTensor())
        self.train_data, self.val_data, self.test_data = random_split(dataset=dataset, lengths=[0.8,0.1,0.1],generator=torch.Generator())

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_data, batch_size=64, shuffle=True)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_data, batch_size=64, shuffle=False)
        return val_dataloader
    
dataset = MNISTDataloader(r'D:\\xyz\\debug_code\\minist\\mnist_dataset\\')
simple_model = LightningModule(Encoder(), Decoder())

# profiler
# trainer = pl.Trainer(profiler="simple", max_epochs=2)
trainer = pl.Trainer(max_epochs=2)
trainer.fit(model=simple_model, datamodule=dataset)