import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

# 模型结构
# pytortch
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 3))
        
    def forward(self, x):
        return self.l1(x)
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 28*28)
        )

    def forward(self, x):
        return self.l1(x)

# pytorch-lightning
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    # training_step一定要返回loss
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss: ", loss)
        return loss

    # add validation step
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)


    # add test step
    def test_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss: ", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
# dataset
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir
    
    # 这里需要添加stage参数，依据训练阶段，pytorch-lightning会自动选择相应的数据
    def setup(self, stage=None):
        dataset = MNIST(self.data_dir, download=True, transform=transforms.ToTensor())
        self.train_data, self.val_data, self.test_data = random_split(
            dataset = dataset,
            lengths=[0.8, 0.1, 0.1],
            generator=torch.Generator()
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(self.train_data, batch_size=64, shuffle=True)
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_data, batch_size=64, shuffle=False)
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloder = DataLoader(self.test_data, batch_size=64, shuffle=False)
        return test_dataloder

dataset = MNISTDataModule(r'D:\\xyz\\debug_code\\minist\\mnist_dataset\\')

# train_dataloader = DataLoader(dataset)

# train the model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

trainer = pl.Trainer()
# print("trainer max_epoch: ", trainer.max_epochs) # 如果不指定max_epochs，lightning版本小于2.0会设置为None，直到触发某些条件才会停止训练，高版本的话，max_epochs=200

# usage method 1
# trainer.fit(model=autoencoder, datamodule=dataset)

# usage method 2：如果按照下面的方法，dataset.setup()不会被自动调用，需要自动调用

# 手动调用 setup 来初始化数据
dataset.setup()
train_dataloder = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()
trainer.fit(model=autoencoder, train_dataloaders=train_dataloder, val_dataloaders=val_dataloader)

trainer.test(model=autoencoder, dataloaders=dataset)

# # pytorch训练过程
# # eliminate the training loop
# optimizer = autoencoder.configure_optimizers()
# autoencoder.train()
# for batch_idx, batch in enumerate(train_dataloader):
#     loss = autoencoder.training_step(batch, batch_idx)

#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
