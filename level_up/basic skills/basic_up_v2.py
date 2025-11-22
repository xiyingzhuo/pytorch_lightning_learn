"""
相比于上一版本，添加日志保存路径指定以及回调函数，其余设置不变
"""

import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

# 日志保存以及回调函数设置
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

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
        self.save_hyperparameters() # 保存所有超参数到日志和检查点
        # self.save_hyperparameters('learning_rate', 'hidden_size') # 如果想保存指定的超参数，必须在__init__方法中指定后才可以保存
    
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



# train the model
autoencoder = LitAutoEncoder(Encoder(), Decoder())


"""
# 实现设置好保存路径
# 日志设置
LOG_DIR = r"./logs/mnist_autoencoder"
CHECKPOINT_DIR = r'./checkpoints/mnist_autoencoder'

# 目录创建
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 日志设置
logger = TensorBoardLogger(
    save_dir="./logs",
    name="mnist_autoencoder",
    version="1.0")

# 配置模型检查点
checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    filename="autoencoder-{epoch:02d}-{val_loss:.2f}",
    monitor="val_loss", # 监控指标
    mode="min", # 监控指标越小越好
    save_top_k=3, # 保存最好的3个模型
    save_last=True, 
    every_n_epochs=1, # 每1个epoch保存一次
    verbose=True # 输出详细的信息和进度
)

"""

# 依据logger和callback的设置来保存日志和模型检测点信息
logger = TensorBoardLogger(
    save_dir='experiments',
    name="mnist_autoencoder",
    version=None # 自动设置日志保存
)

checkpoint_callback = ModelCheckpoint(
    dirpath="model_checkpoint/mnist_autoencoder", # 自动创建目录
    filename="mnist-ae-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    save_top_k=3,
    save_last=True,
    verbose=True
)

# 可选：添加早停回调
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00,
    patience=3,
    verbose=True,
    mode="min"
)

trainer = pl.Trainer(
    max_epochs=5,
    logger=logger,
    callbacks=[checkpoint_callback, early_stop_callback],
    default_root_dir="./lightning_logs",
    enable_checkpointing=True,
    log_every_n_steps=10)
# print("trainer max_epoch: ", trainer.max_epochs) # 如果不指定max_epochs，lightning版本小于2.0会设置为None，直到触发某些条件才会停止训练，高版本的话，max_epochs=200

# # usage method 1
trainer.fit(model=autoencoder, datamodule=dataset)

# 训练完成后查看自动创建的路径结构
print("\n=== 自动创建的路径结构 ===")
print(f"实验日志路径: {logger.log_dir}")
print(f"TensorBoard 命令: tensorboard --logdir={logger.log_dir}")
print(f"最佳模型: {checkpoint_callback.best_model_path}")
print(f"最佳模型分数: {checkpoint_callback.best_model_score}")


trainer.test(model=autoencoder, dataloaders=dataset)

# # usage method 2：如果按照下面的方法，dataset.setup()不会被自动调用，需要自动调用
# # 手动调用 setup 来初始化数据
# dataset.setup()
# train_dataloder = dataset.train_dataloader()
# val_dataloader = dataset.val_dataloader()
# trainer.fit(model=autoencoder, train_dataloaders=train_dataloder, val_dataloaders=val_dataloader)
# trainer.test(model=autoencoder, dataloaders=dataset)





