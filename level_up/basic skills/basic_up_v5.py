import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BackboneFinetuning
from pytorch_lightning import Trainer

class ResNetClassifier(pl.LigthtningModule):
    def __init__(self, num_class=10, learning_rete=1e-3):
        super().__init__()
        self.save_hyperparameters()

        resnet = models.resnet50(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_class)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizer(self):
        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.hparams.learninf_rate)
        return optimizer
    
backbone_finetuning = BackboneFinetuning(
    unfreeze_backbone_at_epoch=10,
    lambda_func=lambda epoch:1.5, # 学习率调整策略
    backbone_initial_ration_lr=0.1,
    should_align=True,
    verbose=True
)

model = ResNetClassifier()
trainer = Trainer(callbacks=[backbone_finetuning], max_epochs=10)
