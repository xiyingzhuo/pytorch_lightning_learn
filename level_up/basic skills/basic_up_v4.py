import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self, freeze_backbone=True, unfreeze_after_epochs=5, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()  # 保存所有参数到self.hparams
        
        # 初始化模型
        backbone = models.resnet50(weights="DEFAULT")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.classifier = nn.Linear(backbone.fc.in_features, 10)
        
        # 初始冻结
        self._freeze_backbone(self.hparams.freeze_backbone)
        
    def _freeze_backbone(self, freeze=True):
        """冻结或解冻主干网络"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze
        
        if freeze:
            self.feature_extractor.eval()
        else:
            self.feature_extractor.train()
    
    def on_train_epoch_start(self):
        """在每个训练epoch开始时检查是否解冻"""
        if (hasattr(self.hparams, 'unfreeze_after_epochs') and 
            self.hparams.unfreeze_after_epochs is not None and 
            self.current_epoch >= self.hparams.unfreeze_after_epochs):
            
            self._freeze_backbone(False)
            print(f"Epoch {self.current_epoch}: 解冻主干网络进行微调")
    
    def forward(self, x):
        if self.hparams.freeze_backbone:
            with torch.no_grad():
                representations = self.feature_extractor(x).flatten(1)
        else:
            representations = self.feature_extractor(x).flatten(1)
        return self.classifier(representations)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# 使用示例
if __name__ == "__main__":
    # 创建模型，指定解冻时机
    model = ImagenetTransferLearning(
        freeze_backbone=True,
        unfreeze_after_epochs=3,  # 第3个epoch后解冻
        learning_rate=1e-3
    )
    
    # 验证参数访问
    print("unfreeze_after_epochs:", model.hparams.unfreeze_after_epochs)  # 输出: 3
    print("freeze_backbone:", model.hparams.freeze_backbone)              # 输出: True
    print("learning_rate:", model.hparams.learning_rate)                  # 输出: 0.001