import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, Precision, F1Score, Recall
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# 模型构建
# 网络构建，前向传播，模型初始化
class Model(pl.LightningModule):
    def __init__(self, num_class, init_model=True):
        super().__init__()
        self.num_class = num_class
        self.build_classification_model()

        if init_model:
            self._init_model()


    def build_classification_model(self):
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),
            nn.Dropout2d(0.5),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4,4))

        # 这里不确定线性层输入维度，暂时将分类器置为None
        self.classifier = None
        self._classifier_dim = None

    def _init_model(self):
        for m,module in enumerate(self.modules()):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
                
    def _build_classifier(self, x):
        x = x.view(x.size(0), -1)
        self._classifier_dim = x.size(1)

        if self.classifier is None:
            self.classifier = nn.Sequential(
                nn.Linear(self._classifier_dim, 512),
                nn.ReLU(),
                nn.Linear(512, self.num_class)
            )

        self._init_classifier()

    def _init_classifier(self):
        for m, module in enumerate(self.classifier.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.adaptive_pool(x)

        if self.classifier is None:
            self._build_classifier(x)

        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y



# 数据集构建，依旧用mnist数据集
class MNISTDataset(pl.LightningDataModule):
    def __init__(self, data_dir):
        super().__init__()
        self.data_dir = data_dir

    # 用于区分不同阶段的数据处理函数，有个stage参数，在训练的时候才清楚要用哪个数据
    def setup(self, stage):
        mnist_transform = transforms.ToTensor()
        mnist = MNIST(self.data_dir, download=True, transform=mnist_transform)

        self.train_data, self.val_data, self.test_data= random_split(
            mnist, lengths=[0.8, 0.1, 0.1], generator=torch.Generator()
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(dataset=self.train_data, batch_size=64, shuffle=True)
        return train_dataloader
    
    def val_dataloader(self):
        val_dataloader = DataLoader(dataset=self.val_data, batch_size=32, shuffle=False)
        return val_dataloader
    
    def test_dataloader(self):
        test_dataloader = DataLoader(dataset=self.test_data, batch_size=32, shuffle=False)
        return test_dataloader
    


# mnist分类
class MNISTClassificationModel(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = nn.CrossEntropyLoss()

        # 定义指标
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        
        # self.train_precision = Precision(num_classes=num_classes, average='macro')
        # self.val_precision = Precision(num_classes=num_classes, average='macro')
        # self.test_precision = Precision(num_classes=num_classes, average='macro')

        # self.train_recall = Recall(num_classes=num_classes, average='macro')
        # self.val_recall = Recall(num_classes=num_classes, average='macro')
        # self.test_recall = Recall(num_classes=num_classes, average='macro')

        # self.train_f1 = F1Score(num_classes=num_classes, average='macro')
        # self.val_f1 = F1Score(num_classes=num_classes, average='macro')
        # self.test_f1 = F1Score(num_classes=num_classes, average='macro')

        # self.save_hyperparameters()

    def _compute_loss(self, y_hat, y):
        loss = self.loss_fn(y_hat, y)
        return loss

    def model_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        pred = torch.argmax(y_hat, dim=1)
        return loss, y_hat, pred

    def training_step(self, batch, batch_idx):
        x,y = batch
        loss, y_hat, pred = self.model_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)

        # 计算训练集准确率
        self.train_accuracy.update(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        loss, y_hat, pred = self.model_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        
        # 计算验证集准确率
        self.val_accuracy.update(y_hat, y)

    def test_step(self, batch, batch_idx):
        x,y = batch
        loss, y_hat, pred = self.model_step(batch, batch_idx)
        
        # 计算测试集准确率
        self.test_accuracy.update(y_hat, y)
     
    def on_train_epoch_end(self):
        train_acc = self.train_accuracy.compute()
        self.log("train accuracy", train_acc)
        
        # 训练指标清零，也可以在epoch_start使用
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        val_acc = self.val_accuracy.compute()
        self.log("val accuracy", val_acc, prog_bar=True)
        
        # 训练指标清零，也可以在epoch_start使用
        self.val_accuracy.reset() 

    def on_test_epoch_end(self):
        test_acc = self.test_accuracy.compute()
        self.log("test accuracy", test_acc, prog_bar=True)
        
        # 训练指标清零，也可以在epoch_start使用
        self.test_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    


if __name__ == "__main__":
    
    mnist_dataloader = MNISTDataset(r'D:\\xyz\\debug_code\\minist\\mnist_dataset\\')

    # # model train
    # logger = TensorBoardLogger(
    #     save_dir="./basic_up_sum/tensorboard",  # 日志保存根目录
    #     name="basic_up_sum", # 实验名称
    #     version=None,
    #     default_hp_metric=True # 超参数记录
    # )

    # modelckpt = ModelCheckpoint(
    #     dirpath="./basic_up_sum/model_checkpoint/", # 自动创建目录"
    #     filename="mnist-{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}",
    #     every_n_epochs=1,
    #     save_last=True,
    #     monitor="val_loss",
    #     save_top_k=3,
    #     verbose=True  # 在训练终端显示模型保存的详细信息
    # )

    # earlystop = EarlyStopping(
    #     monitor="val_loss",
    #     min_delta=0.01,
    #     mode="min",
    #     verbose=True
    # )

    # model = Model(10)
    # mnist_classification_model = MNISTClassificationModel(model, 10, 1e-3)

    # trainer = pl.Trainer(
    #     max_epochs=2,
    #     callbacks=[modelckpt, earlystop],
    #     logger=logger,
    #     enable_checkpointing=True
    # )
    # trainer.fit(model=mnist_classification_model, datamodule=mnist_dataloader)


    # model predict

    # usage 1：通过load_from_checkpoint调用
    model = Model(10)
    # load_from_checkpoint 是一个类方法，需要通过类去调用
    # load_from_checkpoint，需要传入模型的构造参数
    mnist_predict_model = MNISTClassificationModel.load_from_checkpoint(
        checkpoint_path=r"D:\xyz\debug_code\pytorch_lightning\level_up\basic skills\basic_up_sum\model_checkpoint\last.ckpt",
        model=model,
        num_classes = 10,
        learning_rate=1e-3,
        strict=False,
    )

    print("model struct:", mnist_predict_model)
    print("submodule struct: ")
    for name, model in mnist_predict_model.named_children():
        print(f"{name}, {type(model)}")

    # # usage2：先torch.load()加载权重，然后load_state_dict逐个模块加载
    # mnist_predict_model = MNISTClassificationModel(model, 10, 1e-3)
    # checkpoint = torch.load(r"D:\xyz\debug_code\pytorch_lightning\level_up\basic skills\basic_up_sum\model_checkpoint\last.ckpt")
    # mnist_predict_model_state_dict = checkpoint["state_dict"]
    # mnist_predict_model.load_state_dict(mnist_predict_model_state_dict, strict=False)
    # print("model struct:", mnist_predict_model)
    # print("submodule struct: ")
    # for name, model in mnist_predict_model.named_children():
    #     print(f"{name}, {type(model)}")


    def MNISTPredictModel(model, x):
        model.eval()

        if model.model.classifier is None:
            temp_input = torch.randn(1,1,28,28)
            temp_label = torch.tensor([0])
            temp_batch = (temp_input, temp_label)

            with torch.no_grad():
                # 走一遍训练步骤，触发_build_classifier构建
                # _ = model.training_step(temp_batch, batch_idx=0)
                
                # 走模型的forward函数
                _ = model.model(temp_input)

        with torch.no_grad():
            y_hat = model.model(x)
            return y_hat
    
    # 模型预测
    x = torch.randn(1,1,28,28)
    mnist_predict_model.eval()
    y_hat = MNISTPredictModel(mnist_predict_model, x)
    y_hat = torch.argmax(y_hat, dim=1)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y_hat.shape}")
    print(f"预测类别: {y_hat.item()}")
