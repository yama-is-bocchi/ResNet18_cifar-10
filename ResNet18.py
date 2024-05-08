import torch
from torch import nn
import BasicBlock
import copy

class ResNet18(nn.Module):
    '''
    ResNet18モデル
    num_classes: 分類対象の物体クラス数
    '''
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3,
                                     stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock.BasicBlock(64, 64),
            BasicBlock.BasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            BasicBlock.BasicBlock(64, 128, stride=2),
            BasicBlock.BasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            BasicBlock.BasicBlock(128, 256, stride=2),
            BasicBlock.BasicBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            BasicBlock.BasicBlock(256, 512, stride=2),
            BasicBlock.BasicBlock(512, 512),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # ドロップアウトの追加
        self.dropout = nn.Dropout()

        self.linear = nn.Linear(512, num_classes)

        self._reset_parameters()

    '''
    パラメータの初期化関数
    '''
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Heらが提案した正規分布を使って初期化
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="relu")

    '''
    順伝播関数
    x           : 入力, [バッチサイズ, 入力チャネル数, 高さ, 幅]
    return_embed: 特徴量を返すかロジットを返すかを選択する真偽値
    '''
    def forward(self, x: torch.Tensor, return_embed: bool=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.flatten(1)

        if return_embed:
            return x

        x = self.dropout(x)

        x = self.linear(x)

        return x

    '''
    モデルパラメータが保持されているデバイスを返す関数
    '''
    def get_device(self):
        return self.linear.weight.device

    '''
    モデルを複製して返す関数
    '''
    def copy(self):
        return copy.deepcopy(self)