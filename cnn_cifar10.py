import random
from collections import deque
from tqdm import tqdm
from sklearn.manifold import TSNE
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from typing import Callable
from pathlib import Path
from PIL import Image
import Config
import ResNet18


def generate_subset(dataset: Dataset, ratio: float,
                    random_seed: int=0):
    # サブセットの大きさを計算
    size = int(len(dataset) * ratio)

    indices = list(range(len(dataset)))

    # 二つのセットに分ける前にシャッフル
    random.seed(random_seed)
    random.shuffle(indices)

    # セット1とセット2のサンプルのインデックスに分割
    indices1, indices2 = indices[:size], indices[size:]

    return indices1, indices2

'''
dataset: 平均と標準偏差を計算する対象のPyTorchのデータセット
'''
def get_dataset_statistics(dataset: Dataset):
    data = []
    for i in range(len(dataset)):
        # [チャネル数, 高さ, 幅]の画像を取得
        img = dataset[i][0]
        data.append(img)
    data = torch.stack(data)

    # 各チャネルの平均と標準偏差を計算
    channel_mean = data.mean(dim=(0, 2, 3))
    channel_std = data.std(dim=(0, 2, 3))

    return channel_mean, channel_std


'''
data_loader: 評価に使うデータを読み込むデータローダ
model      : 評価対象のモデル
loss_func  : 目的関数
'''
def evaluate(data_loader: Dataset, model: nn.Module,
             loss_func: Callable):
    model.eval()

    losses = []
    preds = []
    for x, y in data_loader:
        with torch.no_grad():
            x = x.to(model.get_device())
            y = y.to(model.get_device())

            y_pred = model(x)

            losses.append(loss_func(y_pred, y, reduction='none'))
            preds.append(y_pred.argmax(dim=1) == y)

    loss = torch.cat(losses).mean()
    accuracy = torch.cat(preds).float().mean()

    return loss, accuracy


'''
t-SNEのプロット関数
data_loader: プロット対象のデータを読み込むデータローダ
model      : 特徴量抽出に使うモデル
num_samples: t-SNEでプロットするサンプル数
'''
def plot_t_sne(data_loader: Dataset, model: nn.Module,
               num_samples: int):
    model.eval()

    # t-SNEのためにデータを整形
    x = []
    y = []
    for imgs, labels in data_loader:
        with torch.no_grad():
            imgs = imgs.to(model.get_device())

            # 特徴量の抽出
            embeddings = model(imgs, return_embed=True)

            x.append(embeddings.to('cpu'))
            y.append(labels.clone())

    x = torch.cat(x)
    y = torch.cat(y)

    # NumPy配列に変換
    x = x.numpy()
    y = y.numpy()

    # 指定サンプル数だけ抽出
    x = x[:num_samples]
    y = y[:num_samples]

    # t-SNEを適用
    t_sne = TSNE(n_components=2, random_state=0)
    x_reduced = t_sne.fit_transform(x)

    # 各ラベルの色とマーカーを設定
    cmap = plt.get_cmap("tab10")
    markers = ['4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D']

    # データをプロット
    plt.figure(figsize=(20, 15))
    for i, cls in enumerate(data_loader.dataset.classes):
        plt.scatter(x_reduced[y == i, 0], x_reduced[y == i, 1],
                    c=[cmap(i / len(data_loader.dataset.classes))],
                    marker=markers[i], s=500, alpha=0.6, label=cls)
    plt.axis('off')
    plt.legend(bbox_to_anchor=(1, 1), fontsize=24, framealpha=0)
    plt.show()


def train_eval():
    
    cnt=0

    config = Config.Config()

    # 入力データ正規化のために学習セットのデータを使って
    # 各チャネルの平均と標準偏差を計算
    dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True,
        transform=T.ToTensor())
    channel_mean, channel_std = get_dataset_statistics(dataset)

    # 画像の整形を行うクラスのインスタンスを用意
    train_transforms = T.Compose((
        T.RandomResizedCrop(32, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=channel_mean, std=channel_std),
    ))
    test_transforms = T.Compose((
        T.ToTensor(),
        T.Normalize(mean=channel_mean, std=channel_std),
    ))

    # 学習、評価セットの用意
    train_dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True,
        transform=train_transforms)
    val_dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True,
        transform=test_transforms)
    test_dataset = torchvision.datasets.CIFAR10(
        root='data', train=False, download=True,
        transform=test_transforms)
    

    # 学習・検証セットへ分割するためのインデックス集合の生成
    val_set, train_set = generate_subset(
        train_dataset, config.val_ratio)

    print(f'Number of samples in the training set: {len(train_set)}')
    print(f'Number of samples in the validation set: {len(val_set)}')
    print(f'Number of samples in the test set: {len(test_dataset)}')

    # インデックス集合から無作為にインデックスをサンプルするサンプラー
    train_sampler = SubsetRandomSampler(train_set)

    # DataLoaderを生成
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
         sampler=train_sampler)
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
         sampler=val_set)
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        )


    # 目的関数の生成
    loss_func = F.cross_entropy

    # 検証セットの結果による最良モデルの保存用変数
    val_loss_best = float('inf')
    model_best = None

    # ResNet18モデルの生成
    model = ResNet18.ResNet18(len(train_dataset.classes))


    # モデルを指定デバイスに転送(デフォルトはGPU)
    model.to(config.device)

    # 最適化器の生成
    optimizer = optim.SGD(model.parameters(), lr=config.lr,
                          momentum=0.9, weight_decay=1e-5)

    # 学習率減衰を管理するスケジューラの生成
    scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[config.lr_drop], gamma=0.1)


    for epoch in range(config.num_epochs):
        model.train()

        with tqdm(train_loader) as pbar:
            pbar.set_description(f'[epoch {epoch + 1}]')
            
            # 移動平均計算用
            losses = deque()
            accs = deque()

            for x, y in pbar:

                # データをモデルと同じデバイスに転送
                x = x.to(model.get_device())
                y = y.to(model.get_device())
                
                # パラメータの勾配をリセット
                optimizer.zero_grad()

                # 順伝播
                y_pred = model(x)

                # 学習データに対する損失と正確度を計算
                loss = loss_func(y_pred, y)
                accuracy = (y_pred.argmax(dim=1) == \
                            y).float().mean()

                # 誤差逆伝播
                loss.backward()

                # パラメータの更新
                optimizer.step()

                # 移動平均を計算して表示
                losses.append(loss.item())
                accs.append(accuracy.item())
                if len(losses) > config.moving_avg:
                    losses.popleft()
                    accs.popleft()
                pbar.set_postfix({
                    'loss': torch.Tensor(losses).mean().item(),
                    'accuracy': torch.Tensor(accs).mean().item()})
            
        # 検証セットを使って精度評価
        val_loss, val_accuracy = evaluate(
            val_loader, model, loss_func)
        print(f'verification : loss = {val_loss:.3f}, '
                f'accuracy = {val_accuracy:.3f}')

        # より良い検証結果が得られた場合、モデルを記録
        if val_loss < val_loss_best:
            val_loss_best = val_loss
            model_best = model.copy()

        # エポック終了時にスケジューラを更新
        scheduler.step()
    cnt+=1

    # テスト
    test_loss, test_accuracy = evaluate(
        test_loader, model_best, loss_func)
    print(f'test: loss = {test_loss:.3f}, '
          f'accuracy = {test_accuracy:.3f}')

    # t-SNEを使って特徴量の分布をプロット
    plot_t_sne(test_loader, model_best, config.num_samples)

    # モデルパラメータを保存
    torch.save(model_best.state_dict(), 'resnet18.pth')


def demo():
    config = Config.Config()

    # 入力データ正規化のために学習セットのデータを使って
    # 各チャネルの平均と標準偏差を計算
    dataset = torchvision.datasets.CIFAR10(
        root='data', train=True, download=True,
        transform=T.ToTensor())
    channel_mean, channel_std = get_dataset_statistics(dataset)

    transforms = T.Compose((
        T.ToTensor(),
        T.Normalize(mean=channel_mean, std=channel_std),
    ))

    # ResNet18モデルの生成とパラメータの読み込み
    model = ResNet18.ResNet18(len(dataset.classes))
    model.load_state_dict(torch.load('resnet18.pth'))

    # モデルを指定デバイスに転送(デフォルトはGPU)
    model.to(config.device)

    model.eval()


    for img_path in Path('classification').glob('*.jpg'):
        img = Image.open(img_path)

        # 画像を整形
        img = transforms(img)

        # バッチ軸の追加
        img = img.unsqueeze(0)

        img = img.to(config.device)

        pred = model(img)

        # 数字表現の予測クラスラベルを取得
        pred = pred[0].argmax()

        print(f'predict: {dataset.classes[pred]}, true: {img_path.stem}')

demo()