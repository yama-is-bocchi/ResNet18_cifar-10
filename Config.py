class Config:
    '''
    ハイパーパラメータとオプションの設定
    '''
    def __init__(self):
        self.val_ratio = 0.2   # 検証に使う学習セット内のデータの割合
        self.num_epochs = 40   # 学習エポック数
        self.lr_drop = 25      # 学習率を減衰させるエポック
        self.lr = 1e-2         # 学習率
        self.moving_avg = 20   # 移動平均で計算する損失と正確度の値の数
        self.batch_size = 32   # バッチサイズ
        self.num_workers = 2   # データローダに使うCPUプロセスの数
        self.device = 'cuda'   # 学習に使うデバイス
        self.num_samples = 200 # t-SNEでプロットするサンプル数