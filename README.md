# SPAI + SVD

## 数据索引 CSV 构建
- 构建读取数据集的csv参考`create_true+fake_csv.py`，传一个real目录和一个fake目录，不过这个是我新写的，没试过。
- 原版代码构建的csv是把train和val都放在一个csv里面了，train的时候直接val和train都读这一个csv就行，我不知道我新写的是怎么样

## 主要部分
- 这个代码是用cursor融了两个库做的，不知道跑的时候还会有什么bug
- weights/下面放两个模型的pth，一个是spai的spai.pth，另一个我用的是effort的‘The checkpoint of "CLIP-L14 + our Effort" training on GenImage (sdv1.4)’
- 主要参数都堆在 `train.py` 130 行附近（batch、epoch、SVD 这些）。这里的learning rate没用，直接在 161行optimizer写
- train直接 `python train.py`
- `infer`、`test` 还没改。

## 注意事项
1. 先把 `train_one_epoch` 的 `print_freq` 调小看看loss，有问题的话前几个step就爆了。
2. 如果几个 step 就 nan，基本是 `models/@svd_backbone.py` 72 行附近的 residual 崩了，。
3. 如果loss 卡在 0.69，这个正好是乱猜的loss，这时候奇异矩阵S_residual基本所有值都是一样的，调小learning rate试一下，或者说改大bathcsize和改小svd的rank（不知道）


