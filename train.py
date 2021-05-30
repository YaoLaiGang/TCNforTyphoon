import torch
from torch.utils.data import SubsetRandomSampler  # 无放回按照给定索引列表采样样本元素
from typhoon_dataset import TyphoonDataset
from model import TCN
import numpy as np
import torch.nn as nn
import os

win_size = 1
batch_size = 32
validata_split = 808
shuffle_dataset = True
random_seed = 36
features = 32
output_size = 2
num_epochs = 3
best_val = None
use_gpu = torch.cuda.is_available()


def to_np(x):
    return x.cpu().data.numpy()


def distance_loss(pre, true):
    """
    距离损失，这里先使用欧氏距离，当然可能会考虑极坐标
    这里的pre,true分别代表了预测值和训练值，都是x、y的变化量
    """
    dealt = torch.pow(pre-true, 2)
    length = pre.shape[0]
    return torch.sum(torch.sqrt(torch.sum(dealt.cpu(), dim=1))).item() / length


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    for step, (x, y) in enumerate(dataloader):
        x = x.cuda() if use_gpu else x
        y = y.cuda() if use_gpu else y
        output = model(x)
        loss = distance_loss(output, y)
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train(epoch, model, optimizer, dataLoader):
    model.train()
    print("epoch {}".format(epoch + 1))
    total_loss = 0
    step_num = 0
    for step, data in enumerate(dataLoader):
        x, y = data
        x = x.cuda() if use_gpu else x
        y = y.cuda() if use_gpu else y
        # 前向传播
        output = model(x)
        loss = distance_loss(output, y)  # 是否转置并不影响损失函数的计算
        total_loss += loss.item()

        # 反向传播
        model.zero_grad()
        loss.backward()
        optimizer.step()

        step_num += 1
        if step_num % 5 == 0:
            print("epoch {}/{} ========= loss: {:.6f}".format(epoch + 1, step + 1, loss.item()))
    epoch_loss = total_loss / step_num
    print("finish epoch {}, loss: {:.6f}".format(epoch + 1, epoch_loss))
    return epoch_loss


def save_best_val(val_loss, model, optimizer, epoch):
    global best_val
    if not best_val or val_loss < best_val:
        print("model save")
        best_val = val_loss
        if not os.path.isdir("./save"):
            os.mkdir("./save")
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict, 'epoch': epoch}
        torch.save(state, "./save/tcn{}_val{:.6f}.pt".format(epoch, val_loss))


def main():
    dataset = TyphoonDataset()
    # 划分训练集和验证集索引
    indices = list(range(len(dataset)))
    train_indices, val_indices = indices[:-validata_split], indices[-validata_split:]
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(train_indices)
        np.random.shuffle(val_indices)

    # 创建数据采样器（samplers）和加载器(loaders)：
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=5)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=5)

    # 定义模型
    model = TCN(features, output_size, win_size, [25, 25, 25, 25], 3, 0.2, False)

    if use_gpu:
        model = model.cuda()

    # loss optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)  # 初始adam参数

    for i in range(num_epochs):
        train_loss = train(i, model, optimizer, train_loader)
        val_loss = evaluate(model, val_loader)
        print("epoch {}, validate loss: {:.6f}".format(i + 1, val_loss))
        save_best_val(val_loss, model, optimizer, i + 1)

if __name__ == "__main__":
    main()