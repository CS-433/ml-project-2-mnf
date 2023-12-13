import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import torchvision.transforms.functional as TF
from torchvision import models
from torch.nn import Module, Conv2d, Parameter, Softmax
from torchvision.transforms import v2

import matplotlib
import matplotlib.pyplot as plt

import os
from metrics import compute_f1_score

def train_epoch(model, optimizer, scheduler, criterion, train_loader, epoch, device):
    model.train()
    #model.use_checkpointing()
    loss_history = []
    accuracy_history = []
    f1_score_history = []
    lr_history = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.float().to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # pred = (output>0.5)
        # correct = (pred==target).sum().item()
        loss_float = loss.item()
        f1_float = compute_f1_score(target, output, 0.5)
        accuracy_float = compute_metrics(target, output, 0.5)

        loss_history.append(loss_float)
        accuracy_history.append(accuracy_float)
        f1_score_history.append(f1_float)
        lr_history.append(scheduler.get_last_lr()[0])
        if batch_idx % (len(train_loader.dataset) // len(data) // 10) == 0:
            print(
                f"Train Epoch: {epoch}-{batch_idx:03d} "
                f"batch_loss={loss_float:0.2e} "
                f"batch_acc={accuracy_float:0.3f} "
                f"batch_f1={f1_float:0.3f} "
                f"lr={scheduler.get_last_lr()[0]:0.3e} "
            )



    return loss_history, accuracy_history, lr_history , f1_score_history


@torch.no_grad()

def validate(model, device, val_loader, criterion):
    model.eval()  # Important: eval mode (affects dropout, batch norm etc)
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item() * len(data)
        pred = (output>0.5)
        correct += compute_metrics(target, pred)
        f1 = compute_f1_score(target, pred)

    test_loss /= len(val_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss,
            correct,
            len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset),
        )
    )
    return test_loss, correct , f1



@torch.no_grad()
def get_predictions(model, device, val_loader, criterion, num=None):
    model.eval()
    points = []
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        pred = output
        data = np.split(data.cpu().numpy(), len(data))
        loss = np.split(loss.cpu().numpy(), len(data))
        pred = np.split(pred.cpu().numpy(), len(data))
        target = np.split(target.cpu().numpy(), len(data))
        points.extend(zip(data, loss, pred, target))

        if num is not None and len(points) > num:
            break

    return points