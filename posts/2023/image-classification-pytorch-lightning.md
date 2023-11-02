---
title: 'Training an Image Classification Model using PyTorch Lightning'
description: 'Learn how to train image classification model using PyTorch Lightning'
date: '2023-10-14'
keywords: pytorch, lightning
layout: post
toc: true
author: ChatGPT
image: https://repository-images.githubusercontent.com/178626720/7eaddebb-17d3-4a19-8fc8-55f2ad6d456a
---

# Training an Image Classification Model Using PyTorch Lightning

Image classification is a core task in the field of computer vision, and PyTorch Lightning makes it easier than ever to build, train, and evaluate image classification models. PyTorch Lightning is a high-level wrapper around PyTorch that simplifies the training process, allowing you to focus on your model and experiment, rather than boilerplate code. In this blog, we will walk you through the process of training an image classification model using PyTorch Lightning.

## Prerequisites

Before we start, make sure you have the following prerequisites:

- **Python:** Ensure you have Python installed on your system.
- **PyTorch:** Install PyTorch as specified on the [official PyTorch website](https://pytorch.org/).
- **PyTorch Lightning:** Install PyTorch Lightning using pip:

```bash
pip install pytorch-lightning
```

- **GPU (optional):** While not mandatory, using a GPU can significantly speed up training, especially for larger models and datasets.

## Dataset Selection

Selecting the right dataset is a crucial first step in image classification. For this blog, we'll use the CIFAR-10 dataset, which is available through the torchvision library. It contains 60,000 32x32 color images in 10 different classes.

```python
import torch
import torchvision
from torchvision import transforms

# Define data transformations
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
```

## Model Definition

With PyTorch Lightning, defining your model is as simple as creating a PyTorch module and subclassing `pl.LightningModule`. Here's an example of a basic Convolutional Neural Network (CNN) for image classification:

```python
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

class Net(pl.LightningModule):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

net = Net()
```

## Data Loaders

PyTorch Lightning simplifies data loading using DataModules. You can create a custom DataModule to encapsulate data loading and preprocessing:

```python
class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # Download the dataset
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

dm = CIFAR10DataModule()
```

## Training

Now, training your image classification model is as simple as initializing a `Trainer` and calling the `fit` method:

```python
trainer = pl.Trainer(max_epochs=5, gpus=1)  # Adjust max_epochs and gpus based on your needs
trainer.fit(net, dm)
```

## Conclusion

Training an image classification model using PyTorch Lightning streamlines the entire process, from data loading to training and evaluation. With its clean, modular structure and high-level abstractions, PyTorch Lightning allows you to focus on building and experimenting with your models. This blog provided a comprehensive guide to help you get started on your image classification journey. Happy experimenting!
