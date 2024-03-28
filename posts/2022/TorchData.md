<!-- ---
title: 'TorchData: PyTorch Data loading utility library'
description: Learn how to load image data with TorchData and train an image classifier.

aliases:
- /posts/2022-04-10-TorchData

categories:
- pytorch

date: '2022-04-10'
date-modified: "2023-05-23"

image: https://images.pexels.com/photos/1029635/pexels-photo-1029635.jpeg?auto=compress
keywords: PyTorch, deep learning, data
toc: true
--- -->
# TorchData: PyTorch Data loading utility library
![Photo by Scott Webb from Pexels](https://images.pexels.com/photos/1029635/pexels-photo-1029635.jpeg?auto=compress&cs=tinysrgb&w=630&h=375&dpr=1)

PyTorch 1.11 introduced a new library called `TorchData`, which provides common data loading primitives for constructing flexible and performant data pipelines. `TorchData` promotes composable data loading for code reusability with `DataPipes`.

`DataPipes` is the building block of `TorchData` and works out of the box with PyTorch's `DataLoader`. It can be chained together to form a data pipeline where the data will be transformed by each `DataPipe`.

For example, suppose we have an image dataset in a folder with a CSV mapping of classes, and we want to create a `DataLoader` that returns a batch of image tensors and labels. To do this, we need to take the following steps:

1. Read and parse the CSV.
2. 
    a. Get the image file path.
    b. Decode the label.
3. Read the image.
4. Convert the image to a tensor.
5. Return the image tensor and the label index.

These steps can be chained together using `DataPipes`, where the initial data flows from the first step to the very last, applying transformations at each step.

Now, let's see how to accomplish the same thing using TorchData code.


```python
!pip install torchdata -q
```


```python
import torch
from torchdata.datapipes.iter import (
    FileOpener,
    Filter,
    FileLister,
    Filter,
)


from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
```

We will use CIFAR-10 dataset which has the same structure as we discussed.

From TorchData docs:

> We have implemented over 50 DataPipes that provide different core functionalities, such as opening files, parsing texts, transforming samples, caching, shuffling, and batching. For users who are interested in connecting to cloud providers (such as Google Drive or AWS S3), the fsspec and iopath DataPipes will allow you to do so. The documentation provides detailed explanations and usage examples of each IterDataPipe and MapDataPipe.

TorchData has 50 prebuilt DataPipes which you can use directly. Here we will use use `FileOpener` and `parse_csv` to read the csv data.


```python
ROOT = "/Users/aniket/datasets/cifar-10/train"

csv_dp = FileLister(f"{ROOT}/../trainLabels.csv")
csv_dp = FileOpener(csv_dp)
csv_dp = csv_dp.parse_csv()

for i, e in enumerate(csv_dp):
    if i > 10:
        break
    print(e)
```

    ['id', 'label']
    ['1', 'frog']
    ['2', 'truck']
    ['3', 'truck']
    ['4', 'deer']
    ['5', 'automobile']
    ['6', 'automobile']
    ['7', 'bird']
    ['8', 'horse']
    ['9', 'ship']
    ['10', 'cat']


We don't need the header of csv in our datapipe (`[id, label]`), so we will use the inbuilt `Filter` DataPipe to remove it.


```python
csv_dp = Filter(csv_dp, lambda x: x[1] != "label")
labels = {e: i for i, e in enumerate(set([e[1] for e in csv_dp]))}

for i, e in enumerate(csv_dp):
    if i > 4:
        break
    print(e)
```

    ['1', 'frog']
    ['2', 'truck']
    ['3', 'truck']
    ['4', 'deer']
    ['5', 'automobile']


We have a DataPipe called `csv_dp`, which flows `file id` and `label`. We need to convert the `file id` into a filepath and the `label` into a label index.

We can map functions to the DataPipe and even form a chain of mappings to apply transformations.


```python
def get_filename(data):
    idx, label = data
    return f"{ROOT}/{idx}.png", label


dp = csv_dp.map(get_filename)
for i, e in enumerate(dp):
    if i > 4:
        break
    print(e)
```

    ('/Users/aniket/datasets/cifar-10/train/1.png', 'frog')
    ('/Users/aniket/datasets/cifar-10/train/2.png', 'truck')
    ('/Users/aniket/datasets/cifar-10/train/3.png', 'truck')
    ('/Users/aniket/datasets/cifar-10/train/4.png', 'deer')
    ('/Users/aniket/datasets/cifar-10/train/5.png', 'automobile')



```python
from IPython.display import display


def load_image(data):
    file, label = data
    return Image.open(file), label


dp = dp.map(load_image)

for i, e in enumerate(dp):
    display(e[0])
    print(e[1])
    if i >= 5:
        break
```


    
![png](output_12_0.png)
    


    frog



    
![png](output_12_2.png)
    


    truck



    
![png](output_12_4.png)
    


    truck



    
![png](output_12_6.png)
    


    deer



    
![png](output_12_8.png)
    


    automobile



    
![png](output_12_10.png)
    


    automobile


Finally, we map the datapipe to process image to Tensor and label to index.


```python
def process(data):
    img, label = data
    return to_tensor(img), labels[label]


dp = dp.map(process)
```

If you have come this far, then I have a bonus for you: learn how to train an image classifier using DataPipe and PyTorch Lightning Flash ⚡️.

Flash expects the dataloader to be in the form of a dictionary with keys `input` and `target`. The `input` key should contain the image tensor, and the `target` key should contain the label index.


```python
dp = dp.map(lambda x: {"input": x[0], "target": x[1]})
```

As we discussed that `DataPipes` are fully compatible with DataLoader so this is how you convert a DataPipe to DataLoader.


```python
dl = DataLoader(
    dp,
    batch_size=32,
    shuffle=True,
)
```

Training an Image Classifier with Flash is super easy. Flash provides Deep Learning tasks based APIs that you can use to train your model. Currently, our task is image classification so let's import the ImageClassifier and build our model.


```python
from flash.image import ImageClassifier
import flash
```


```python
model = ImageClassifier(
    num_classes=len(labels), backbone="efficientnet_b0", pretrained=False
)
```

    Using 'efficientnet_b0' provided by rwightman/pytorch-image-models (https://github.com/rwightman/pytorch-image-models).



```python
# Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3)
```

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs



```python
trainer.fit(model, dl)
```

    
      | Name          | Type           | Params
    -------------------------------------------------
    0 | train_metrics | ModuleDict     | 0     
    1 | val_metrics   | ModuleDict     | 0     
    2 | test_metrics  | ModuleDict     | 0     
    3 | adapter       | DefaultAdapter | 4.0 M 
    -------------------------------------------------
    4.0 M     Trainable params
    0         Non-trainable params
    4.0 M     Total params
    16.081    Total estimated model params size (MB)



    Training: 0it [00:00, ?it/s]



```python

```


```python

```


```python

```
