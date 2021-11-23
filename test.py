# ML Assignment 3
# LXXHSI007
# part 2
"""
I used the ML assignment 0, Pytorch.py as refernce to code this script
but using a cnn instead of fnn
"""

#imports
import sys
from io import BytesIO
import torch, torchvision
from torch import nn
from torch import optim
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy
import copy
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("No Cuda Available")

T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
train_data = torchvision.datasets.MNIST('./', train=True, download=False, transform=T)
val_data = torchvision.datasets.MNIST('./', train=False, download=False, transform=T)

numb_batch = 64

train_dl = torch.utils.data.DataLoader(train_data = torchvision.datasets.MNIST('./', train=True, download=False, transform=T), batch_size = numb_batch)
val_dl = torch.utils.data.DataLoader(val_data, batch_size = numb_batch)

def net():
    # c
    model = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, 5, padding=0),
        nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(400,  120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    )
    return model

def validate(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        images = images.to(device)
        x = model(images)
        value, pred = torch.max(x,1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == labels)
    return correct*100./total

def train(numb_epoch=3, lr=1e-3, device="cpu"):
    cnn = net().to(device)
    cec = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    for epoch in range(numb_epoch):
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = cnn(images)
            loss = cec(pred, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(validate(cnn, val_dl))
        print('Epoch:', epoch+1, "Accuracy :", accuracy, '%')
    return cnn


def inference(path, model, device):
    ima = Image.open(path)

    with BytesIO() as f:
        ima.save(f, format='JPEG')
        f.seek(0)
        img = Image.open(f)
        img = img.resize((28, 28))
    x = (numpy.expand_dims(numpy.array(img), -1)) / 255.
    with torch.no_grad():
        pred = model(torch.unsqueeze(T(x), axis=0).float().to(device)) # feed the image to the network
        return F.softmax(pred, dim=-1).cpu().numpy() # get the probability value


if __name__ == '__main__':

    lenet = train(4, device=device)
    print("Enter a image address: ")
    for line in sys.stdin:
        if 'exit' == line.rstrip():
            print("Exiting....")
            break
        pred = inference(line.rstrip(), lenet, device=device)
        pred_idx = numpy.argmax(pred)
        print(f"Predicted: {pred_idx}, Prob: {pred[0][pred_idx] * 100} %")
        print("Enter a image address: ")