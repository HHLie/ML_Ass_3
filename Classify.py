# ML Assignment 3
# LXXHSI007
# part 2
"""
using the lenet_5 model
"""

# imports
import sys
from io import BytesIO
import torch, torchvision
from torch import nn
import torch.nn.functional as F
import numpy
import copy
from PIL import Image

# check if any cuda's are available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("Using CPU, no cuda found")

# Transform function to tensor
T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# get data file_path of mnist
train_data = torchvision.datasets.MNIST('./', train=True, download=False, transform=T)
val_data = torchvision.datasets.MNIST('./', train=False, download=False, transform=T)

numb_batch = 64
# Load data into
train_dl = torch.utils.data.DataLoader(train_data, batch_size=numb_batch)
val_dl = torch.utils.data.DataLoader(val_data, batch_size=numb_batch)

"""
create a lenet-5 cnn
conv 1 > pooling 1 > conv 2 > pooling 2 > conv 3 > layer 1 > layer 2 > output
i did this in one function, but it can be done in a class like in the Pytorch_example.py
"""


def net():
    # Define the model
    LeNet = nn.Sequential(
        nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Conv2d(6, 16, 5, padding=0), nn.ReLU(),
        nn.AvgPool2d(2, stride=2),
        nn.Flatten(),
        nn.Linear(400, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10)
    )
    return LeNet


# validate and return the accuracy
def validate(model, data):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        temp = model(images.to(device))  # get image from model
        value, pred = torch.max(temp, 1)  # get prediction value
        total += temp.size(0)
        correct += torch.sum(pred.data.cpu() == labels)  # compare prediction to label, if not correct don't increment
        # else increment
    return correct * 100. / total


# train the network
def train(numb_epoch=4, lr=1e-3, d=device):
    cnn = net().to(d)
    cec = nn.CrossEntropyLoss()
    torch.optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    best_acc = 0
    for epoch in range(numb_epoch):
        for i, (images, labels) in enumerate(train_dl):
            images = images.to(d)  # get image
            labels = labels.to(d)  # get label
            torch.optimizer.zero_grad()  # zero the gradient
            p = cnn(images)  # predict output
            loss = cec(p, labels)  # compute loss function
            loss.backward()  # do backwards pass
            torch.optimizer.step()  # compute gradient
        accuracy = float(validate(cnn, val_dl))  # get accuracy
        print('Epoch:', epoch + 1, "\nAccuracy :", round(accuracy, 2), '%')
        if accuracy >= best_acc:  # get global minima
            best_acc = accuracy
            best_model = copy.deepcopy(cnn)
            print("Saving Best Model with Accuracy: ", round(accuracy, 2), "%")
    print("Done")
    print("\nUsing the Best Model with an Accuracy of :", round(accuracy, 2), "%")
    return best_model


# predict function
def predict(path, model, device):
    # open image file
    ima = Image.open(path)
    # reformat file to binary
    with BytesIO() as f:
        ima.save(f, format='JPEG')
        f.seek(0)
        img = Image.open(f)
        # resize to 28 x 28 so it matches the training data size
        img = img.resize((28, 28))
    x = (numpy.expand_dims(numpy.array(img), -1)) / 255.
    # do the inference function
    with torch.no_grad():
        prediction = model(torch.unsqueeze(T(x), axis=0).float().to(device))
        return F.softmax(prediction, dim=-1).cpu().numpy()


if __name__ == '__main__':
    # set number of epochs
    epoch_num = 6

    print("Training Output...")
    lenet = train(epoch_num, d=device)  # train data for epochs_num
    # do loop
    print("Enter a image address: ")
    for line in sys.stdin:  # get path
        if 'exit' == line.rstrip():
            print("Exiting....")
            break
        pred = predict(line.rstrip(), lenet, device=device)  # get the predicton
        index = numpy.argmax(pred)  # get index
        print("Predicted: ", index, " with a Probabilty of ", round(pred[0][index] * 100, 2), "%")
        print("Enter a image address: ")
