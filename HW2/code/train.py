import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from pathlib import Path
import copy
from dataset import IMAGE_Dataset
from test import test
import time
import math
import pickle
import matplotlib.pyplot as plt

torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CUDA_DEVICES = 0
DATASET_ROOT = "./train"

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def train():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)
    print(train_set.num_classes)

    model = models.resnet101(pretrained=False)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, 196)

    model = model.cuda(CUDA_DEVICES)
    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 20
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)
    start = time.time()
    losses = []
    acces = []
    sec = []

    for epoch in range(1, num_epochs+1):
        print(f'Epoch: {epoch}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0

        for i, (inputs, labels) in enumerate(data_loader):
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            training_loss += loss.item() * inputs.size(0)
			#revise loss.data[0]-->loss.item()
            training_corrects += torch.sum(preds == labels.data)
			#print(f'training_corrects: {training_corrects}')

        training_loss = training_loss / len(train_set)
        training_acc = training_corrects.double() /len(train_set)

        print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')
        
        print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, epoch / num_epochs), epoch, epoch / num_epochs * 100, training_loss, training_acc))

        losses.append(training_loss)
        acces.append(training_acc)

        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_model_params)
        if epoch % 1 == 0:
            torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')

        corrects = test(f'model-{best_acc:.02f}-best_train_acc.pth')
        sec.append(corrects)
        
    with open('losses.txt', 'wb') as fp:
        pickle.dump(losses, fp)
        
    with open('acces.txt', 'wb') as fp:
        pickle.dump(acces, fp)

    with open('sec.txt', 'wb') as fp:
        pickle.dump(sec, fp)

if __name__ == "__main__":
    train()
