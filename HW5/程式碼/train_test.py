import torch
import torch.nn as nn
from torch.autograd import Variable
from models import LSTM
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from preprocess import data_split,normalize_data, shuffle
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DATASET_ROOT = './'
path = "./rnn.pth"

df = pd.read_csv("./STT.csv", index_col = 0)
# STT company copy
STT = df[df.symbol == 'STT'].copy()
# print(STT)

STT_new = normalize_data(STT)
# print(STT_new)

past = 15

x_train, y_train, x_test, y_test = data_split(STT_new, past)

x_train, y_train = shuffle(x_train, y_train)


INPUT_SIZE = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 1
OUTPUT_SIZE = 1

learning_rate = 0.001
num_epochs = 50

rnn = LSTM(input_dim=INPUT_SIZE,hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_dim=OUTPUT_SIZE)
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

rnn.to(device)
rnn.train()
print_loss = []

for epoch in range(1, num_epochs+1):
    for inputs, label in zip(x_train, y_train):
        inputs = torch.from_numpy(inputs).type(torch.Tensor).to(device)
        label = torch.from_numpy(np.array(label)).type(torch.Tensor).view(-1).to(device)
        # print(label)
        optimizer.zero_grad()

        output = rnn(inputs) # forward   
        # print(label)
        loss = criterion(output, label) # compute loss
        print_loss.append(loss)

        loss.backward() #back propagation
        optimizer.step() #update the parameters

    print('epoch {}, loss {}'.format(epoch,loss.item()))
# above for train
torch.save(rnn.state_dict(), path)

result = []
with torch.no_grad():
    for inputs, label in zip(x_test,y_test):
        inputs = torch.from_numpy(inputs).type(torch.Tensor).to(device)
        label = torch.from_numpy(np.array(label)).type(torch.Tensor).view(-1).to(device)
        output = rnn(inputs)    
        result.append(output)
result = np.array(result)
# above for test


plt.plot(y_test, color='blue', label='Actual')
plt.plot(result, color='red', label='Prediction')
plt.title('STT')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')
plt.show()
#print (X_train.shape, y_train.shape,X_test.shape,y_test.shape)

