import torch
import torch.nn as nn
from models import VGG16
from dataset import IMAGE_Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from test import test

import copy
import matplotlib.pyplot as plt
import time
import math
import pickle

##REPRODUCIBILITY
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#args = parse_args()
#CUDA_DEVICES = args.cuda_devices
#DATASET_ROOT = args.path
CUDA_DEVICES = 0
DATASET_ROOT = './seg_train'

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
		transforms.Resize((224,224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
	#print(DATASET_ROOT)
	train_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
	data_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=1)
	#print(train_set.num_classes)
	model = VGG16(num_classes=train_set.num_classes)
	model = model.cuda(CUDA_DEVICES)
	model.train()

	best_model_params = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	num_epochs = 50
	start = time.time()
	losses = []
	acces = []
	acc = []
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

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
			# revise loss.data[0]-->loss.item()
			training_corrects += torch.sum(preds == labels.data)
			# print(f'training_corrects: {training_corrects}')

		training_loss = training_loss / len(train_set)
		training_acc = training_corrects.double() /len(train_set)

		# print(training_acc.type())
		# print(f'training_corrects: {training_corrects}\tlen(train_set):{len(train_set)}\n')
		print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')
		print('%s (%d %d%%) %.4f %.4f' % (timeSince(start, epoch / num_epochs),
                                         epoch, epoch / num_epochs * 100, training_loss, training_acc))

		losses.append(training_loss)
		acces.append(training_acc)

		if training_acc > best_acc:
			best_acc = training_acc
			best_model_params = copy.deepcopy(model.state_dict())

		path = f'epoch_train_model.pth'
		
		model.load_state_dict(best_model_params)
		torch.save(model, path)


		corrects = test(path)

		acc.append(corrects)

	print(losses)
	print(acces)
	print(acc)

	with open('losses.txt', 'wb') as fp:
		pickle.dump(losses, fp)

	with open('acces.txt', 'wb') as fp:
		pickle.dump(acces, fp)

	with open('acc.txt', 'wb') as fp:
		pickle.dump(acc, fp)
		
	plt.figure(figsize=(5,5))
	plt.title("loss")
	plt.plot(losses, label="Training")
	plt.xlabel("epochs")
	plt.legend(loc="upper right")

	plt.figure(num=2, figsize=(5,5))
	plt.title("Accuracy")
	plt.plot(acces, label="Training")
	plt.plot(acc, label="Testing")
	plt.xlabel("epochs")
	plt.legend(loc="upper left")
	plt.show()


if __name__ == '__main__':
	train()
