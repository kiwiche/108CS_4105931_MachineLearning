import numpy as np
import scipy.io as sio 
import os
import cv2
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import pprint
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_labels(loc):
    annos = sio.loadmat(loc)
    _, total_size = annos["annotations"].shape
    print("total sample size is", total_size)
    labels = np.zeros((total_size, 5))
    for i in range(total_size):
        path = annos["annotations"][:,i][0][5][0].split(".")
        id = int(path[0]) - 1
        for j in range(5):
            labels[id, j] = int(annos["annotations"][:,i][0][j][0])
    return labels
    
labels = get_labels('./devkit/cars_train_annos.mat')
print(labels[0])
pprint.pprint(labels)

image_names = os.listdir('./cars_train')
os.mkdir('./train')

for i in tqdm(image_names):
    image = Image.open('./cars_train/'+ i)
    filename = i.split('.')[0]
    y = labels[int(filename)-1]
    dirname = str(int(y[4]))
    if not os.path.exists('./train/'+dirname):
        os.mkdir('./train/'+dirname)

    image = image.crop((y[0], y[1], y[2], y[3]))
    image.save('./train/'+dirname+'/'+i)

image_names = os.listdir('cars_test')
labels = get_labels('./devkit/cars_test_annos_withlabels.mat')
os.mkdir('./test')

for i in tqdm(image_names):
    image = Image.open('cars_test/'+i)
    filename = i.split('.')[0]
    y = labels[int(filename) - 1]
    dirname = str(int(y[4]))

    if not os.path.exists('./test/'+dirname):
        os.mkdir('./test/'+dirname)

    image = image.crop((y[0], y[1], y[2], y[3]))
    image.save('./test/'+dirname+'/'+i)

print('Finish!')

