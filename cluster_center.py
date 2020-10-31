# -*- coding: utf-8 -*-
"""
get cluster centers of features
RenMin20181122
"""

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from txt_dataset import TxtDataset
from model import Maxout_VLAD
import pdb

# parameters
torch.cuda.set_device(5)

num_center = 25

num_class_th = 2000
num_class_la = 819
num_class_in = 366
num_class_cs = 800

data_folder_th = '../../../data5/min.ren/iris/CASIA-Iris-Thousand/'

center_file = 'center/maxout_th.pth'

load_file = 'checkpoint/maxout_40.pth'


# define network
pdb.set_trace()

pre_data = torch.load(load_file)
pre_dict = pre_data['model']

model = Maxout_VLAD(num_class_th, num_class_la, num_class_in, num_class_cs)
model_dict = model.state_dict()

pre_dict = {k:v for k,v in pre_dict.items() if k in model_dict}
model_dict.update(pre_dict)
print pre_dict.keys()
model.load_state_dict(model_dict)
del pre_data
del pre_dict

model = model.cuda()
print model


# pre-process
transform_th = transforms.Compose([
        transforms.Resize(size=[128,128]),
        transforms.ToTensor(),
        transforms.Normalize((0.293,),(0.0833,))
        ])


# get data
txt_th = '../../../data5/min.ren/iris/CASIA-Iris-Thousand/'
trainset_th = TxtDataset(txt=txt_th+'Thousand_train.txt', data_folder=data_folder_th, transform=transform_th)
train_loader_th = DataLoader(trainset_th, batch_size = 1, shuffle=True)

num_sample = int(len(trainset_th))
#print 'number of samples:', num_sample

#pdb.set_trace()

# get features
KM = np.zeros((num_sample, 5*5*192))
model.eval()
for i, data in enumerate(train_loader_th, 0):
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    images, labels = Variable(images), Variable(labels)
        
    _, _, _, _, vectors, _ = model(images)

    KM[i,:] = vectors.cpu().detach().numpy()
    

# get pca transformation
pdb.set_trace()
pca = PCA(n_components=512).fit(KM)

com = pca.components_

com = torch.from_numpy(com)

torch.save(com,'fc1_pca_weights.pth')


"""
# get centers
pdb.set_trace()
kmeans = KMeans(n_clusters=num_center, max_iter=10, random_state=0).fit(KM)

centers = kmeans.cluster_centers_

centers = torch.from_numpy(centers)

torch.save(centers, center_file)
"""












