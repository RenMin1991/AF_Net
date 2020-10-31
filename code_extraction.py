# -*- coding: utf-8 -*-
"""
    feature extraction of netvlad
    Ren Min
    20181206
"""

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import csv

from txt_dataset import TxtDataset
from model import Maxout_VLAD
from ordinalnet_function import translation
import pdb


# parameters
torch.cuda.set_device(6)

# pixels of iris image transform
iris_rot = 32

# number of classes of trainning data
num_class_la_open = 409

# size of mini batch, batch=1 during feature extraction
batch = 1

# name of feature file
code_file = 'code/vlad_la_nrot.csv' 

# path to test data
data_folder_la = '../../../data5/min.ren/iris/CASIA-Iris-Lamp/'

# pre-trained model
load_file = 'checkpoint/maxoutVLAD_la_nrot.pth' 


# define network
model = Maxout_VLAD(2000, 819, num_class_la_open, 800) 
all_data = torch.load(load_file)
model.load_state_dict(all_data['model'])
model = model.cuda()
print model


# pre-process
transform_la = transforms.Compose([
        transforms.Resize(size=[128,128]),
        transforms.ToTensor(),
        transforms.Normalize((0.3506,),(0.1366,))
        ])


# get data
txt_la = '../../../data5/min.ren/iris/CASIA-Iris-Lamp/'
testset_la = TxtDataset(txt=txt_la+'Lamp_test_open.txt', data_folder=data_folder_la, transform=transform_la)

print 'la',len(testset_la)

pdb.set_trace()
test_loader_la = DataLoader(testset_la, batch_size = batch, shuffle=False)


# extraction
model.eval()

float_codes = []

for i, data in enumerate(test_loader_la, 0):
    images, labels = data
    images, labels = images.cuda(), labels.cuda()
    images, labels = Variable(images), Variable(labels)
    
    # rotation, comment this part when trainning without rotation
    #trans = np.random.randint(0,iris_rot)
    #images = translation(images, trans_col=trans, trans_row=0)
        
    _, _, _, _, f_0 = model(images)
    
    #pdb.set_trace()
    float_code = []
    
    for code in f_0[0]:
        float_code.append(code.item())

    float_code.append(labels.item())
    float_codes.append(float_code)
    
    
# save hash codes
pdb.set_trace()
f = open(code_file, 'w')
writer = csv.writer(f)
for f_c in float_codes:
    writer.writerow(f_c)
f.close()

