# -*- coding: utf-8 -*-
"""
Float features extraction
MinRen 20181019
"""


import torch as t
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision as tv
#import torchvision.transforms as transforms
import torchvision.models as models

import csv
import time

from txt_dataset import TxtDataset
from model import Maxout_4, Maxout_4_hash, Maxout_4_in
import torchvision_transforms as transforms
from loss import Hash_Loss

# parameters
t.cuda.set_device(0)

batch = 32

checkpoint = 'checkpoint/maxoutVLAD_ms_210.pth'

code_folder = 'hash_codes/_float_code.csv'

data_folder_in = '../../../data5/min.ren/iris/CASIA-Iris-Interval/'


cuda = True
num_class = 184



# define networks
model = Maxout_4_in(num_class)
all_data = t.load(checkpoint)
model.load_state_dict(all_data['model'])
del all_data

if cuda:
    model = model.cuda()
print model



# pre-process
transform_t = transforms.Compose([
        transforms.Resize(size=[128,128]),
        transforms.ToTensor(),
        transforms.Normalize((0.612,),(0.1155,))
        ])



# get data
txt_in = '../../../data5/min.ren/iris/CASIA-Iris-Interval/'
testset_in = TxtDataset(txt=txt_in+'Interval_test.txt', data_folder=data_folder_in, transform=transform_t)

test_loader = DataLoader(testset_in, batch_size = batch, shuffle=False)




# float code extraction
float_codes = []

print 'float code extracting...'

#timing
start = time.time()

model.eval()

for i, data in enumerate(test_loader, 0):
    
    inputs, labels = data
    if cuda:
        inputs = inputs.cuda()
        labels = labels.cuda()
    inputs, labels = Variable(inputs), Variable(labels)

    features, _ = model(inputs)

    for j, codes in enumerate(features.data, 0):
        float_code = []

        for code in codes:
            float_code.append(code)

        float_code.append(labels.data[j])
        float_codes.append(float_code)


# timing
end = time.time()

print 'extraction finished'
print 'time of the extraction',end-start, 's'



# save hash codes
f = open(code_folder, 'w')
writer = csv.writer(f)
for f_c in float_codes:
    writer.writerow(f_c)
f.close()
