import os
import numpy as np
import logging
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from resnet import ResNet18
import pdb
import math
import  matplotlib.pyplot as plt
from capsule_resnet import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

EPOCH = 135   #
pre_epoch = 0  #
BATCH_SIZE = 64      #
LR = 0.1        #



datasets_path = '/media/D/tianda/IOT/project_data'
# datasets_path = '/home/focus-lee/IOT_project/project_data'
#datasets_path = '/media/D/tianda/IOT_project/project_data'
traincsv = os.path.join(datasets_path, 'train.csv')
valcsv = os.path.join(datasets_path, 'val.csv')
testcsv = os.path.join(datasets_path, 'test.csv')


with open(testcsv) as f:
    test_csv = csv.reader(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = ResNet18().to(device)
#
# #
# criterion = nn.CrossEntropyLoss()  #
# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
net = CapsuleNet().to(device)
# model.load_state_dict(torch.load('epochs/epoch_327.pt'))

print("# parameters:", sum(param.numel() for param in net.parameters()))

optimizer = Adam(net.parameters())



train_label =[]
train_data = []
with open(traincsv) as f:
    train_csv = csv.reader(f)
    for i, (labels,inputs) in enumerate(train_csv):
        if i ==0:
            continue
        train_label.append(labels)
        train_data.append(np.asarray([float(p) for p in inputs.split()]).reshape(48,48))

val_label = []
val_data = []
with open(valcsv) as f:
    val_csv = csv.reader(f)
    for i, (labels, inputs) in enumerate(val_csv):
        #pdb.set_trace()
        if i == 0:
            continue
        val_label.append(labels)
        val_data.append(np.asarray([float(p) for p in inputs.split()]).reshape(48,48))

def get_batch(datas,labels):
    batch = len(datas)
    shp = (batch,1,48,48)
    label = np.zeros(batch)
    data = np.zeros(shp)
    for i in range(batch):
        label[i] = labels[i]
        data[i,:,:] = datas[i]
    return torch.from_numpy(data).float(),torch.from_numpy(label).long()

def augmentation(x, max_shift=2):
#    pdb.set_trace()
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[ :, :, source_height_slice, source_width_slice] = x[ :, :, target_height_slice, target_width_slice]
    return shifted_image.float()

if __name__ == "__main__":
    best_acc = 85  #2
    capsule_loss = CapsuleLoss()

    print("Start Training, Resnet-18!")  #
    with open("capsule_resnet.acc", "w") as f:
        with open("capsule_resnet.log", "w")as f2:
            for epoch in range(pre_epoch, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0

                for i in range(0, len(train_data), BATCH_SIZE):
                    inputs,labels = get_batch(train_data[i:i+BATCH_SIZE],train_label[i:i+BATCH_SIZE])
                    #if i == 0:
                    #    break
                    length = len(inputs)
                    # inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    inputs = augmentation(inputs.float() / 255.0)
                    labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
                    images = Variable(inputs).cuda()
                    labels = Variable(labels).cuda()


                    # forward + backward
                    classes,reconstruct = net(images,labels)
                    #pdb.set_trace()
                    loss = capsule_loss(images, labels, classes, reconstruct)


                    loss.backward()
                    optimizer.step()

                    sum_loss = loss
                    _, predicted = torch.max(classes.data, 1)
                    llabel = torch.max(labels.data, 1)
                    total += labels.size(0)
                    #pdb.set_trace()
                    correct += predicted.eq(llabel[1].data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i*length + 1 + epoch * len(train_data)), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()

                #
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for i in range(0, len(val_data), BATCH_SIZE):
                        images, labels = get_batch(val_data[i:i + BATCH_SIZE], val_label[i:i + BATCH_SIZE])
                        net.eval()
                        # images, labels = images.to(device), labels.to(device)
                        inputs = augmentation(images.float() / 255.0)
                        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)
                        images = Variable(inputs).cuda()
                        labels = Variable(labels).cuda()

                        classes, reconstruct = net(images,labels)
                        #
                        _, predicted = torch.max(classes.data, 1)
                        llabel = torch.max(labels.data, 1)
                        total += labels.size(0)
                        correct += (predicted == llabel[1].data).sum()
                    print('accuracy%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    #
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % ('save_model', epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    #
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)

