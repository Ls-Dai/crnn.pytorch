from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import utils
import dataset

import models.crnn as crnn
from dataset import *


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

parser = argparse.ArgumentParser()
parser.add_argument('--train_root', type=str, help="dataset root", default='./data/train/')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=640, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--alphabet_str', type=str, default="0123456789abcdefghijklmnopqrstuvwxyz-")

parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Device selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_im_root = opt.train_root + "images/"
train_label_path = opt.train_root + "labels.csv"

batch_size = 4
train_dataset = OcrDataset(train_im_root, train_label_path)
# val_dataset = OcrDataset(train_im_root, train_label_path)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

alphabet = Alphabet(alphabet_str=opt.alphabet_str)

crnn = crnn.CRNN(32, 1, len(alphabet.dict.keys())+1, 640).to(device=device)
crnn.apply(weights_init)
# crnn.load_state_dict(torch.load(opt.pretrained))

criterion = torch.nn.CTCLoss()
criterion = criterion.to(device=device)

optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizer = optim.Adadelta(crnn.parameters())
# optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)

min_val_loss = 0.2
for epoch in range(500):
    loss_avg = 0
    crnn.train()
    for patch, (im, labels) in enumerate(train_loader):
        preds = crnn(Variable(im).to(device=device))
        preds = preds.log_softmax(2)
        target, target_length = get_labels(alphabet.dict, labels)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size)).to(device=device)

        loss = criterion(preds, Variable(target).to(device=device), preds_size, Variable(target_length).to(device=device)) / batch_size

        crnn.zero_grad()
        loss.backward()
        optimizer.step()        
        loss_avg += loss.item()

        if (patch+1)%50 == 0 or patch == len(train_loader)-1:
            print('[Train][Epoch: {}/200][Patch: {}/{}][Loss: {:.4f}]' .format(epoch+1, patch+1, len(train_loader),
                                                                       loss_avg/(patch+1)))
    # loss_avg = 0
    # crnn.eval()
    # for patch, (im, labels) in enumerate(val_loader):
    #     preds = crnn(Variable(im).to(device=device))
    #     preds = preds.log_softmax(2)
    #     target, target_length = get_labels(alphabet.dict, labels)
    #     preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size)).to(device=device)

    #     loss = criterion(preds, Variable(target).to(device=device), preds_size, Variable(target_length).to(device=device)) / batch_size
    #     loss_avg += loss.item()

    #     if patch == len(val_loader) - 1:
    #         print(
    #             '[Validation][Epoch: {}/200][Loss: {:.4f}]'.format(epoch + 1, loss_avg / (patch + 1)))

    # loss_avg = loss_avg / (patch + 1)
    # if loss_avg < min_val_loss:
    #     min_val_loss = loss_avg
    #     model_path = './saved_model/model_{}_{:.2f}.pth'.format(epoch+1, min_val_loss*100)
    #     torch.save(crnn.state_dict(), model_path)


        # do checkpointing
        # if i % opt.saveInterval == 0:
        #     torch.save( crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))

# def val(net, dataset, criterion, max_iter=100):
#     print('Start val')
#
#     for p in crnn.parameters():
#         p.requires_grad = False
#
#     net.eval()
#     data_loader = torch.utils.data.DataLoader(
#         dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
#     val_iter = iter(data_loader)
#
#     i = 0
#     n_correct = 0
#     loss_avg = utils.averager()
#
#     max_iter = min(max_iter, len(data_loader))
#     for i in range(max_iter):
#         data = val_iter.next()
#         i += 1
#         cpu_images, cpu_texts = data
#         batch_size = cpu_images.size(0)
#         utils.loadData(image, cpu_images)
#         t, l = converter.encode(cpu_texts)
#         utils.loadData(text, t)
#         utils.loadData(length, l)
#
#         preds = crnn(image)
#         preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
#         cost = criterion(preds, text, preds_size, length) / batch_size
#         loss_avg.add(cost)
#
#         _, preds = preds.max(2)
#         preds = preds.squeeze(2)
#         preds = preds.transpose(1, 0).contiguous().view(-1)
#         sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
#         for pred, target in zip(sim_preds, cpu_texts):
#             if pred == target.lower():
#                 n_correct += 1
#
#     raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
#     for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
#         print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
#
#     accuracy = n_correct / float(max_iter * opt.batchSize)
#     print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


