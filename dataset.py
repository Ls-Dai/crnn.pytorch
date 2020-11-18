#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np


import cv2
import time
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader


class Alphabet:
    def __init__(self, alphabet_str) -> None:
        pass

    @property
    def dict(self):
        alphabet_dict = {}
        for i, key in enumerate(alphabet_str):
            alphabet_dict[key] = i + 1
        return alphabet_dict


def get_labels(alphabet, text_list):
    target = []
    target_length = []
    for text in text_list:
        for key in text:
            target.append(alphabet[key.lower()])
        target_length.append(len(text))
    return torch.LongTensor(target), torch.LongTensor(target_length)


class OcrDataset(data.Dataset):
    def __init__(self, root_im, label_path):
        self.im_list = []
        self.labels = []
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        start_time = time.time()
        for line in lines:
            line = line.replace('\n', '')
            line = line.replace(' ', '')
            # index, name, Left, Right, Top, Bottom, Quality, Text
            line = line.split(',')
            if line[7] != 'None':
                # print(root_im + line[1])
                im = cv2.imread(root_im + line[1])

                # In case that we cannot find the image via labels
                if im is None:
                    continue

                im = self.preprocess(im, line)
                self.im_list.append(im)
                self.labels.append(line[7])
            else:
                continue
        print('Finish Loading {} images in {:2f} seconds.' .format(
            len(self.labels), time.time()-start_time))

    def preprocess(self, im, line):
        im_map = np.zeros((32, 640)).astype('uint8')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        left, right, top, bottom = int(line[2]), int(
            line[3]), int(line[4]), int(line[5])
        im = im[top:bottom, left:right]
        h, w = im.shape[0:2]
        im = cv2.resize(im, (int(w * 32 / h), 32))
        im_map[:, :im.shape[1]] = im
        return im_map

    def __getitem__(self, index):
        im = self.im_list[index]
        im = torch.FloatTensor(im)
        im = torch.unsqueeze(im, 0)

        label = self.labels[index]
        return im, label

    def __len__(self):
        return len(self.labels)


class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels


if __name__ == '__main__':
    train_im_root = './data/train/images/'
    train_label_path = './data/train/labels.csv'

    alphabet_str = '0123456789abcdefghijklmnopqrstuvwxyz-'
    alphabet = Alphabet(alphabet_str=alphabet_str)

    train_dataset = OcrDataset(train_im_root, train_label_path)
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=1)

    for im, labels in train_loader:
        target, target_length = get_labels(alphabet.dict, labels)
        print("target:{}".format(target))
        print("target_length:{}".format(target_length))
