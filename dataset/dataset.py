import sys
import re
import math
import six
import lmdb
import torch
import argparse
import random

from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import sampler

class LmdbDataset(Dataset):

    def __init__(self, root, dataset_config):

        self.root = root
        self.dataset_config = dataset_config
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        # print(self.env)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.dataset_config['data_filtering_off']:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.dataset_config['batch_max_length']:
                        # print(f'The length of the label is longer than max_length: length
                        #       {len(label)}, {label} in dataset {self.root}')
                        continue
                    if '[' in label or '#' in label:
                        continue
                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    # out_of_char = f'[^{self.dataset_config.character}]'
                    out_of_char = self.dataset_config['character']
                    # out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.dataset_config['rgb']:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.dataset_config['rgb']:
                    img = Image.new('RGB', (self.dataset_config['imgW'], self.dataset_config['imgH']))
                else:
                    img = Image.new('L', (self.dataset_config['imgW'], self.dataset_config['imgH']))
                label = '[dummy_label]'

            if not self.dataset_config['sensitive']:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            # out_of_char = f'[^{self.dataset_config.character}]'
            out_of_char = self.dataset_config['character']
            label = re.sub(out_of_char, '', label)
            # print('0')
            label_length = len(label)

        return img, label, label_length
    
class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels, lengths = zip(*batch)
        b_lengths = torch.IntTensor(lengths)
        b_labels = labels
        # b_labels = torch.IntTensor(labels)

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
            imgW = min(imgW, 400)

        transform = ResizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        b_images = torch.stack(images)

        return b_images, b_labels, b_lengths

class ResizeNormalize(object):
  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation
    self.toTensor = transforms.ToTensor()

  def __call__(self, img):
    img = img.resize(self.size, self.interpolation)
    img = self.toTensor(img)
    img.sub_(0.5).div_(0.5)
    return img

class RandomSequentialSampler(sampler.Sampler):

  def __init__(self, data_source, batch_size):
    self.num_samples = len(data_source)
    self.batch_size = batch_size

  def __len__(self):
    return self.num_samples

  def __iter__(self):
    n_batch = len(self) // self.batch_size
    tail = len(self) % self.batch_size
    index = torch.LongTensor(len(self)).fill_(0)
    for i in range(n_batch):
      random_start = random.randint(0, len(self) - self.batch_size)
      batch_index = random_start + torch.arange(0, self.batch_size)
      index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
    # deal with tail
    if tail:
      random_start = random.randint(0, len(self) - self.batch_size)
      tail_index = random_start + torch.arange(0, tail)
      index[(i + 1) * self.batch_size:] = tail_index

    return iter(index.tolist())

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length') # T的大小
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')  # 关键点
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode') # 能区分大写小写
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')  # ？
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    opt = parser.parse_args()

    data = LmdbDataset('E:/DLVC/Datasets/STR/data_lmdb_release/validation', opt=opt)
    img, label = data.__getitem__(0)
    print(len(data))
    print(img)
    print(label)