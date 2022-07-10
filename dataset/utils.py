from lib2to3.pytree import convert
import os
import argparse

from dataset import LmdbDataset, RandomSequentialSampler, AlignCollate
from dataset.concatdataset import ConcatDataset
import torch
from torch.utils.data import DataLoader
# from tools.utils import CTCLabelConverter

def get_data(data_dir, opt, is_train, randomSampler=False):
    if isinstance(data_dir, list):
        dataset_list = []
        for data_dir_ in data_dir:
            print(data_dir_)
            dataset_list.append(LmdbDataset(data_dir_, opt))
        dataset = ConcatDataset(dataset_list)
    else:
        dataset = LmdbDataset(data_dir, opt)
    print('total image: ', len(dataset))
    # print(dataset_list)

    if randomSampler:
        sampler = RandomSequentialSampler(dataset, batch_size=opt.batch_size)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    if is_train:
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                shuffle=shuffle, pin_memory=True, drop_last=True,sampler=sampler,
                                collate_fn=AlignCollate(imgH=opt.imgH, imgW=opt.imgW))
    else:
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                shuffle=True, pin_memory=True, drop_last=False, 
                                collate_fn=AlignCollate(imgH=opt.imgH, imgW=opt.imgW))

    return dataset, data_loader

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length') # T的大小
    parser.add_argument('--batch_size', type=int, default=4, help='the batch size of each epoch')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')  # 关键点
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode') # 能区分大写小写
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')  # ？
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    opt = parser.parse_args()

    root = 'E:\DLVC\Datasets\STR\scene_text_train'
    data_dir = [os.path.join(root, dir_name) for dir_name in os.listdir(root)]
    print(data_dir)
    dataset, data_loader = get_data(data_dir, opt=opt, is_train=True, randomSampler=True)
    print(len(dataset))

    # converter = CTCLabelConverter(opt.character)
    # text, length = converter.encode(label, batch_max_length=opt.batch_max_length)
    # print(text.size())
    # print(text)