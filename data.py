import glob

from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import os.path
import numpy as np

"""
test set은 domain 별로 정리되서 나오게
"""


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


class UnpairedImageFileList(data.Dataset):
    def __init__(self, dataroot, config, file_a, file_b, transform=None, loader=default_loader):
        self.dataroot = dataroot
        self.config = config
        self.transform = transform
        self.loader = loader

        self.imlist_a = default_flist_reader(file_a)
        self.imlist_b = default_flist_reader(file_b)

        self.size_a = len(self.imlist_a)
        self.size_b = len(self.imlist_b)

        self.shuffle_interval = max(self.size_a, self.size_b)
        self.shuffle_cnt = 0

    def __getitem__(self, index):
        impath_a = self.imlist_a[index % self.size_a]
        img_a = self.loader(os.path.join(self.dataroot, impath_a))

        impath_b = self.imlist_b[index % self.size_a]
        img_b = self.loader(os.path.join(self.dataroot, impath_b))

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        return img_a, img_b, impath_a, impath_b

    def __len__(self):
        return max(self.size_a, self.size_b)


###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class ImageFolder(data.Dataset):
    def __init__(self, root, config, transform=None, return_paths=False, loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in: " + root + "\n"
                                                               "Supported image extensions are: " +
                                ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.config = config
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        if self.config['split']:
            width = self.config['crop_image_width']
            img_a, img_b = img[..., width:], img[..., :width]
            return img_a, img_b, path

        return img, path

    def __len__(self):
        return len(self.imgs)


class MultiDomainImageFolder(data.Dataset):
    def __init__(self, img_path, mode, transform, loader=default_loader):
        self.img_path = img_path
        self.mode = mode
        self.transform = transform
        self.loader = loader
        assert mode == 'train' or mode == 'test'

        self.domains = os.listdir(os.path.join(img_path, mode))
        self.main_data = []
        self.asd = []

        self.distribute_data()
        self.size = len(self.main_data)

    def distribute_data(self):
        for n, domain in enumerate(self.domains):
            imgs = glob.glob(os.path.join(self.img_path, self.mode, domain, '*'))
            for img in imgs:
                if is_image_file(img):
                    self.main_data.append([img, np.array(n).reshape(-1, 1)])

    def __getitem__(self, index):
        img_path, domain = self.main_data[index]
        img = self.loader(img_path)
        if self.transform:
            img = self.transform(img)

        return img, domain, img_path

    def __len__(self):
        return self.size


class UnpairedImageFolderWithAttr(data.Dataset):
    def __init__(self, img_path, attr_file_name, domain_attrs, mode, transform, loader=default_loader):
        self.img_path = img_path
        self.attr_file_name = attr_file_name
        self.domain_attrs = domain_attrs
        self.mode = mode
        self.transform = transform
        self.loader = loader
        assert mode == 'train' or mode == 'test'

        with open(attr_file_name, 'r') as f:
            readlines = f.readlines()
            self.attr_names, self.attrs = readlines[1], readlines[2:]
            self.attr_names = self.attr_names.strip().split(' ')

        self.attr_indices = self.attr_names.index(domain_attrs)

        self.train_data = []
        self.test_data = []
        self.distribute_data()

        self.main_data = self.train_data if self.mode == 'train' else self.test_data
        self.size = len(self.main_data)

    def distribute_data(self):
        for n, attr in enumerate(self.attrs):
            split = attr.split()
            img_name, attr_value = split[0], split[1:]

            is_male = attr_value[self.attr_indices] == '1'
            is_male = np.array(int(is_male)).reshape(-1, 1)
            if n < 500:
                self.test_data.append([img_name, is_male])
            else:
                self.train_data.append([img_name, is_male])

    def __getitem__(self, index):
        img_name, isMale = self.main_data[index]
        img = self.loader(os.path.join(self.img_path, img_name))
        if self.transform:
            img = self.transform(img)
        return img, isMale, img_name

    def __len__(self):
        return self.size


def get_transform(crop_size, resize, is_flip):
    transform_list = []
    transform_list += [transforms.CenterCrop(crop_size)]
    transform_list += [transforms.Resize(resize)]

    if is_flip:
        transform_list += [transforms.RandomHorizontalFlip()]

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)
    return transform


def get_data_loader(config, dataset_root, shuffle=True, is_train=True):
    img_path = os.path.join(dataset_root, config['img_path'])
    mode = 'train' if is_train else 'test'

    is_flip = config['is_flip']
    crop_size = config['crop_size']
    re_size = config['re_size']
    num_workers = config['num_workers']

    batch_size = config['batch_size'] if is_train else config['batch_size_test']

    transform = get_transform(crop_size, re_size, is_flip)

    dataset = MultiDomainImageFolder(img_path, mode, transform)
    loader = DataLoader(dataset, batch_size, shuffle, drop_last=True, num_workers=num_workers)
    return loader


if __name__ == '__main__':
    from utils import get_config
    from common.visualizer import show
    from common.utils_torch import show_batch_torch

    config = './config/afhq.yaml'
    if False:
        dataset_root = '/mnt/disks/sdb/datasets'
    else:
        dataset_root = '/Users/bochan/_datasets'
    config = get_config(config)

    loader = get_data_loader(config, dataset_root)

    img, domain, _ = next(iter(loader))
    print(img.shape)
    print(domain)
    print(domain.shape, domain)
