from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set
from utils import is_image_file, load_img, save_img

# Testing settings
T_DATA = "C:/Users/Lab/PycharmProjects/pix2pix-pytorch/dataset/test"
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=False, default='.', help='dataset')
parser.add_argument('--direction', type=str, default='B2A', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=200, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader')
parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
opt = parser.parse_args()
print(opt)

print('===> Loading datasets')
root_path = "dataset/"

test_set = get_test_set(os.path.join(root_path, opt.dataset), opt.direction)

testing_data_loader = DataLoader(dataset=test_set,
                                 num_workers=opt.threads,
                                 batch_size=opt.test_batch_size,
                                 shuffle=False)


device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)

net_g = torch.load(model_path).to(device)

if opt.direction == "a2b":
    image_dir = os.path.join("dataset", opt.dataset, "test", "a")
else:
    image_dir = os.path.join("dataset", opt.dataset, "test", "b")

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(os.path.join(image_dir, image_name))
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    result_dir = os.path.join("result", opt.dataset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_img(out_img, os.path.join(result_dir, image_name))
