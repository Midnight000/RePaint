import os

import numpy as np

from utils import Blur_Shapen
import torch
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from guided_diffusion.image_datasets import load_data_inpa
import argparse
from PIL import Image
import conf_mgt
from utils import yamlread
import utils.mask


# Image Scaler
# input_path = 'data/mytest1024'
# output_path = 'data/mytest256/gt'
# files = os.listdir('data/mytest1024')
# to_tensor = transforms.ToTensor()
# to_image = transforms.ToPILImage()
# for file in files:
#     real_file = os.path.join(input_path, file)
#     real_file = Image.open(real_file)
#     out = real_file.resize((256, 256))
#     real_file.close()
#     out.save(os.path.join(output_path, file))
#
#
# Mask Generator
# to_tensor = transforms.ToTensor()
# to_image = transforms.ToPILImage()
# files = os.listdir('data/mytest256/gt')
# i = 0
# for file in files:
#     rand = torch.rand((1, 256, 256))
#     mask = utils.mask.random_irregular_mask(rand)
#     mask = to_image(mask)
#     mask.save('data/mytest256/mask/mask' + str(i) + '.jpg')
#     i += 1


# parser = argparse.ArgumentParser()
# parser.add_argument('--conf_path', type=str, required=False, default='confs/mytest.yml')
# args = vars(parser.parse_args())
# conf_arg = conf_mgt.conf_base.Default_Conf()
# conf_arg.update(yamlread(args.get('conf_path')))
# dset = 'eval'
# eval_name = conf_arg.get_default_eval_name()
#
# dataset = conf_arg.get_dataloader(dset=dset, dsName=eval_name)
# for image in dataset:
#     print(type(image))

# 打开一张图片测试
img = Image.open("data/test/2.png")
# print(img)
tmp = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float()/255
blur = Blur_Shapen.GaussianBlur(channels=3, kernel_size=5, sigma=0.2)
shapen = Blur_Shapen.Sharpen(0.1)
tmp = torch.clamp(shapen(tmp), 0, 1)
tmp = tmp.numpy().transpose(1, 2, 0)
# print(tmp.shape)
pil_image = Image.fromarray(np.uint8(tmp * 255))
pil_image.show()
pil_image.save("data/test/2.png")
img.close()

# duplicate image and mask
# gtname = 'data/mytest256/gt/1.png'
# maskname = 'data/mytest256/mask/1.png'
# img = Image.open(gtname)
# mask = Image.open(maskname)
# for i in range(2, 101):
#     tmpimg = img
#     tmpmask = mask
#
#     dirname, filename = os.path.split(gtname)
#     num, suffix = os.path.splitext(filename)
#     gtname = os.path.join(dirname, str(i) + suffix)
#     tmpimg.save(gtname)
#
#     dirname, filename = os.path.split(maskname)
#     num, suffix = os.path.splitext(filename)
#     maskname = os.path.join(dirname, str(i) + suffix)
#     tmpmask.save(maskname)
