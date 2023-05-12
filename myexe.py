import os

import numpy as np
from utils import Dis_Transform
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
# output_path = 'data/mytest256/random_image/random_mask/gt'
# files = os.listdir('data/mytest1024')
# to_tensor = transforms.ToTensor()
# to_image = transforms.ToPILImage()
# for file in files:
#     num, suffix = os.path.splitext(file)
#     num = num.zfill(5)
#     file = num + '.' + suffix
#     real_file = os.path.join(input_path, file)
#     real_file = Image.open(real_file)
#     out = real_file.resize((256, 256))
#     real_file.close()
#     out.save(os.path.join(output_path, file))
#
#
# # Mask Generator
# to_tensor = transforms.ToTensor()
# to_image = transforms.ToPILImage()
# files = os.listdir('data/mytest256/random_image/random_mask/gt')
# i = 0
# for file in files:
#     rand = torch.rand((1, 256, 256))
#     mask = utils.mask.random_irregular_mask(rand)
#     mask = to_image(mask)
#     mask.save('data/mytest256/random_image/random_mask/mask/mask' + str(i) + '.jpg')
#     i += 1

# # load dataset
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--conf_path', type=str, required=False, default='confs/eval.yml')
#     args = vars(parser.parse_args())
#     conf_arg = conf_mgt.conf_base.Default_Conf()
#     conf_arg.update(yamlread(args.get('conf_path')))
#     dset = 'eval'
#     eval_name = conf_arg.get_default_eval_name()
#     dataset = conf_arg.get_dataloader(dset=dset, dsName=eval_name)
#     weight_mask = iter(dataset).__next__().get('gt_keep_mask')
#     weight_mask = (weight_mask*255).to(torch.uint8)
#     weight_mask = weight_mask.squeeze().cpu().numpy()
#     print(weight_mask.shape)
#     weight_mask = np.transpose(weight_mask, (1, 2, 0))
#     weight_mask = Image.fromarray(weight_mask, 'RGB')
#     weight_mask = Dis_Transform.dis_transform(weight_mask)
#     print((weight_mask / conf_arg.pget('image_size'))+1)
# if __name__ == '__main__':
#     main()


# # 打开一张图片测试
# img = Image.open("data/fun/gt/1.png")
# # print(img)
# tmp = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float()/255
# blur = Blur_Shapen.GaussianBlur(channels=3, kernel_size=5, sigma=0.2)
# shapen = Blur_Shapen.Sharpen(0.1)
# tmp = torch.clamp(shapen(tmp), 0, 1)
# tmp = tmp.numpy().transpose(1, 2, 0)
# # print(tmp.shape)
# pil_image = Image.fromarray(np.uint8(tmp * 255))
# pil_image.show()
# pil_image.save("data/fun/gt/2.png")
# img.close()

# duplicate image and mask
# gtname = 'data/eval/mask/genhalf'
# maskname = 'data/eval/mask/genhalf/000000.png'
# img = Image.open(gtname)
# mask = Image.open(maskname)
# for i in range(2, 2000):
#     # tmpimg = img
#     tmpmask = mask
#
#     # dirname, filename = os.path.split(gtname)
#     # num, suffix = os.path.splitext(filename)
#     # gtname = os.path.join(dirname, str(i) + suffix)
#     # tmpimg.save(gtname)
#
#     dirname, filename = os.path.split(maskname)
#     num, suffix = os.path.splitext(filename)
#     num = str(int(num)+1).zfill(6)
#     maskname = os.path.join(dirname, str(num) + suffix)
#     print(maskname)
#     tmpmask.save(maskname)


# import numpy as np
# from scipy.ndimage import distance_transform_edt
# from PIL import Image
#
# # 读取图片并转换成numpy数组
# img = Image.open('eval/mask/thin/000000.png').convert('L')
# img_array = np.array(img)
# print(img_array)
# # 将黑色像素值设为1，白色像素值设为0
# binary_array = np.where(img_array == 0, 1, 0)
#
# # 计算距离变换
# dist_transform = distance_transform_edt(binary_array)
#
# # 将距离转换成整数类型
# dist_transform = dist_transform.astype(int)
#
# # 将所有黑色像素值的距离乘以-1
# dist_transform = np.where(img_array == 0, dist_transform * -1, dist_transform)
#
# # 保存结果到文件
# result_img = Image.fromarray(dist_transform.astype(np.uint8))
# result_img.show()
# result_img.save('result.png')


files = os.listdir('lama_data/gt')
for file in files:
    print(file)
    num, suffix = os.path.splitext(file)
    newname = str(int(num)).zfill(5) + '_crop000.png'
    os.rename('gt/' + file, 'gt/' + newname)