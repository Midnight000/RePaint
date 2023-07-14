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


# duplicate image and mask
# gtname = 'data/fun/gt2/0.png'
# dirname = 'log/step_1000_1_10_1/eval/half_mask/dense_sample/gt'
# files = os.listdir(dirname)
# # img = Image.open(gtname)
# # mask = Image.open(dirname)
# i = 0
# for file in files:
#     # tmpimg = img
#     # tmpmask = mask
#     img = Image.open(os.path.join(dirname, file))
# #
# #     # dirname, filename = os.path.split(gtname)
# #     # num, suffix = os.path.splitext(filename)
# #     # num = str(int(i)).zfill(6)
# #     # gtname = os.path.join(dirname, num + suffix)
# #     # tmpimg.save(gtname)
# # #
# #     dirname, filename = os.path.split(maskname)
#     num = str(int(i))
#     newdir = 'DDNMmask'
#     maskname = os.path.join(newdir, str(num) + '_0.png')
#     img.save(maskname)
#     i = i + 1

#################################################################################### distance transform
# import numpy as np
# from scipy.ndimage import distance_transform_edt
# from PIL import Image

# 读取图片并转换成numpy数组
# dir = 'data/datasets/gt_keep_masks/nn2'
# files = os.listdir(dir)
# counter = 0
# total = 0
# for file in files:
#     counter = counter + 1
#     img = Image.open(os.path.join(dir, file)).convert('L')
#     # img = img.resize(
#     #     tuple(x // 2 for x in img.size), resample=Image.BOX
#     # )
#     img_array = np.array(img)
#
#     # 将黑色像素值设为1，白色像素值设为0
#     binary_array = np.where(img_array == 0, 1, 0)
#
#     # 计算距离变换
#     dist_transform = distance_transform_edt(binary_array)
#
#     # 将距离转换成整数类型
#     sum = -dist_transform.sum()
#     dist_transform = dist_transform.astype(int)
#
#     # 将所有黑色像素值的距离乘以-1
#     dist_transform = np.where(img_array == 0, dist_transform * -1, dist_transform)
#     # print(dist_transform)
#     print(counter, sum)
#     total = total + sum
# print('avr', total / 100)
# # 保存结果到文件
# result_img = Image.fromarray(dist_transform.astype(np.uint8))
# result_img.show()
# result_img.save('result.png')

############################################################################################## single
# import numpy as np
# from scipy.ndimage import distance_transform_edt
# from PIL import Image
# img = Image.open("mask512.png").convert('L')
# img_array = np.array(img)
# # 将黑色像素值设为1，白色像素值设为0
# binary_array = np.where(img_array == 0, 1, 0)
# # 计算距离变换
# dist_transform = distance_transform_edt(binary_array)
# # 将距离转换成整数类型
# sum = -dist_transform.sum()
# dist_transform = dist_transform.astype(int)
# # 将所有黑色像素值的距离乘以-1
# dist_transform = np.where(img_array == 0, dist_transform * -1, dist_transform)
# print(sum/512/512/512)
####################################################################################################
# lama
# gtdir = 'data/eval/gt'
# maskdir = 'log/step_1000_1_10_1/eval/ex64/dense_sample/gt_keep_mask'
# gts = os.listdir(gtdir)
# masks = os.listdir(maskdir)
# cnt = 0
# for gt in gts:
#     cnt = cnt + 1
#     img = Image.open(os.path.join(gtdir, gt))
#     newname = str(cnt).zfill(5) + '_crop000.png'
#     img.save(os.path.join('lama_data/gt', newname))
#
# cnt = 0
# for mask in masks:
#     cnt = cnt + 1
#     img = Image.open(os.path.join(maskdir, mask))
#     tmpmask = np.array(img)
#     tmpmask = Image.fromarray(255 - tmpmask)
#     newname = str(cnt).zfill(5) + '_crop000_mask000.png'
#     tmpmask.save(os.path.join('lama_data/ex64', newname))

# import numpy as np
# import matplotlib.pyplot as plt
#
# ####################################################################################################
# # 生成二维数组
# img = Image.open("result.png")
# data = np.random.rand(5, 5)
# data = np.array(img)
# # 绘制热点图
# plt.imshow(data, cmap='hot', vmax=3)
# # plt.colorbar()
# plt.axis('off')
# # plt.show()
# plt.savefig("hot.jpg", bbox_inches='tight', pad_inches = -0.1, dpi=73.5)

#####################################################################################################
# # 打开一张图片测试
# img = Image.open("2.jpg")
# # print(img)
# tmp = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float()/255
# blur = Blur_Shapen.GaussianBlur(channels=3, kernel_size=5, sigma=0.2)
# shapen = Blur_Shapen.Sharpen(1)
# tmp = torch.clamp(shapen(tmp), 0, 1)
# tmp = tmp.numpy().transpose(1, 2, 0)
# # print(tmp.shape)
# pil_image = Image.fromarray(np.uint8(tmp * 255))
# pil_image.show()
# # pil_image.save("data/fun/gt/2.png")
# img.close()

# reverse mask
# dirname = 'D:/Users/Administrator/Desktop/datasets/mask/testing_mask_dataset'
# newdir = 'D:/Users/Administrator/Desktop/datasets/mask/reverse'
# files = os.listdir(dirname)
# for filename in files:
#     mask = Image.open(os.path.join(dirname, filename))
#     tmpmask = np.array(mask)
#     tmpmask = Image.fromarray(255 - tmpmask)
#
#     maskname = os.path.join(os.path.join(newdir, filename))
#     tmpmask.save(maskname)



#####################################################################################################
# import numpy as np
# import cv2
#
# dir = 'log/step_1000_1_10_1/eval/40/dense_sample/gt_keep_mask'
# files = os.listdir(dir)
# for file in files:
#     mask = cv2.imread(os.path.join(dir, file))
#     name = os.path.splitext(file)[0]
#     # print(mask.shape)
#     mask = mask[:, :, 0]
#     mask = (mask == 255) * 1
#     # print(mask.shape)
#     np.save(os.path.join('npy/40', name), mask)

# gtdir = 'DDRM'
# gts = os.listdir(gtdir)
# for gt in gts:
#     img = Image.open(os.path.join(gtdir, gt))
#     cnt = os.path.splitext(gt.replace('orig_', ''))[0]
#     print(cnt)
#     newname = str(cnt) + '_0.png'
#     img.save(os.path.join('DDRM', newname))


#####################################################################################################
# Time-Quality line chart
# import matplotlib.pyplot as plt
#
# # 时间和LPIPS值数据
# time_HS = [250, 500, 750, 1000]
# lpips_HS = [0.124651, 0.117053, 0.114361, 0.109498]
#
# time_repaint = [250, 500, 750, 1000]
# lpips_repaint = [0.18888, 0.15996, 0.14529, 0.13440]
#
# time_ddrm = [250, 500, 750, 1000]
# lpips_ddrm = [0.131286, 0.126561, 0.125746, 0.127554]
#
# time_ddnm = [250, 500, 750, 1000]
# lpips_ddnm = [0.122097, 0.121399, 0.120999, 0.121632]
#
# # 绘制折线图
# plt.plot(time_HS, lpips_HS, 'o-', color='red', label='HS(Ours)')
# plt.plot(time_repaint, lpips_repaint, 's-', color='orange', label='Repaint')
# plt.plot(time_ddrm, lpips_ddrm, '^-', color='cyan', label='ddrm')
# plt.plot(time_ddnm, lpips_ddnm, '*-', color='green', label='ddnm')
#
# # 添加标题和标签
# plt.title('LPIPS on Different Sampling Steps')
# plt.xlabel('Sampling Steps')
# plt.ylabel('LPIPS')
# plt.legend()
#
# # for i in range(len(time_repaint)):
# #     plt.scatter(time_HS[i], lpips_HS[i], color='blue')
# #     plt.scatter(time_repaint[i], lpips_repaint[i], color='orange')
# #     if i < 4:
# #         plt.scatter(time_ddrm[i], lpips_ddrm[i], color='green')
# #         plt.scatter(time_ddnm[i], lpips_ddnm[i], color='red')
# # 显示图像
# plt.savefig('line_chart.png')
# plt.show()

#####################################################################################################
# Ablation study line chart
# import matplotlib.pyplot as plt
#
# # 时间和LPIPS值数据
# half_eta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# lpips_half_eta = [0.219847, 0.218006, 0.213073, 0.210259, 0.203068, 0.195147, 0.194620, 0.191573, 0.189264, 0.197354, 0.237985]
#
# twenty_eta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# lpips_twenty_eta = [0.013535, 0.013680, 0.013625, 0.013454, 0.012922, 0.012526, 0.012806, 0.012668, 0.013211, 0.013798, 0.021202]
#
# fourty_eta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# lpips_fourty_eta = [0.060118, 0.059363, 0.058468, 0.057001, 0.056630, 0.053877, 0.053154, 0.053831, 0.054145, 0.057162, 0.087439]
#
# sixty_eta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# lpips_sixty_eta = [0.140437, 0.139487, 0.136538, 0.133202, 0.131313, 0.126144, 0.125514, 0.124234, 0.124669, 0.131002, 0.195527]
#
# # 绘制折线图
# plt.plot(half_eta, lpips_half_eta, 'o-', color='red', label='half_mask')
# # plt.plot(sixty_eta, lpips_sixty_eta, 's-', color='orange', label='40-60%mask')
# # plt.plot(fourty_eta, lpips_fourty_eta, '*-', color='green', label='20-40%mask')
# # plt.plot(twenty_eta, lpips_twenty_eta, '^-', color='blue', label='0-20%mask')
#
# # 添加标题和标签
# plt.title('LPIPS with different setting for mid')
# plt.xlabel('mid')
# plt.ylabel('LPIPS')
# plt.legend()
#
# plt.savefig('line_chart.png')
# plt.show()

###################################################################################################### index
width = 32
startx = 128
starty = 128
tmptensor = torch.ones([1, 3, 256, 256])
index = (0, slice(0, 3), slice(startx, startx + width), slice(starty, starty + width))
tmptensor[index] = 0
print(index)
print(tmptensor)
