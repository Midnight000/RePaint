import numpy as np
from scipy.ndimage import distance_transform_edt
from PIL import Image


def dis_transform(img):
    img.convert('L')
    # 读取图片并转换成numpy数组
    #     img = Image.open('eval/mask/thin/000000.png').convert('L')
    img_array = np.array(img)
    # print(img)
    # 将黑色像素值设为1，白色像素值设为0
    binary_array = np.where(img_array == 0, 1, 0)

    # 计算距离变换
    dist_transform = distance_transform_edt(binary_array)

    # 将距离转换成整数类型
    dist_transform = dist_transform.astype(int)

    # 将所有黑色像素值的距离乘以-1
    # dist_transform = np.where(img_array == 0, dist_transform * -1, dist_transform)
    return dist_transform
