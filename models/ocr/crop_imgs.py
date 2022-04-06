
"""
根据生成的json文件 裁剪出识别训练数据
"""
import os
import cv2
import math
import shutil
import pathlib
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

from ocr_utils import load_gt, save


def order_points(pts):
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype="float32")
    # 获取左上角和右下角坐标点
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts):
    # 获取坐标点，并将它们分离开来
    rect = original_coordinate_transformation(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获取仿射变换矩阵并应用它
    M = cv2.getPerspectiveTransform(rect, dst)
    # 进行仿射变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后的结果
    return warped


def original_coordinate_transformation(polygon):
    """
    调整坐标顺序为：
      x1,y1    x2,y2
      x4,y4    x3,y3
    :param polygon:
    :return:
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = polygon.astype(float).reshape(-1)
    # 判断x1和x3大小，x3调整为大的数
    if x1 > x3:
        x1, y1, x3, y3 = x3, y3, x1, y1
    # 判断x2和x4大小，x4调整为大的数
    if x2 > x4:
        x2, y2, x4, y4 = x4, y4, x2, y2
    # 判断y1和y2大小，y1调整为大的数
    if y2 > y1:
        x2, y2, x1, y1 = x1, y1, x2, y2
    # 判断y3和y4大小，y4调整为大的数
    if y3 > y4:
        x3, y3, x4, y4 = x4, y4, x3, y3
    return np.array([[x2, y2], [x3, y3], [x4, y4], [x1, y1]], dtype=np.float32)


def crop(save_gt_path, json_path, save_path):
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    os.makedirs(save_path, exist_ok=True)
    data = load_gt(json_path)
    file_list = []
    for img_path, gt in tqdm(data.items()):
        img = Image.open(img_path).convert('RGB')
        img_name = pathlib.Path(img_path).stem
        for i, (polygon, text, illegibility, language) in enumerate(
                zip(gt['polygons'], gt['texts'], gt['illegibility_list'], gt['language_list'])):
            if illegibility:
                continue
            polygon = np.array(polygon)
            roi_img_save_path = os.path.join(save_path, '{}_{}.jpg'.format(img_name, i))
            # 对于只有四个点的图片，反射变换后存储
            if len(polygon) == 4:
                np_img = np.asarray(img)
                roi_img = four_point_transform(np_img, polygon)
                roi_img = Image.fromarray(roi_img).convert('RGB')
            else:
                x_min = polygon[:, 0].min()
                x_max = polygon[:, 0].max()
                y_min = polygon[:, 1].min()
                y_max = polygon[:, 1].max()
                roi_img = img.crop((x_min, y_min, x_max, y_max))
            roi_img.save(roi_img_save_path)
            file_list.append(roi_img_save_path + '\t' + text + '\t' + language)
            # plt.title(text)
            # plt.imshow(roi_img)
            # plt.show()
    save(file_list, save_gt_path)


def crop_image_by_yolo_results(img_filepath, det_filepath, result_dirpath):
    img = Image.open(img_filepath).convert('RGB')
    img_name = pathlib.Paht(img_filepath).stem





if __name__ == '__main__':
    print()