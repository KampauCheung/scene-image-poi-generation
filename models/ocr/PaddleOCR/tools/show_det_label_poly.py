
import cv2
import os
import json
import numpy as np


def draw_det_label_poly(img_filepath, box_filepath):
    with open(box_filepath, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    img_label_dict = {item.split('\t')[0].split('/')[-1]: item.split('\t')[1] for item in lines}
    img_name = os.path.basename(img_filepath)
    label = img_label_dict[img_name]
    label_json = json.loads(label)

    img = cv2.imread(img_filepath)
    for label in label_json:
        text = label['transcription']
        box = np.array(label['points'])
        cv2.circle(img, (box[0][0], box[0][1]), 5, (0, 255, 0), -1)
        box = box.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [box], True, color=(255, 255, 0), thickness=2)
    cv2.namedWindow(img_name)
    cv2.imshow(img_name, img)
    cv2.waitKey(0)


if __name__ == '__main__':
    draw_det_label_poly('D:/ocr_dataset/art_train_train/gt_1497.jpg',
                        'D:/ocr_dataset/art_train_train.txt')

