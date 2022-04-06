# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import subprocess
import config.ocr_config as ocr_config
import json

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../models/ocr')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import PaddleOCR.tools.infer.predict_system as predict_system
from PaddleOCR.ppocr.utils.logging import get_logger
from PaddleOCR.tools.infer.utility import draw_ocr_box_txt, draw_ocr

logger = get_logger()


def ocr_rec_roi_det_json(text_sys, args, det_json_filepath, dst_ocr_filepath):

    drop_score = args.drop_score

    with open(det_json_filepath, 'r', encoding='utf-8') as f:
        json_object = json.load(f)

    img_dirpath = json_object['img_dirpath']
    img_name = json_object['img_name']

    img_path = os.path.join(img_dirpath, img_name)
    img = cv2.imread(img_path)

    for bi, box_dict in enumerate(json_object['det_result']):
        box = np.array(box_dict['bbox'])
        segm = np.array(box_dict['segmentation'])
        score = box_dict['score']
        label = box_dict['label']

        empty_img = np.zeros(img.shape, np.uint8)
        segm = cv2.drawContours(empty_img, [segm.reshape((-1, 1, 2))], -1, (255, 255, 255), cv2.FILLED)

        segm_gray = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)
        mask = segm_gray[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        image = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]

        image_mask = cv2.bitwise_and(image, image, mask=mask)

        dt_boxes, rec_res = text_sys(image_mask)

        ocr_result = []
        for oi, ocr_box in enumerate(dt_boxes):
            text, score = rec_res[oi]
            if score < drop_score:
                continue
            ocr_det_dict = {}
            ocr_det_dict['text'] = text
            ocr_det_dict['score'] = float(score)
            ocr_det_dict['text_box'] = ocr_box.tolist()
            ocr_result.append(ocr_det_dict)

        json_object['det_result'][bi]['ocr_result'] = ocr_result

    with open(dst_ocr_filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_object, indent=2))


def predict_all_det_result(args, det_result_dirpath):

    text_sys = predict_system.TextSystem(args)

    for road_dirname in os.listdir(det_result_dirpath):
        print('processing road_dir: {}'.format(road_dirname))
        road_dirpath = os.path.join(det_result_dirpath, road_dirname)

        for img_dirname in tqdm(os.listdir(road_dirpath)):
            img_dirpath = os.path.join(road_dirpath, img_dirname)
            det_json_filename = '{}_det_result.json'.format(img_dirname)
            det_json_filepath = os.path.join(img_dirpath, det_json_filename)

            if not os.path.exists(det_json_filepath):
                continue

            dst_ocr_json_filename = '{}_det_result_ocr.json'.format(img_dirname)
            dst_ocr_json_filepath = os.path.join(img_dirpath, dst_ocr_json_filename)

            try:
                with open(dst_ocr_json_filepath, 'r', encoding='utf-8') as f:
                    json_object = json.load(f)
            except Exception as e:
                ocr_rec_roi_det_json(text_sys, args, det_json_filepath, dst_ocr_json_filepath)
            # if os.path.exists(dst_ocr_json_filepath) and os.path.getsize(dst_ocr_json_filepath) != 0:
            #     continue
            # ocr_rec_roi_det_json(text_sys, args, det_json_filepath, dst_ocr_json_filepath)


class OCRRecognizer(object):
    def __init__(self, args):
        self.text_sys = predict_system.TextSystem(args)
        self.drop_score = args.drop_score
        print()

    def __call__(self, img):
        dt_boxes, rec_res = self.text_sys(img)
        ocr_result = []
        for oi, ocr_box in enumerate(dt_boxes):
            text, score = rec_res[oi]
            if score < self.drop_score:
                continue
            ocr_det_dict = {}
            ocr_det_dict['text'] = text
            ocr_det_dict['score'] = float(score)
            ocr_det_dict['text_box'] = ocr_box.tolist()
            ocr_result.append(ocr_det_dict)
        return ocr_result


def predict_and_save_json(args, img_filepath, mask_filepath, dst_json_filepath):
    text_sys = predict_system.TextSystem(args)
    drop_score = args.drop_score

    img = cv2.imread(img_filepath)
    mask = cv2.imread(mask_filepath)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    img_mask = cv2.bitwise_and(img, img, mask=mask_gray)

    dt_boxes, rec_res = text_sys(img_mask)

    ocr_det_result_dict = {}
    ocr_result = []

    for di, ocr_box in enumerate(dt_boxes):
        text, score = rec_res[di]
        if score < drop_score:
            continue

        ocr_det_dict = {}
        ocr_det_dict['text'] = text
        ocr_det_dict['score'] = float(score)
        ocr_det_dict['text_box'] = ocr_box.tolist()
        ocr_result.append(ocr_det_dict)
    ocr_det_result_dict['ocr_result'] = ocr_result

    with open(dst_json_filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(ocr_det_result_dict, indent=2))


def show_predict_result(img_filepath, mask_filepath, dst_json_filepath, dst_img_filepath, img_mask_filepath, dst_black_filepath):
    ori_img = cv2.imread(img_filepath)
    mask = cv2.imread(mask_filepath)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    img_new = np.zeros(ori_img.shape, dtype=np.uint8)
    img_new.fill(255)

    # img_new = cv2.bitwise_and(img_new, img_new, mask=mask_gray)

    img = cv2.bitwise_and(ori_img, ori_img, mask=mask_gray)
    cv2.imwrite(img_mask_filepath, img)

    with open(dst_json_filepath, 'r', encoding='utf-8') as f:
        json_object = json.load(f)

    for text_dict in json_object['ocr_result']:
        box = np.array(text_dict['text_box'])
        box = box.astype(np.int32).reshape((-1, 1, 2))
        text = text_dict['text']
        score = float(text_dict['score'])

        cv2.polylines(img, [box], True, color=(0, 153, 255), thickness=10)
        cv2.polylines(img_new, [box], True, color=(0, 153, 255), thickness=10)
        # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # draw = ImageDraw.Draw(img)
        # fontStyle = ImageFont.truetype(
        #     "font/msyh.ttc", 32, encoding="utf-8")
        # draw.text((int(box[0, 0, 0]), int(box[0, 0, 1])), text, (153, 204, 51), font=fontStyle, stroke_width=1)
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    cv2.imshow('', img_new)
    cv2.waitKey()
    cv2.imwrite(dst_img_filepath, img)
    cv2.imwrite(dst_black_filepath, img_new)


if __name__ == "__main__":
    args = ocr_config.parse_args()
    det_result_dirpath = r'Z:\Data\baidu_street_view_new_shenzhen_history_split_result_json\baidu_street_view_new_shenzhen_panorama_009'
    predict_all_det_result(args, det_result_dirpath)
