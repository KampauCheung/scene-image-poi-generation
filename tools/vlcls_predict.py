import torch
import math
import sys
import time
import os
import json
import cv2
import numpy as np
import torchvision
from tqdm import tqdm

from models.utils import utils
from config import vlcls_config
from models.vlcls.vlcls_dataset import VlClsDataset, collate_fn
# from faster_rcnn_feature import FasterRCNNFeatureCls
from models.vlcls.vlcls_model import FasterRCNNFeatureMultiClsModel, LanguageMultiClsModel, VisualLanguageMultiClsModel
import torch.nn.functional as F


@torch.no_grad()
def predict(model, ocr_rec_roi_det_json_filepath, final_json_filepath, device,
            ocr_drop_score=0.5, ROI_LABEl_MAP=vlcls_config.ROI_LABEL_MAP,
            TEXT_LABEL_MAP=vlcls_config.TEXT_LABEL_MAP):


    ROI_lable_list = list(ROI_LABEl_MAP.keys())
    Text_label_list = list(TEXT_LABEL_MAP.keys())

    cpu_device = torch.device("cpu")

    with open(ocr_rec_roi_det_json_filepath, 'r', encoding='utf-8') as f:
        json_object = json.load(f)

    img_dirpath = json_object['img_dirpath']
    img_name = json_object['img_name']

    img_path = os.path.join(img_dirpath, img_name)
    img = cv2.imread(img_path)

    loader = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])

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
        img = cv2.cvtColor(image_mask, cv2.COLOR_BGR2RGB)
        img_tensor = loader(img)

        boxes, texts = [], []
        for oi, ocr in enumerate(box_dict['ocr_result']):
            text = ocr['text']
            ocr_score = ocr['score']

            if float(ocr_score) < ocr_drop_score:
                continue

            text_box = np.array(ocr['text_box'])
            x1, x2 = min(text_box[:, 0]), max(text_box[:, 0])
            y1, y2 = min(text_box[:, 1]), max(text_box[:, 1])
            boxes.append([x1, y1, x2, y2])
            texts.append(text)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["area"] = area

        images = [img_tensor.to(device)]
        targets = [{k: v.to(device) for k, v in target.items()}]

        ROI_logits, text_logits = model(images, targets, [texts])
        ROI_logits = F.softmax(ROI_logits, 1)
        text_logits = F.softmax(text_logits[0], 1)

        ROI_pred = torch.max(ROI_logits, 1)
        ROI_label = ROI_pred[1].to(cpu_device).numpy()[0]
        ROI_score = ROI_pred[0].to(cpu_device).numpy()[0]

        box_dict['ROI_label'] = ROI_lable_list[ROI_label]
        box_dict['ROI_score'] = float(ROI_score)

        text_pred = torch.max(text_logits, 1)

        for ti, ocr_box in enumerate(box_dict['ocr_result']):
            text_label = text_pred[1][ti].to(cpu_device).numpy().item()
            text_socre = text_pred[0][ti].to(cpu_device).numpy().item()

            ocr_box['text_label'] = Text_label_list[text_label]
            ocr_box['text_score'] = float(text_socre)

    with open(final_json_filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_object, indent=2))


class VlClsClassifier(object):
    def __init__(self, args):
        self.args = args
        self.model = torch.load(args.vlcls_checkpoint_file)
        self.ROI_label_list = list(json.loads(args.roi_label_map).keys())
        self.Text_label_list = list(json.loads(args.text_label_map).keys())

        if args.use_gpu:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        else:
            device = torch.device('cpu')

        self.model.to(device)
        self.model.eval()
        self.device = device
        self.cpu_device = torch.device('cpu')

        self.loader = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

        self.ocr_drop_score = args.drop_score

    @torch.no_grad()
    def __call__(self, img, roi_dict):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.loader(img)

        boxes, texts = [], []
        for oi, ocr in enumerate(roi_dict['ocr_result']):
            text = ocr['text']
            ocr_score = ocr['score']

            if float(ocr_score) < self.ocr_drop_score:
                continue

            text_box = np.array(ocr['text_box'])
            x1, x2 = min(text_box[:, 0]), max(text_box[:, 0])
            y1, y2 = min(text_box[:, 1]), max(text_box[:, 1])
            boxes.append([x1, y1, x2, y2])
            texts.append(text)

        if len(boxes) == 0:
            return roi_dict

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["area"] = area

        images = [img_tensor.to(self.device)]
        targets = [{k: v.to(self.device) for k, v in target.items()}]

        ROI_logits, text_logits = self.model(images, targets, [texts])
        ROI_logits = F.softmax(ROI_logits, 1)
        text_logits = F.softmax(text_logits[0], 1)

        ROI_pred = torch.max(ROI_logits, 1)
        ROI_label = ROI_pred[1].to(self.cpu_device).numpy()[0]
        ROI_score = ROI_pred[0].to(self.cpu_device).numpy()[0]

        roi_dict['ROI_label'] = self.ROI_label_list[ROI_label]
        roi_dict['ROI_score'] = float(ROI_score)

        text_pred = torch.max(text_logits, 1)
        for ti, ocr_box in enumerate(roi_dict['ocr_result']):
            if ti < 31:
                text_label = text_pred[1][ti].to(self.cpu_device).numpy().item()
                text_socre = text_pred[0][ti].to(self.cpu_device).numpy().item()
            else:
                text_label = 4
                text_socre = 0.5

            ocr_box['text_label'] = self.Text_label_list[text_label]
            ocr_box['text_score'] = float(text_socre)

        return roi_dict


def predict_vlcls_ocr_det_json(vlcls_classifier, args, ocr_json_filepath, dst_vlcls_json_filepath):
    with open(ocr_json_filepath, 'r', encoding='utf-8') as f:
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

        roi_dict = vlcls_classifier(image_mask, box_dict)
        json_object['det_result'][bi] = roi_dict

    with open(dst_vlcls_json_filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_object, indent=2))


def predict_all_ocr_result(args, ocr_result_dirpath):

    vlcls_classifier = VlClsClassifier(args)

    for road_dirname in tqdm(os.listdir(ocr_result_dirpath)):
        # print('processing road_dir: {}'.format(road_dirname))
        road_dirpath = os.path.join(ocr_result_dirpath, road_dirname)

        for img_dirname in os.listdir(road_dirpath):
            img_dirpath = os.path.join(road_dirpath, img_dirname)
            ocr_json_filename = '{}_det_result_ocr.json'.format(img_dirname)
            ocr_json_filepath = os.path.join(img_dirpath, ocr_json_filename)

            if not os.path.exists(ocr_json_filepath):
                continue

            dst_vlcls_json_filename = '{}_det_result_ocr_vlcls.json'.format(img_dirname)
            dst_vlcls_json_filepath = os.path.join(img_dirpath, dst_vlcls_json_filename)

            try:
                predict_vlcls_ocr_det_json(vlcls_classifier, args, ocr_json_filepath, dst_vlcls_json_filepath)
                if os.path.exists(dst_vlcls_json_filepath) and os.path.getsize(dst_vlcls_json_filepath) != 0:
                    continue
            except Exception as e:
                print(e)
                print(img_dirpath)


def predict_ocr_sample(args, ocr_sample_dirpath):

    vlcls_classifier = VlClsClassifier(args)

    for json_filename in os.listdir(ocr_sample_dirpath):
        if not json_filename.endswith('_ocr.json'):
            continue
        basename = json_filename.replace('_ocr.json', '')
        img_name = '{}_image_mask.jpg'.format(basename)
        dst_json_name = '{}_ocr_vlcls.json'.format(basename)

        json_filepath = os.path.join(ocr_sample_dirpath, json_filename)
        img_filepath = os.path.join(ocr_sample_dirpath, img_name)

        img = cv2.imread(img_filepath)

        with open(json_filepath, 'r', encoding='utf-8') as f:
            json_object = json.load(f)

        roi_dict = vlcls_classifier(img, json_object)

        dst_json_filepath = os.path.join(ocr_sample_dirpath, dst_json_name)
        with open(dst_json_filepath, 'w', encoding='utf-8') as f:
            f.write(json.dumps(roi_dict, indent=2))


if __name__ == "__main__":

    args = vlcls_config.parse_args()
    ocr_result_dirpath = r'Z:\Data\baidu_street_view_new_shenzhen_history_split_result_json\baidu_street_view_new_shenzhen_panorama_009'
    predict_all_ocr_result(args, ocr_result_dirpath)

