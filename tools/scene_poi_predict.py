
import mmcv
import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm
import json
from glob import glob
import csv

import config.predict_scene_poi_config as config

import det_predict as det_predict
import ocr_predict as ocr_predict
import vlcls_predict as vlcls_predict
import predict_utils as predict_utils
import Levenshtein
import random
import pandas as pd

from osgeo import gdal


def predict_scene_poi(args):

    img_path = args.img_path
    img_file_list = predict_utils.get_image_file_list(img_path)

    save_dirpath = args.save_dirpath

    roi_detector = det_predict.ROIDetector(args)
    ocr_recognizer = ocr_predict.OCRRecognizer(args)
    vlcls_classifier = vlcls_predict.VlClsClassifier(args)

    for img_filepath in tqdm(img_file_list):
        basename = os.path.basename(img_filepath)
        save_filepath = os.path.join(save_dirpath, basename.split('.')[0] + '.json')
        if os.path.exists(save_filepath) and os.path.getsize(save_filepath) != 0:
            continue

        result_dict = {}
        img_fullpath = os.path.abspath(img_filepath)
        result_dict['img_dirpath'] = os.path.dirname(img_fullpath)
        result_dict['img_name'] = os.path.basename(img_fullpath)

        img = mmcv.imread(img_filepath)
        result_list = roi_detector(img)

        result_dict['det_result'] = result_list
        for ri, roi_dict in enumerate(result_dict['det_result']):
            box = np.array(roi_dict['bbox'])
            segm = np.array(roi_dict['segmentation'])
            empty_img = np.zeros(img.shape, np.uint8)
            segm = cv2.drawContours(empty_img, [segm.reshape((-1, 1, 2))], -1, (255, 255, 255), cv2.FILLED)

            segm_gray = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)
            mask = segm_gray[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            image = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]

            image_mask = cv2.bitwise_and(image, image, mask=mask)

            ocr_result = ocr_recognizer(image_mask)

            roi_dict['ocr_result'] = ocr_result

            vlcls_dict = vlcls_classifier(image_mask, roi_dict)
            result_dict['det_result'][ri] = vlcls_dict


        with open(save_filepath, 'w', encoding='utf-8') as f:
            f.write(json.dumps(result_dict, indent=2))


def scene_poi_result_to_labelme(scene_poi_result_filepath, labelme_filepath):
    labelme_dict = {}
    with open(scene_poi_result_filepath, 'r', encoding='utf-8') as f:
        json_object = json.load(f)

    img_dirpath = json_object['img_dirpath']
    img_name = json_object['img_name']
    img_filepath = os.path.join(img_dirpath, img_name)

    image = Image.open(img_filepath)
    w, h = image.size

    labelme_dict['version'] = '4.5.6'
    labelme_dict['flags'] = {}
    labelme_dict['shapes'] = []
    labelme_dict['imagePath'] = os.path.basename(img_filepath)
    labelme_dict['imageData'] = None
    labelme_dict['imageHeight'] = h
    labelme_dict['imageWidth'] = w

    for di, det_box_dict in enumerate(json_object['det_result']):

        roi_dict = {}
        roi_dict['type'] = 'roi'
        roi_dict['roi_id'] = di
        roi_dict['label'] = det_box_dict['ROI_label']
        roi_dict['label_score'] = det_box_dict['ROI_score']

        det_box = det_box_dict['bbox']
        lt_x, lt_y = int(det_box[0]), int(det_box[1])
        rb_x, rb_y = int(det_box[2]), int(det_box[3])

        roi_dict['points'] = np.array([[lt_x, lt_y], [rb_x, lt_y],
                                       [rb_x, rb_y], [lt_x, rb_y]]).tolist()
        roi_dict['group_id'] = None
        roi_dict['shape_type'] = 'polygon'
        roi_dict['flags'] = {}

        labelme_dict['shapes'].append(roi_dict)

        for ti, text_box_dict in enumerate(det_box_dict['ocr_result']):
            points = np.array(text_box_dict['text_box'])

            min_coord = points.min(axis=0)
            max_coord = points.max(axis=0)

            text_lt_x, text_lt_y = min_coord[0], min_coord[1]
            text_rb_x, text_rb_y = max_coord[0], max_coord[1]

            text_lt_x, text_lt_y = text_lt_x + lt_x, text_lt_y + lt_y
            text_rb_x, text_rb_y = text_rb_x + lt_x, text_rb_y + lt_y

            text_dict = {}
            text_dict['text'] = text_box_dict['text']
            text_dict['text_score'] = text_box_dict['score']
            text_dict['type'] = 'text'
            text_dict['roi_id'] = di
            text_dict['label'] = text_box_dict['text_label']
            text_dict['label_score'] = text_box_dict['text_score']
            text_dict['points'] = np.array([[text_lt_x, text_lt_y], [text_rb_x, text_lt_y],
                                            [text_rb_x, text_rb_y], [text_lt_x, text_rb_y]]).tolist()

            text_dict['group_id'] = None
            text_dict['shape_type'] = 'polygon'
            text_dict['flags'] = {}

            labelme_dict['shapes'].append(text_dict)

    with open(labelme_filepath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(labelme_dict, indent=2))


def load_road_panoids_dict(fn_svid, sv_type):
    road_panoids_dict = {}

    with open(fn_svid, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            road_id = row['road_id']

            if sv_type == 'baidu' or sv_type == 'baidu_new':
                pid = row['pid']
            else:
                pid = row['panoid']

            if road_id not in road_panoids_dict:
                road_panoids_dict[road_id] = {}
                road_panoids_dict[road_id][pid] = row
            else:
                road_panoids_dict[road_id][pid] = row

    return road_panoids_dict


def load_road_panoids_history_dict(fn_svid):
    road_panoids_dict = {}

    with open(fn_svid, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            road_id = row['road_id']
            timeline = row['timeline']

            timeline_list = timeline.split(';')
            if len(timeline_list) == 1:
                continue

            if road_id not in road_panoids_dict:
                road_panoids_dict[road_id] = {}

            for history in timeline_list[1:]:
                _list = history.split('_')
                pid = _list[0]

                if pid in road_panoids_dict[road_id]:
                    continue

                road_panoids_dict[road_id][pid] = row
    return road_panoids_dict


def poi_generation(result_dirpath, pano_info_dirpath, dst_poi_filepath, sv_type='baidu'):
    if sv_type == 'baidu':
        pano_basename = 'baidu_shenzhen_all_filter_time_line_isin_final_part_{}.csv'
    elif sv_type == 'tencent':
        pano_basename = 'tencent_shenzhen_all_roads_panoids_filter_part_{}.csv'
    elif sv_type == 'baidu_new':
        pano_basename = 'baidu_new_part_{}.csv'
    else:
        raise Exception('sv_type error.')

    outfile = open(dst_poi_filepath, 'w', encoding='utf-8')
    outfile.write('fid,road_id,pano_id,side,id,type,lng,lat,prefix,title,subtitle,address,tag,tel,others\n')
    _count = 0
    for part_dirname in os.listdir(result_dirpath):
        print('Processing part {} ...'.format(part_dirname))
        part_name = part_dirname.split('_')[-1]
        pano_filename = pano_basename.format(part_name)
        pano_filepath = os.path.join(pano_info_dirpath, pano_filename)

        road_panoids_dict = load_road_panoids_dict(pano_filepath, sv_type)

        part_dirpath = os.path.join(result_dirpath, part_dirname)

        for road_dirname in tqdm(os.listdir(part_dirpath)):
            road_dirpath = os.path.join(part_dirpath, road_dirname)
            road_info = road_panoids_dict[road_dirname]

            for pano_dirname in os.listdir(road_dirpath):
                pano_generation_filename = '{}_det_result_ocr_vlcls.json'.format(pano_dirname)
                pano_generation_filepath = os.path.join(road_dirpath, pano_dirname, pano_generation_filename)

                if not os.path.exists(pano_generation_filepath):
                    continue

                pano_name = pano_dirname.split('_')[1]
                if '(' in pano_name:
                    pano_name = pano_name[:-3]
                pano_info = road_info[pano_name]
                side = pano_dirname.split('_')[2]

                if pano_info['is_in_shenzhen'] == '0':
                    continue

                if sv_type == 'baidu' or sv_type == 'baidu_new':
                    lng = float(pano_info['lng'])
                    lat = float(pano_info['lat'])
                else:
                    lng = float(pano_info['wgs_x'])
                    lat = float(pano_info['wgs_y'])

                with open(pano_generation_filepath, 'r', encoding='utf-8') as f:
                    json_object = json.load(f)

                    img_dirpath = json_object['img_dirpath']
                    img_name = json_object['img_name']
                    img_filepath = os.path.join(img_dirpath, img_name)

                    image = Image.open(img_filepath)
                    img_width = image.width

                    quarter_img_width = int(img_width / 4)
                    three_quarter_img_width = int(img_width * 3 / 4)

                    for di, det_result in enumerate(json_object['det_result']):
                        if 'ROI_label' not in det_result:
                            continue

                        bbox = np.array(det_result['bbox'])
                        minx = bbox[0]
                        maxx = bbox[2]

                        if maxx < quarter_img_width or minx > three_quarter_img_width:
                            continue

                        type = det_result['ROI_label']
                        text_dict = {'Prefix': '', 'Title': '', 'Subtitle': '', 'Address': '', 'Tag': '', 'Tel': '', 'Others': ''}

                        for ocr_result in det_result['ocr_result']:
                            text = ocr_result['text']
                            text = text.replace(',', ';')
                            text = text.replace('"', ' ')
                            text_label = ocr_result['text_label']

                            text_dict[text_label] = text_dict[text_label] + text

                        poi_lng = random.uniform(lng-0.0002, lng+0.0002)
                        poi_lat = random.uniform(lat-0.0002, lat+0.0002)

                        text_list = [item for item in text_dict.values()]
                        out_str = '{},{},{},{},{},{},{},{},{}\n'\
                            .format(_count, road_dirname, pano_name, side, di, type, poi_lng, poi_lat, ','.join(text_list))
                        outfile.write(out_str)
                        outfile.flush()
                        _count += 1
    outfile.flush()
    outfile.close()


if __name__ == "__main__":
    args = config.parse_args()
    predict_scene_poi(args)
