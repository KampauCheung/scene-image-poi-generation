from mmdet.apis import init_detector, inference_detector
import mmcv
import numpy as np
import cv2
import random
import torch
import os
import shutil
from PIL import Image
from tqdm import tqdm
import glob
import json
from mmdet.core.visualization import imshow_det_bboxes
import predict_utils as predict_utils
from models.utils import det_utils


def get_outputs(model, result, threshold=0.3):
    bbox_result, segm_result = result

    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        segms = np.stack(segms, axis=0)

    scores = bboxes[:, -1]
    inds = scores > threshold
    scores = scores[inds]
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    if segms is not None:
        segms = segms[inds, ...]

    labels = np.array([model.CLASSES[i] for i in labels])
    return segms, bboxes, labels, scores


def predict(model_path, config_file, img_path, output_dirpath, predict_batch_size=4, is_show=True):
    os.makedirs(output_dirpath, exist_ok=True)
    if os.path.isdir(img_path):
        img_fns = os.listdir(img_path)
        img_paths = [os.path.join(img_path, item) for item in img_fns]
    else:
        img_paths = [img_path]

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = init_detector(config_file, model_path, device=device)

    img_paths = [img_paths[i:i+predict_batch_size] for i in range(0, len(img_paths), predict_batch_size)]

    for si, sub_img_paths in tqdm(enumerate(img_paths)):
        results = inference_detector(model, sub_img_paths)

        for ri, result in enumerate(results):
            img_path = sub_img_paths[ri]
            img = mmcv.imread(img_path)
            img = img.copy()

            if is_show:
                model.show_result(img, result, show=True)

            segms, bboxes, labels, scores = get_outputs(model, result, threshold=0.3)
            if len(bboxes) == 0:
                continue

            pick = det_utils.non_max_suppression_iou_intersect(bboxes, scores, 0.3, 0.5)

            scores = scores[pick]
            bboxes = bboxes[pick, :]
            labels = labels[pick]
            if segms is not None:
                segms = segms[pick, ...]

            basename = os.path.basename(img_path).split('.')[0]
            save_dirpath = os.path.join(output_dirpath, basename)
            if os.path.isdir(save_dirpath):
                shutil.rmtree(save_dirpath)

            os.makedirs(save_dirpath)
            model.show_result(img, result, out_file=os.path.join(save_dirpath, basename+'_det_result.jpg'))

            for si, score in enumerate(scores):
                box = bboxes[si]
                segm = segms[si]
                label = labels[si]

                sub_segm_path = os.path.join(save_dirpath, '{}_{}_{}_{:.2}_segm.jpg'.format(basename, si, label, score))
                sub_image_path = os.path.join(save_dirpath, '{}_{}_{}_{:.2}_image.jpg'.format(basename, si, label, score))

                mask = segm[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                im = Image.fromarray(mask)
                im.save(sub_segm_path)

                image = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
                mmcv.imwrite(image, sub_image_path)


def predict_json(model_path, config_file, img_path, output_dirpath, predict_batch_size=4, is_show=True):
    os.makedirs(output_dirpath, exist_ok=True)
    if os.path.isdir(img_path):
        img_fns = os.listdir(img_path)
        img_paths = [os.path.join(img_path, item) for item in img_fns if not os.path.exists(os.path.join(output_dirpath, item.split('.')[0], item.split('.')[0] + '_det_result.jpg'))]
        img_paths = [item for item in img_paths if item.endswith('.jpg')]
        img_dirpath = img_path
    else:
        img_paths = [img_path]
        img_dirpath = os.path.dirname(img_path)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = init_detector(config_file, model_path, device=device)

    img_paths = [img_paths[i:i+predict_batch_size] for i in range(0, len(img_paths), predict_batch_size)]

    for si, sub_img_paths in tqdm(enumerate(img_paths)):
        results = inference_detector(model, sub_img_paths)
        for ri, result in enumerate(results):
            per_img_path = sub_img_paths[ri]
            basename = os.path.basename(per_img_path).split('.')[0]
            save_dirpath = os.path.join(output_dirpath, basename)

            os.makedirs(save_dirpath, exist_ok=True)
            if os.path.exists(os.path.join(save_dirpath, basename+'_det_result.jpg')):
                continue

            img = mmcv.imread(per_img_path)
            img = img.copy()

            if is_show:
                model.show_result(img, result, show=True)

            segms, bboxes, labels, scores = get_outputs(model, result, threshold=0.5)
            if len(bboxes) == 0:
                continue

            pick = det_utils.non_max_suppression_iou_intersect(bboxes, scores, 0.3, 0.5)

            scores = scores[pick]
            bboxes = bboxes[pick, :]
            labels = labels[pick]
            if segms is not None:
                segms = segms[pick, ...]

            per_img_result_dict = {}
            per_img_result_dict['img_dirpath'] = img_dirpath
            per_img_result_dict['img_name'] = os.path.basename(per_img_path)
            result_list = []

            for si, score in enumerate(scores):
                box_dict = {}
                box = bboxes[si]
                segm = segms[si]
                label = labels[si]

                outs = cv2.findContours((segm * 255).astype(np.uint8), cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)
                if len(outs) == 3:
                    img, contours, _ = outs[0], outs[1], outs[2]
                elif len(outs) == 2:
                    contours, _ = outs[0], outs[1]

                sub_segm_path = os.path.join(save_dirpath, '{}_{}_{}_{:.2}_segm.jpg'.format(basename, si, label, score))
                sub_image_path = os.path.join(save_dirpath, '{}_{}_{}_{:.2}_image.jpg'.format(basename, si, label, score))

                box_dict['bbox'] = box.tolist()
                box_dict['segmentation'] = contours[0].reshape((-1, 2)).tolist()
                box_dict['score'] = np.float(score)
                box_dict['label'] = label
                box_dict['sub_image_path'] = '{}_{}_{}_{:.2}_image.jpg'.format(basename, si, label, score)
                box_dict['sub_segm_path'] = '{}_{}_{}_{:.2}_segm.jpg'.format(basename, si, label, score)
                result_list.append(box_dict)

                mask = segm[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                im = Image.fromarray(mask)
                im.save(sub_segm_path)

                image = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
                mmcv.imwrite(image, sub_image_path)

            per_img_result_dict['det_result'] = result_list
            with open(os.path.join(save_dirpath, basename+'_det_result.json'), 'w', encoding='utf-8') as f:
                f.write(json.dumps(per_img_result_dict))
            model.show_result(img, result, score_thr=0.5,
                              out_file=os.path.join(save_dirpath, basename+'_det_result.jpg'), show=False)


def predict_multi(model_path, config_file, img_path, output_dirpath,
                  done_dirname_filepath, predict_batch_size=4, is_show=True):
    os.makedirs(output_dirpath, exist_ok=True)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = init_detector(config_file, model_path, device=device)

    done_file = open(done_dirname_filepath, 'a+', encoding='utf-8')
    done_file.seek(0)
    done_dirname = set(done_file.read().splitlines())
    for img_dir in os.listdir(img_path):
        if img_dir in done_dirname:
            continue
        print('prcessing img_dir: {}'.format(img_dir))
        img_dirpath = os.path.join(img_path, img_dir)
        save_det_result_path = os.path.join(output_dirpath, img_dir)
        if os.path.isdir(img_dirpath):
            img_fns = os.listdir(img_dirpath)
            img_paths = [os.path.join(img_dirpath, item) for item in img_fns if not os.path.exists(os.path.join(save_det_result_path, item.split('.')[0], item.split('.')[0] + '_det_result.jpg'))]
            img_paths = [item for item in img_paths if item.endswith('.jpg')]
        else:
            img_paths = [img_dirpath]

        img_paths = [img_paths[i:i+predict_batch_size] for i in range(0, len(img_paths), predict_batch_size)]

        is_done = True
        for si, sub_img_paths in tqdm(enumerate(img_paths)):
            try:
                results = inference_detector(model, sub_img_paths)
                for ri, result in enumerate(results):
                    curr_img_path = sub_img_paths[ri]
                    img = mmcv.imread(curr_img_path)
                    img = img.copy()

                    if is_show:
                        model.show_result(img, result, show=True)

                    segms, bboxes, labels, scores = get_outputs(model, result, threshold=0.3)
                    if len(bboxes) == 0:
                        continue

                    pick = det_utils.non_max_suppression_iou_intersect(bboxes, scores, 0.3, 0.5)

                    scores = scores[pick]
                    bboxes = bboxes[pick, :]
                    labels = labels[pick]
                    if segms is not None:
                        segms = segms[pick, ...]

                    basename = os.path.basename(curr_img_path).split('.')[0]
                    save_dirpath = os.path.join(save_det_result_path, basename)
                    # if os.path.isdir(save_dirpath):
                    #     shutil.rmtree(save_dirpath)

                    os.makedirs(save_dirpath, exist_ok=True)
                    if os.path.exists(os.path.join(save_dirpath, basename+'_det_result.jpg')):
                        continue

                    for si, score in enumerate(scores):
                        box = bboxes[si]
                        segm = segms[si]
                        label = labels[si]

                        sub_segm_path = os.path.join(save_dirpath, '{}_{}_{}_{:.2}_segm.jpg'.format(basename, si, label, score))
                        sub_image_path = os.path.join(save_dirpath, '{}_{}_{}_{:.2}_image.jpg'.format(basename, si, label, score))

                        mask = segm[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        im = Image.fromarray(mask)
                        im.save(sub_segm_path)

                        image = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
                        mmcv.imwrite(image, sub_image_path)

                    model.show_result(img, result, out_file=os.path.join(save_dirpath, basename+'_det_result.jpg'))
            except Exception as e:
                is_done = False
                print(e)
                print(sub_img_paths)
        if not is_done:
            continue
        done_dirname.add(img_dir)
        done_file.write(img_dir + '\n')
        done_file.flush()
    done_file.close()


def predict_multi_json(model_path, config_file, img_path, output_dirpath,
                       done_dirname_filepath, predict_batch_size=4, is_show=True):

    os.makedirs(output_dirpath, exist_ok=True)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = init_detector(config_file, model_path, device=device)

    done_file = open(done_dirname_filepath, 'a+', encoding='utf-8')
    done_file.seek(0)
    done_dirname = set(done_file.read().splitlines())
    for img_dir in os.listdir(img_path):
        if img_dir in done_dirname:
            continue
        print('processing img_dir: {}'.format(img_dir))
        img_dirpath = os.path.join(img_path, img_dir)
        save_det_result_path = os.path.join(output_dirpath, img_dir)
        if os.path.isdir(img_dirpath):
            img_fns = os.listdir(img_dirpath)
            img_paths = [os.path.join(img_dirpath, item) for item in img_fns]
        else:
            img_paths = [img_dirpath]

        img_paths = [img_paths[i:i+predict_batch_size] for i in range(0, len(img_paths), predict_batch_size)]

        for si, sub_img_paths in tqdm(enumerate(img_paths)):
            try:
                results = inference_detector(model, sub_img_paths)
                for ri, result in enumerate(results):
                    curr_img_path = sub_img_paths[ri]
                    basename = os.path.basename(curr_img_path).split('.')[0]
                    save_dirpath = os.path.join(save_det_result_path, basename)

                    os.makedirs(save_dirpath, exist_ok=True)
                    dst_json_filepath = os.path.join(save_dirpath, basename+'_det_result.json')
                    if os.path.exists(dst_json_filepath) and os.path.getsize(dst_json_filepath) != 0:
                        continue

                    if is_show:
                        img = mmcv.imread(curr_img_path)
                        img = img.copy()
                        model.show_result(img, result, show=True)

                    segms, bboxes, labels, scores = get_outputs(model, result, threshold=0.5)
                    if len(bboxes) == 0:
                        continue

                    pick = det_utils.non_max_suppression_iou_intersect(bboxes, scores, 0.3, 0.5)

                    scores = scores[pick]
                    bboxes = bboxes[pick, :]
                    labels = labels[pick]
                    if segms is not None:
                        segms = segms[pick, ...]

                    per_img_result_dict = {}
                    per_img_result_dict['img_dirpath'] = img_dirpath
                    per_img_result_dict['img_name'] = os.path.basename(curr_img_path)
                    result_list = []

                    for si, score in enumerate(scores):
                        box_dict = {}
                        box = bboxes[si]
                        segm = segms[si]
                        label = labels[si]

                        outs = cv2.findContours((segm * 255).astype(np.uint8), cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_SIMPLE)
                        if len(outs) == 3:
                            img, contours, _ = outs[0], outs[1], outs[2]
                        elif len(outs) == 2:
                            contours, _ = outs[0], outs[1]

                        box_dict['bbox'] = box.tolist()
                        box_dict['segmentation'] = contours[0].reshape((-1, 2)).tolist()
                        box_dict['score'] = np.float(score)
                        box_dict['label'] = label
                        box_dict['sub_image_path'] = '{}_{}_{}_{:.2}_image.jpg'.format(basename, si, label, score)
                        box_dict['sub_segm_path'] = '{}_{}_{}_{:.2}_segm.jpg'.format(basename, si, label, score)
                        result_list.append(box_dict)

                        # sub_segm_path = os.path.join(save_dirpath,
                        #                              '{}_{}_{}_{:.2}_segm.jpg'.format(basename, si, label, score))
                        # sub_image_path = os.path.join(save_dirpath,
                        #                               '{}_{}_{}_{:.2}_image.jpg'.format(basename, si, label, score))
                        # mask = segm[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                        # im = Image.fromarray(mask)
                        # im.save(sub_segm_path)
                        #
                        # image = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
                        # mmcv.imwrite(image, sub_image_path)
                    per_img_result_dict['det_result'] = result_list
                    with open(dst_json_filepath, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(per_img_result_dict, indent=2))
                    # model.show_result(img, result, score_thr=0.5,
                    #                   out_file=os.path.join(save_dirpath, basename+'_det_result.jpg'))
            except Exception as e:
                print(e)
                print(sub_img_paths)
        done_dirname.add(img_dir)
        done_file.write(img_dir + '\n')
        done_file.flush()
    done_file.close()


def det_json_to_img_result(json_filepath, dst_dirpath, CLASSES=('shopsign', 'name')):

    with open(json_filepath, 'r', encoding='utf-8') as f:
        json_object = json.load(f)

    img_dirpath = json_object['img_dirpath']
    img_name = json_object['img_name']

    img_path = os.path.join(img_dirpath, img_name)

    img = mmcv.imread(img_path)
    img = img.copy()

    box_result = []
    segm_result = []
    labels = []
    for box_dict in json_object['det_result']:
        box = np.array(box_dict['bbox'])
        segm = np.array(box_dict['segmentation'])
        score = box_dict['score']
        label = box_dict['label']
        sub_image_path = os.path.join(dst_dirpath, box_dict['sub_image_path'])
        sub_segm_path = os.path.join(dst_dirpath, box_dict['sub_segm_path'])

        empty_img = np.zeros(img.shape, np.uint8)
        segm = cv2.drawContours(empty_img, [segm.reshape((-1, 1, 2))], -1, (255, 255, 255), cv2.FILLED)

        segm = segm[:, :, 0] != 0
        box_result.append(box.reshape(1, -1))

        mask = segm[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        im = Image.fromarray(mask)

        segm = segm.reshape((1, segm.shape[0], segm.shape[1]))
        segm_result.append(segm)
        labels.append(np.array([CLASSES.index(label)]))

        im.save(sub_segm_path)

        image = img[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
        mmcv.imwrite(image, sub_image_path)

    bboxes = np.vstack(box_result)
    labels = np.concatenate(labels)
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    outfile = os.path.join(dst_dirpath, os.path.basename(json_filepath).replace('.json', '.jpg'))
    dst = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms,
        class_names=CLASSES,
        score_thr=0.5,
        bbox_color=(72, 101, 241),
        text_color=(72, 101, 241),
        mask_color=None,
        thickness=2,
        font_size=13,
        win_name='',
        show=False,
        wait_time=0,
        out_file=outfile)


class ROIDetector(object):
    def __init__(self, args):
        self.args = args
        if args.use_gpu:
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        else:
            device = torch.device('cpu')
        self.device = device
        self.model = init_detector(args.config_file, args.checkpoint_file, device=device)
        self.roi_threshold = args.roi_threshold

    def __call__(self, img):
        ori_im = img.copy()
        result = inference_detector(self.model, ori_im)

        segms, bboxes, labels, scores = predict_utils.get_det_outputs(self.model, result, threshold=self.roi_threshold)

        if len(bboxes) == 0:
            return []

        pick = det_utils.non_max_suppression_iou_intersect(bboxes, scores, 0.3, 0.5)

        scores = scores[pick]
        bboxes = bboxes[pick, :]
        labels = labels[pick]
        if segms is not None:
            segms = segms[pick, ...]

        result_list = []
        for si, score in enumerate(scores):
            box_dict = {}
            box = bboxes[si]
            segm = segms[si]
            label = labels[si]

            outs = cv2.findContours((segm * 255).astype(np.uint8), cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
            if len(outs) == 3:
                ori_im, contours, _ = outs[0], outs[1], outs[2]
            elif len(outs) == 2:
                contours, _ = outs[0], outs[1]

            box_dict['bbox'] = box.tolist()
            box_dict['segmentation'] = contours[0].reshape((-1, 2)).tolist()
            box_dict['score'] = float(score)
            box_dict['label'] = label
            result_list.append(box_dict)

        return result_list


if __name__ == '__main__':
    part = '009'
    backbone = 'swin'
    model_type = 'cascade_mask_rcnn'
    checkpoint_file = '../output/cascade_mask_rcnn_swin_cls2/latest.pth'
    img_path = r'Z:\Data\baidu_street_view_new_shenzhen_history_split\baidu_street_view_new_shenzhen_panorama_{}'.format(part)  # 'H:/tencent_china_sample_image_split_1000'
    save_dirpath = r'Z:\Data\baidu_street_view_new_shenzhen_history_split_result_json\baidu_street_view_new_shenzhen_panorama_{}'.format(part)  # '../output/cascade_mask_rcnn_swin_det_tencent_1000'

    config_file = os.path.join('../output/cascade_mask_rcnn_swin_cls2', 'shopsign_{}_{}.py'.format(model_type, backbone))

    done_dirname_filepath = '../output/cascade_mask_rcnn_swin_cls2/baidu_street_view_new_shenzhen_history_panorama_{}_det_json_done.txt'.format(part)

    predict_multi_json(checkpoint_file, config_file, img_path, save_dirpath, done_dirname_filepath, predict_batch_size=4, is_show=False)
