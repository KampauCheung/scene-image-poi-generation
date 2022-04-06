
import warnings
from collections import namedtuple
from string import Template

import numpy as np
import torch
from torch import nn
from torch.autograd import Function
from torchvision.ops import RoIPool, RoIAlign

from sentence_transformers import models, SentenceTransformer
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision
import torch.nn.functional as F
import os
import json
import labelme
from vlcls_dataset import VlClsDataset, collate_fn
from typing import Tuple, List, Dict, Optional, Union
from simcse import SimCSE
import math
import sys
import time

from models.utils import utils
from config import vlcls_config


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


def resize_boxes(boxes, original_size, new_size):
    # type: (Tensor, List[int], List[int]) -> Tensor
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios[0], ratios[1]
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def get_normalize_boxes_feature(boxes, original_size):
    # type: (Tensor, List[int]) -> Tensor

    ratios = [torch.tensor(1.0, dtype=torch.float32, device=boxes.device) /
              torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
              for s_orig in original_size]

    ratio_height, ratio_width = ratios[0], ratios[1]
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height

    w = xmax - xmin
    h = ymax - ymin
    a = w * h

    return torch.stack((xmin, ymin, xmax, ymax, w, h, a), dim=1)


class FasterRCNNFeature(nn.Module):
    def __init__(self, out_dim=768):
        super(FasterRCNNFeature, self).__init__()
        fasterrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.backbone = fasterrcnn_model.backbone
        self.transform = fasterrcnn_model.transform

        self.box_roi_pool = fasterrcnn_model.roi_heads.box_roi_pool

        out_channels = self.backbone.out_channels
        resolution = self.box_roi_pool.output_size[0]
        self.box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            out_dim)
        self.feature_head = TwoMLPHead(
            out_channels * resolution ** 2,
            out_dim
        )

        self.box_location_head = TwoMLPHead(
            7,
            out_dim
        )

    def forward(self, images, targets=None):
        original_image_sizes: List[Tuple[int, int]] = []
        img_areas, img_locations = [], []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
            img_areas.append(val[0] * val[1])
            img_locations.append([0, 0, 1, 1, 1, 1, 1])  # x1, y1, x2, y2, h, w, a

        img_locations = torch.tensor(img_locations, dtype=torch.float32, device=images[0].device)
        images, _ = self.transform(images)
        features = self.backbone(images.tensors)

        boxes, boxes_ids, boxes_locations = [], [], []
        boxes_num = 0
        for ti, target in enumerate(targets):
            _box = resize_boxes(target['boxes'], list(original_image_sizes[ti]), list(images.image_sizes[ti]))
            boxes.append(_box)
            boxes_ids.append(list(range(boxes_num, boxes_num+len(_box))))
            boxes_num += len(_box)

            _box_location = get_normalize_boxes_feature(target['boxes'], list(original_image_sizes[ti]))
            boxes_locations.append(_box_location)
        boxes_locations = torch.cat(boxes_locations, dim=0)

        box_features = self.box_roi_pool(features, boxes, images.image_sizes)
        box_features = self.box_head(box_features)
        box_features = [box_features[idx, :] for idx in boxes_ids]
        box_location_features = self.box_location_head(boxes_locations)
        box_location_features = [box_location_features[idx, :] for idx in boxes_ids]

        img_boxes = [torch.as_tensor([[0, 0, val[0]-1, val[1]-1]], dtype=torch.float32).to(images.tensors.device) for val in images.image_sizes]
        img_features = self.box_roi_pool(features, img_boxes, images.image_sizes)
        img_features = self.feature_head(img_features)
        img_location_features = self.box_location_head(img_locations)

        out_features = {}
        out_features['img_features'] = img_features
        out_features['box_features'] = box_features
        out_features['img_location_features'] = img_location_features
        out_features['box_location_features'] = box_location_features

        return out_features


class FasterRCNNFeatureCls(nn.Module):
    def __init__(self, max_seq_length, ROI_cls_num, text_cls_num,
                 representation_dim=768):
        super(FasterRCNNFeatureCls, self).__init__()
        self.representation_dim = representation_dim
        self.feature_extractor = FasterRCNNFeature(out_dim=representation_dim)
        self.max_seq_length = max_seq_length
        self.special_embeddings = nn.Embedding(1, representation_dim)  # [CLS] and [SEP]
        self.padding_embeddings = torch.zeros((max_seq_length, representation_dim))
        encoder_layer = nn.TransformerEncoderLayer(representation_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.ROI_cls_num = ROI_cls_num
        self.text_cls_num = text_cls_num

        self.ROI_classifier = nn.Linear(representation_dim, ROI_cls_num)
        self.text_classifier = nn.Linear(representation_dim, text_cls_num)

    # [CLS] sent1 set2 sent3 [SEP]
    # put img_features to [CLS]
    def forward(self, images, targets=None):
        features = self.feature_extractor(images, targets)

        img_features = features['img_features']
        box_features = features['box_features']
        img_location_features = features['img_location_features']
        box_location_features = features['box_location_features']

        img_features = img_features + img_location_features

        src, src_key_padding_mask, text_ids = [], [], []
        text_num = 0
        for ii, img_feat in enumerate(img_features):
            box_feat = box_features[ii] + box_location_features[ii]
            seq_length = box_feat.shape[0]
            if seq_length >= self.max_seq_length - 2:
                _src = torch.cat([img_feat.unsqueeze(dim=0), box_feat[:self.max_seq_length-2,],
                                 self.special_embeddings(torch.tensor(0).to(img_feat.device)).unsqueeze(dim=0)]).unsqueeze(dim=1)
                src_key_padding_mask.append(torch.zeros((1, self.max_seq_length), dtype=torch.bool))
                text_ids.append(list(range(text_num, text_num+self.max_seq_length-1)))
                text_num += self.max_seq_length-1
            else:
                need_pad_length = self.max_seq_length - 2 - seq_length
                _src = torch.cat([img_feat.unsqueeze(dim=0), box_feat,
                                  self.special_embeddings(torch.tensor(0).to(img_feat.device)).unsqueeze(dim=0),
                                  self.padding_embeddings[:need_pad_length,].to(img_features.device)]).unsqueeze(dim=1)
                src_key_padding_mask.append(torch.tensor([[0]*(2+seq_length) + [1]*need_pad_length], dtype=torch.bool))
                text_ids.append(list(range(text_num, text_num+seq_length+1)))
                text_num += seq_length+1
            src.append(_src)

        src = torch.cat(src, dim=1)
        src_key_padding_mask = torch.cat(src_key_padding_mask, dim=0).to(img_features.device)

        transformer_out = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        ROI_features = transformer_out[0,...]
        ROI_logits = self.ROI_classifier(ROI_features)

        text_features = transformer_out[1:,...].view(len(box_features), -1, self.representation_dim)
        text_logits = self.text_classifier(text_features)

        active_text_ids = ~ src_key_padding_mask[:,1:].contiguous().view(-1)
        active_text_logits = text_logits.view(-1, self.text_cls_num)[active_text_ids]
        active_text_logits = [active_text_logits[idx[:-1], :] for idx in text_ids]

        return ROI_logits, active_text_logits


def train_one_epoch(model, optimizer, loss_func, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None

    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets, texts in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        ROI_logits, text_logits = model(images, targets)

        ROI_labels = torch.cat([label['ROI_label'].unsqueeze(dim=0) for label in targets])
        loss_ROI = loss_func(ROI_logits, ROI_labels)

        text_labels = [label['labels'] for label in targets]
        loss_text = 0
        for ti, text_label in enumerate(text_labels):
            seq_length = text_logits[ti].shape[0]
            text_label = text_label[0:min(text_label.shape[0], seq_length)]
            loss_text += loss_func(text_logits[ti], text_label)
        loss_text = loss_text / len(text_labels)

        loss_dict = {'loss_ROI': loss_ROI, 'loss_text': loss_text}
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, loss_func, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    batch_num = len(data_loader.batch_sampler)
    sample_num = len(data_loader.batch_sampler.sampler)

    loss_ROI_all, loss_text_all = 0., 0.
    acc_ROI, acc_text = 0., 0.
    all_text_num = 0.
    for images, targets, texts in metric_logger.log_every(data_loader, 10, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        ROI_logits, text_logits = model(images, targets)

        model_time = time.time() - model_time

        evaluator_time = time.time()
        ROI_labels = torch.cat([label['ROI_label'].unsqueeze(dim=0) for label in targets])
        loss_ROI = loss_func(ROI_logits, ROI_labels)
        loss_ROI_all += loss_ROI

        pred = torch.max(ROI_logits, 1)[1]
        num_correct = (pred == ROI_labels).sum()
        acc_ROI += num_correct.data.to(cpu_device).numpy().reshape(-1)[0]

        text_labels = [label['labels'] for label in targets]
        loss_text = 0
        for ti, text_label in enumerate(text_labels):
            seq_length = text_logits[ti].shape[0]
            text_label = text_label[0:min(text_label.shape[0], seq_length)]
            loss_text += loss_func(text_logits[ti], text_label)

            pred = torch.max(text_logits[ti], 1)[1]
            num_correct = (pred == text_label).sum()
            acc_text += num_correct.data.to(cpu_device).numpy().reshape(-1)[0]
            all_text_num += min(text_label.shape[0], seq_length)

        loss_text = loss_text / len(text_labels)
        loss_text_all += loss_text
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    loss_ROI_all /= batch_num
    loss_text_all /= batch_num
    acc_ROI /= sample_num
    acc_text /= sample_num
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("Accuracy:  loss_ROI: {},  loss_text: {},  acc_ROI: {},  acc_text: {}".format(loss_ROI_all, loss_text_all, acc_ROI, acc_text))

    return acc_ROI + acc_text


if __name__ == '__main__':
    save_model_dirpath = vlcls_config.save_model_dirpath
    os.makedirs(save_model_dirpath, exist_ok=True)

    save_latest_model_path = os.path.join(save_model_dirpath, 'latest.pth')
    save_best_model_path = os.path.join(save_model_dirpath, 'best_accuracy.pth')

    save_latest_batch_num = vlcls_config.save_latest_batch_num

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = vlcls_config.batch_size

    dataset_train = VlClsDataset(vlcls_config.train_data_dirpath,
                                 vlcls_config.ROI_LABEL_MAP, vlcls_config.TEXT_LABEL_MAP)
    dataset_test = VlClsDataset(vlcls_config.test_data_dirpath,
                                vlcls_config.ROI_LABEL_MAP, vlcls_config.TEXT_LABEL_MAP)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    model = FasterRCNNFeatureCls(max_seq_length=8, ROI_cls_num=vlcls_config.ROI_cls_num,
                                 text_cls_num=vlcls_config.text_cls_num)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.005,
    #                             momentum=0.9, weight_decay=0.0005)
    loss_func = torch.nn.CrossEntropyLoss()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    best_accuracy = 0
    for epoch in range(vlcls_config.train_batch_num):
        print('epoch {}'.format(epoch))
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, loss_func, data_loader_train, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset

        if epoch % save_latest_batch_num == 0:
            torch.save(model.state_dict(), save_latest_model_path)

        accuracy = evaluate(model, data_loader_test, loss_func, device=device)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), save_best_model_path)
