
import time
import numpy as np
import torch
from torch import nn
import math
import sys
import torch.nn.functional as F
import torchvision
from typing import Tuple, List, Dict, Optional, Union
from transformers import AutoModel, AutoTokenizer

# from faster_rcnn_feature import FasterRCNNFeature
from config import vlcls_config
from models.vlcls.vlcls_dataset import VlClsDataset, collate_fn
from models.utils import utils


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


class MLPClassifier(nn.Module):
    def __init__(self, in_channels, cls_num):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Linear(in_channels, in_channels)
        self.classifier = nn.Linear(in_channels, cls_num)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.classifier(out)
        return out


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
    def __init__(self, faster_rcnn_model_path=None, out_dim=768):
        super(FasterRCNNFeature, self).__init__()
        fasterrcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
                                                                                pretrained_backbone=False,
                                                                                num_classes=2)
        if faster_rcnn_model_path:
            fasterrcnn_model.load_state_dict(torch.load(faster_rcnn_model_path))
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

        # with torch.no_grad():
        #     images, _ = self.transform(images)
        #     features = self.backbone(images.tensors)

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


class FasterRCNNFeatureMultiClsModel(nn.Module):
    def __init__(self, ROI_cls_num, text_cls_num, max_seq_length=32,
                 representation_dim=768, faster_rcnn_model_path=None):
        super(FasterRCNNFeatureMultiClsModel, self).__init__()
        self.representation_dim = representation_dim
        self.feature_extractor = FasterRCNNFeature(faster_rcnn_model_path=faster_rcnn_model_path,
                                                   out_dim=representation_dim)
        self.max_seq_length = max_seq_length
        self.special_embeddings = nn.Embedding(1, representation_dim)  # [CLS]
        self.padding_embeddings = torch.zeros((max_seq_length, representation_dim))
        encoder_layer = nn.TransformerEncoderLayer(representation_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.ROI_cls_num = ROI_cls_num
        self.text_cls_num = text_cls_num

        self.ROI_classifier = MLPClassifier(representation_dim, ROI_cls_num)
        self.text_classifier = MLPClassifier(representation_dim, text_cls_num)

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
            if seq_length >= self.max_seq_length - 1:
                _src = torch.cat([img_feat.unsqueeze(dim=0), box_feat[:self.max_seq_length-1,]]).unsqueeze(dim=1)

                src_key_padding_mask.append(torch.zeros((1, self.max_seq_length), dtype=torch.bool))
                text_ids.append(list(range(text_num, text_num+self.max_seq_length-1)))
                text_num += self.max_seq_length-1
            else:
                need_pad_length = self.max_seq_length - 1 - seq_length
                _src = torch.cat([img_feat.unsqueeze(dim=0), box_feat,
                                  self.padding_embeddings[:need_pad_length,].to(img_features.device)]).unsqueeze(dim=1)
                src_key_padding_mask.append(torch.tensor([[0]*(1+seq_length) + [1]*need_pad_length], dtype=torch.bool))
                text_ids.append(list(range(text_num, text_num+seq_length)))
                text_num += seq_length
            src.append(_src)

        src = torch.cat(src, dim=1)
        src_key_padding_mask = torch.cat(src_key_padding_mask, dim=0).to(img_features.device)

        transformer_out = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        ROI_features = transformer_out[0,...]
        ROI_logits = self.ROI_classifier(ROI_features)

        active_text_ids = ~ src_key_padding_mask[:,1:].contiguous().view(-1)
        text_features = transformer_out[1:,...].view(-1, self.representation_dim)
        active_text_features = text_features[active_text_ids]
        active_text_logits = self.text_classifier(active_text_features)

        active_text_logits = [active_text_logits[idx] for idx in text_ids]

        return ROI_logits, active_text_logits


class LanguageFeature(nn.Module):
    def __init__(self, simcse_model_path):
        super(LanguageFeature, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(simcse_model_path)
        self.model = AutoModel.from_pretrained(simcse_model_path)
        self.batch_size = 32
        self.max_length = 128
        self.pooler = "cls_before_pooler"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, sentence: Union[str, List[str]], device: str =None):
        target_device = self.device if device is None else device

        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = []

        total_batch = len(sentence) // self.batch_size + (1 if len(sentence) % self.batch_size > 0 else 0)
        for batch_id in range(total_batch):
            inputs = self.tokenizer(
                sentence[batch_id*self.batch_size:(batch_id+1)*self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            outputs = self.model(**inputs, return_dict=True)
            if self.pooler == "cls":
                embeddings = outputs.pooler_output
            elif self.pooler == "cls_before_pooler":
                embeddings = outputs.last_hidden_state[:, 0]
            else:
                raise NotImplementedError
            embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
            embedding_list.append(embeddings)
        embeddings = torch.cat(embedding_list, 0)

        if single_sentence:
            embeddings = embeddings[0]

        return embeddings


class LanguageMultiClsModel(nn.Module):
    def __init__(self, simcse_model_path, ROI_cls_num, text_cls_num,
                 max_seq_length=32, representation_dim=768):
        super(LanguageMultiClsModel, self).__init__()
        self.sentence_embedding_model = LanguageFeature(simcse_model_path)
        self.max_seq_length = max_seq_length
        self.representation_dim = representation_dim

        self.special_embeddings = nn.Embedding(1, representation_dim)  # [CLS]
        self.padding_embeddings = torch.zeros((max_seq_length, representation_dim))
        encoder_layer = nn.TransformerEncoderLayer(representation_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.ROI_cls_num = ROI_cls_num
        self.text_cls_num = text_cls_num

        self.ROI_classifier = MLPClassifier(representation_dim, ROI_cls_num)
        self.text_classifier = MLPClassifier(representation_dim, text_cls_num)

    def forward(self, texts):
        all_texts, text_ids = [], []
        text_num = 0
        for text in texts:
            all_texts = all_texts + text
            text_ids.append(list(range(text_num, text_num+len(text))))
            text_num += len(text)
        text_features = self.sentence_embedding_model(all_texts).to(self.sentence_embedding_model.device)

        src, src_key_padding_mask, mask_text_ids = [], [], []
        text_num = 0
        for ti, text in enumerate(texts):
            text_feat = text_features[text_ids[ti]]
            seq_length = len(text)

            if seq_length >= self.max_seq_length - 1:
                _src = torch.cat([self.special_embeddings(torch.tensor(0).to(text_feat.device)).unsqueeze(dim=0),
                                  text_feat[:self.max_seq_length-1,]]).unsqueeze(dim=1)
                src_key_padding_mask.append(torch.zeros((1, self.max_seq_length), dtype=torch.bool))
                mask_text_ids.append(list(range(text_num, text_num+self.max_seq_length-1)))
                text_num += self.max_seq_length-1
            else:
                need_pad_length = self.max_seq_length - 1 - seq_length
                _src = torch.cat([self.special_embeddings(torch.tensor(0).to(text_feat.device)).unsqueeze(dim=0), text_feat,
                                  self.padding_embeddings[:need_pad_length,].to(text_feat.device)]).unsqueeze(dim=1)
                src_key_padding_mask.append(torch.tensor([[0]*(1+seq_length) + [1]*need_pad_length], dtype=torch.bool))
                mask_text_ids.append(list(range(text_num, text_num+seq_length)))
                text_num += seq_length
            src.append(_src)
        src = torch.cat(src, dim=1)
        src_key_padding_mask = torch.cat(src_key_padding_mask, dim=0).to(text_features.device)

        transformer_out = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        ROI_features = transformer_out[0,...]
        ROI_logits = self.ROI_classifier(ROI_features)

        active_text_ids = ~ src_key_padding_mask[:,1:].contiguous().view(-1)
        text_features = transformer_out[1:,...].view(-1, self.representation_dim)
        active_text_features = text_features[active_text_ids]
        active_text_logits = self.text_classifier(active_text_features)

        # text_features = transformer_out[1:,...].view(len(texts), -1, self.representation_dim)
        # text_logits = self.text_classifier(text_features)
        # active_text_logits = text_logits.view(-1, self.text_cls_num)[active_text_ids]

        active_text_logits = [active_text_logits[idx] for idx in mask_text_ids]

        return ROI_logits, active_text_logits


class VisualLanguageMultiClsModel(nn.Module):
    def __init__(self, simcse_model_path, ROI_cls_num, text_cls_num,
                 max_seq_length=32, representation_dim=768, faster_rcnn_model_path=None):
        super(VisualLanguageMultiClsModel, self).__init__()

        self.text_feature_extractor = LanguageFeature(simcse_model_path)
        self.representation_dim = representation_dim
        self.visual_feature_extractor = FasterRCNNFeature(faster_rcnn_model_path=faster_rcnn_model_path,
                                                          out_dim=representation_dim)

        self.max_seq_length = max_seq_length

        self.special_embeddings = nn.Embedding(1, representation_dim)  # [CLS]
        self.padding_embeddings = torch.zeros((max_seq_length, representation_dim))

        encoder_layer = nn.TransformerEncoderLayer(representation_dim*2, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.ROI_cls_num = ROI_cls_num
        self.text_cls_num = text_cls_num

        self.ROI_classifier = MLPClassifier(representation_dim*2, ROI_cls_num)
        self.text_classifier = MLPClassifier(representation_dim*2, text_cls_num)

    # [CLS] [roi1,text1] [roi2, text2] [roi3, text3] [SEP]
    # put img_features to [CLS], roi_features = [visual_feature, location_feature]
    def forward(self, images, targets, texts):
        all_texts, text_ids = [], []
        text_num = 0
        for text in texts:
            all_texts = all_texts + text
            text_ids.append(list(range(text_num, text_num+len(text))))
            text_num += len(text)
        text_features = self.text_feature_extractor(all_texts).to(self.text_feature_extractor.device)

        visual_features = self.visual_feature_extractor(images, targets)
        img_features = visual_features['img_features']
        box_features = visual_features['box_features']
        img_location_features = visual_features['img_location_features']
        box_location_features = visual_features['box_location_features']

        img_features = img_features + img_location_features

        src, src_key_padding_mask, mask_text_ids = [], [], []
        text_num = 0
        for ti, text in enumerate(texts):
            text_feat = text_features[text_ids[ti]]
            box_feat = box_features[ti] + box_location_features[ti]
            img_feat = img_features[ti]

            seq_length = len(text)

            if seq_length >= self.max_seq_length - 1:
                src_text = torch.cat([self.special_embeddings(torch.tensor(0).to(text_feat.device)).unsqueeze(dim=0),
                                      text_feat[:self.max_seq_length-1,]]).unsqueeze(dim=1)
                src_img = torch.cat([img_feat.unsqueeze(dim=0), box_feat[:self.max_seq_length-1,]]).unsqueeze(dim=1)

                src_key_padding_mask.append(torch.zeros((1, self.max_seq_length), dtype=torch.bool))
                mask_text_ids.append(list(range(text_num, text_num+self.max_seq_length-1)))
                text_num += self.max_seq_length-1
            else:
                need_pad_length = self.max_seq_length - 1 - seq_length
                src_text = torch.cat([self.special_embeddings(torch.tensor(0).to(text_feat.device)).unsqueeze(dim=0), text_feat,
                                      self.padding_embeddings[:need_pad_length,].to(text_feat.device)]).unsqueeze(dim=1)
                src_img = torch.cat([img_feat.unsqueeze(dim=0), box_feat,
                                     self.padding_embeddings[:need_pad_length,].to(img_features.device)]).unsqueeze(dim=1)

                src_key_padding_mask.append(torch.tensor([[0]*(1+seq_length) + [1]*need_pad_length], dtype=torch.bool))
                mask_text_ids.append(list(range(text_num, text_num+seq_length)))
                text_num += seq_length
            src.append(torch.cat([src_img, src_text], dim=2))
            # src.append(src_img + src_text)
        src = torch.cat(src, dim=1)
        src_key_padding_mask = torch.cat(src_key_padding_mask, dim=0).to(text_features.device)

        transformer_out = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        ROI_features = transformer_out[0,...]
        ROI_logits = self.ROI_classifier(ROI_features)

        active_text_ids = ~ src_key_padding_mask[:,1:].contiguous().view(-1)
        text_features = transformer_out[1:,...].view(-1, self.representation_dim*2)
        active_text_features = text_features[active_text_ids]
        active_text_logits = self.text_classifier(active_text_features)

        # text_features = transformer_out[1:,...].view(len(texts), -1, self.representation_dim*2)
        # text_logits = self.text_classifier(text_features)
        # active_text_logits = text_logits.view(-1, self.text_cls_num)[active_text_ids]

        active_text_logits = [active_text_logits[idx] for idx in mask_text_ids]

        return ROI_logits, active_text_logits


if __name__ == '__main__':

    simcse_model_path = vlcls_config.simcse_model_dirpath
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    dataset_train = VlClsDataset(vlcls_config.train_data_dirpath,
                                 vlcls_config.ROI_LABEL_MAP, vlcls_config.TEXT_LABEL_MAP)
    dataset_test = VlClsDataset(vlcls_config.test_data_dirpath,
                                vlcls_config.ROI_LABEL_MAP, vlcls_config.TEXT_LABEL_MAP)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=vlcls_config.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=vlcls_config.batch_size, shuffle=False, num_workers=0,
        collate_fn=collate_fn)

    # model = LanguageMultiClsModel(simcse_model_path, ROI_cls_num=vlcls_config.ROI_cls_num,
    #                               text_cls_num=vlcls_config.text_cls_num, max_seq_length=8)
    # model = VisualLanguageMultiClsModle(simcse_model_path, ROI_cls_num=vlcls_config.ROI_cls_num,
    #                                     text_cls_num=vlcls_config.text_cls_num, max_seq_length=4)
    model = LanguageFeature(simcse_model_path)
    model.to(device)

    # images, targets, texts = next(iter(data_loader_train))
    # images = list(image.to(device) for image in images)
    # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

