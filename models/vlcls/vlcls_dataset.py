import torch
import os
import json
import labelme
import numpy as np
import torchvision


def collate_fn(batch):
    return tuple(zip(*batch))


class VlClsDataset(torch.utils.data.Dataset):
    def __init__(self, image_ano_dirpath, ROI_label_map, text_label_map):
        _list = os.listdir(image_ano_dirpath)
        self.image_ano_list = [os.path.join(image_ano_dirpath, item) for item in _list if item.endswith('.json')]
        self.ROI_label_map = ROI_label_map
        self.text_label_map = text_label_map
        self.loader = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

    def __getitem__(self, idx):
        json_path = self.image_ano_list[idx]
        data = json.load(open(json_path, 'rb'))
        img = labelme.utils.image.img_b64_to_arr(data['imageData'])
        flags = data["flags"]
        flag = list(flags.keys())[list(flags.values()).index(True)]
        ROI_label = self.ROI_label_map[flag]

        boxes, texts, labels = [], [], []

        for shape in data['shapes']:
            label_name = shape['label']
            labels.append(self.text_label_map[label_name])
            points = np.array(shape['points'])
            x1, x2 = min(points[:, 0]), max(points[:, 0])
            y1, y2 = min(points[:, 1]), max(points[:, 1])
            boxes.append([x1, y1, x2, y2])
            texts.append(shape['text'])

        num_objs = len(boxes)

        if len(boxes) == 0:
            print(json_path)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["texts"] = texts
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target['ROI_label'] = torch.as_tensor(ROI_label, dtype=torch.int64)

        return self.loader(img), target, texts

    def __len__(self):
        return len(self.image_ano_list)

