
import torch
import os
import json
import numpy as np
from PIL import Image
import torchvision
from models.utils.engine import train_one_epoch, evaluate


def collate_fn(batch):
    return tuple(zip(*batch))


class TextDetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dirpath, label_filepaths):
        if not isinstance(label_filepaths, list):
            label_filepaths = [label_filepaths]

        self.anno_list = []
        self.img_list = []
        for filepath in label_filepaths:
            with open(os.path.join(dataset_dirpath, filepath), 'r', encoding='utf-8') as f:
                all_lines = f.read().splitlines()
                self.anno_list = self.anno_list + [item.split('\t')[1] for item in all_lines]
                self.img_list = self.img_list + [os.path.join(dataset_dirpath, item.split('\t')[0]) for item in all_lines]

        self.loader = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])

    def __getitem__(self, idx):

        json_str = self.anno_list[idx]
        img_path = self.img_list[idx]

        img = Image.open(img_path).convert('RGB')
        json_object = json.loads(json_str)

        boxes = []
        labels = []
        for obj in json_object:
            points = np.array(obj['points'])
            transcription = obj['transcription']
            if transcription == '###':
                continue
            labels.append(np.int(1))

            x1, x2 = min(points[:, 0]), max(points[:, 0])
            y1, y2 = min(points[:, 1]), max(points[:, 1])
            if x1 < 0 or y1 < 0 or x1 >= x2 or y1 >= y2:
                continue
            boxes.append([x1, y1, x2, y2])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except Exception as e:
            print(e)
            print(img_path)
            area = 0

        iscrowd = torch.zeros((len(json_object),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return self.loader(img), target

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 2 classes, background, text
    num_classes = 2

    dataset_dirpath = 'D:/ocr_dataset'
    save_model_dirpath = '../../output/faster_rcnn_det_text'
    os.makedirs(save_model_dirpath, exist_ok=True)

    train_filepaths = ['art_train_train_det.txt', 'icdar2017rctw_train_train_det.txt',
                       'lsvt_train_train_det.txt', 'rects_train_train_det.txt']

    test_filepaths = ['art_train_val_det.txt', 'icdar2017rctw_train_val_det.txt',
                      'lsvt_train_val_det.txt', 'rects_train_val_det.txt']

    dataset_train = TextDetDataset(dataset_dirpath, train_filepaths)
    dataset_test = TextDetDataset(dataset_dirpath, test_filepaths)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=4, shuffle=True,
        collate_fn=collate_fn)
    data_loader_test =torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False,
        collate_fn=collate_fn)

    # get the model using our helper function
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes, pretrained_backbone=True)

    model.load_state_dict(torch.load(os.path.join(save_model_dirpath, 'latest.pth')))
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # SGD
    optimizer = torch.optim.SGD(params, lr=0.0003,
                            momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    # cos学习率
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    # let's train it for   epochs
    num_epochs = 100

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        # engine.py的train_one_epoch函数将images和targets都.to(device)了
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        save_model_filepath = os.path.join(save_model_dirpath, 'latest.pth')
        torch.save(model.state_dict(), save_model_filepath)
        
        print('')
        print('==================================================')
        print('')

    print("That's it!")
