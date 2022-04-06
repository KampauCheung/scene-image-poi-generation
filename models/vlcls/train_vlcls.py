
import torch
import math
import sys
import time
import os

from models.utils import utils
from config import vlcls_config
from vlcls_dataset import VlClsDataset, collate_fn
# from faster_rcnn_feature import FasterRCNNFeatureCls
from vlcls_model import FasterRCNNFeatureMultiClsModel, LanguageMultiClsModel, VisualLanguageMultiClsModel


def train_one_epoch(model, optimizer, loss_func, data_loader, device, epoch, print_freq, mode='vl'):
    """

    :param model:
    :param optimizer:
    :param loss_func:
    :param data_loader:
    :param device:
    :param epoch:
    :param print_freq:
    :param mode: image, text, or vl
    :return:
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    cpu_device = torch.device("cpu")

    lr_scheduler = None

    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        # lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    batch_num = len(data_loader.batch_sampler)
    sample_num = len(data_loader.batch_sampler.sampler)

    loss_ROI_all, loss_text_all = 0., 0.
    acc_ROI, acc_text = 0., 0.
    all_text_num = 0.
    for images, targets, texts in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if mode == 'image':
            ROI_logits, text_logits = model(images, targets)
        elif mode == 'text':
            ROI_logits, text_logits = model(texts)
        else:
            ROI_logits, text_logits = model(images, targets, texts)

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

        loss_dict = {'loss_ROI': loss_ROI, 'loss_text': loss_text * 2}
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

        loss_text = loss_text / len(text_labels)
        loss_text_all += loss_text
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    loss_ROI_all /= batch_num
    loss_text_all /= batch_num
    acc_ROI /= sample_num
    acc_text /= all_text_num
    print("Training accuracy:  loss_ROI: {},  loss_text: {},  acc_ROI: {},  acc_text: {}".format(loss_ROI_all, loss_text_all, acc_ROI, acc_text))

    return metric_logger


@torch.no_grad()
def evaluate(model, data_loader, loss_func, device, mode='vl'):
    """
    :param model:
    :param data_loader:
    :param loss_func:
    :param device:
    :param mode:
    :return: image, text, or vl
    """

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
    for images, targets, texts in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        if mode == 'image':
            ROI_logits, text_logits = model(images, targets)
        elif mode == 'text':
            ROI_logits, text_logits = model(texts)
        else:
            ROI_logits, text_logits = model(images, targets, texts)

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
    acc_text /= all_text_num
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("Eval accuracy:  loss_ROI: {},  loss_text: {},  acc_ROI: {},  acc_text: {}".format(loss_ROI_all, loss_text_all, acc_ROI, acc_text))

    return acc_ROI, acc_text


if __name__ == "__main__":
    save_model_dirpath = vlcls_config.save_model_dirpath
    os.makedirs(save_model_dirpath, exist_ok=True)

    model_mode = vlcls_config.model_mode
    simcse_model_dirpath = vlcls_config.simcse_model_dirpath
    max_seq_length = vlcls_config.max_seq_length

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

    if model_mode == 'image':
        model = FasterRCNNFeatureMultiClsModel(ROI_cls_num=vlcls_config.ROI_cls_num,
                                               text_cls_num=vlcls_config.text_cls_num,
                                               max_seq_length=max_seq_length,
                                               faster_rcnn_model_path=vlcls_config.faster_rcnn_model_dirpath)
    elif model_mode == 'text':
        model = LanguageMultiClsModel(simcse_model_path=simcse_model_dirpath,
                                      max_seq_length=max_seq_length,
                                      ROI_cls_num=vlcls_config.ROI_cls_num,
                                      text_cls_num=vlcls_config.text_cls_num)
    else:
        model = VisualLanguageMultiClsModel(simcse_model_path=simcse_model_dirpath,
                                            max_seq_length=max_seq_length,
                                            ROI_cls_num=vlcls_config.ROI_cls_num,
                                            text_cls_num=vlcls_config.text_cls_num,
                                            faster_rcnn_model_path=vlcls_config.faster_rcnn_model_dirpath)

    if vlcls_config.resume_from_checkpoint:
        model.load_state_dict(torch.load(vlcls_config.resume_checkpoint))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=0.00001, weight_decay=0.00001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.000005,
                                momentum=0.9, weight_decay=0.0005)
    loss_func = torch.nn.CrossEntropyLoss()

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)

    best_accuracy, best_acc_ROI, best_acc_text, best_epoch = 0, 0, 0, 0
    for epoch in range(vlcls_config.train_batch_num):
        print('epoch {}'.format(epoch))
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, loss_func, data_loader_train, device, epoch, print_freq=100, mode=model_mode)
        # update the learning rate
        # lr_scheduler.step()
        # evaluate on the test dataset

        if epoch % save_latest_batch_num == 0:
            torch.save(model.state_dict(), save_latest_model_path)

        acc_ROI, acc_text = evaluate(model, data_loader_test, loss_func, device=device, mode=model_mode)
        accuracy = acc_ROI + acc_text
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_acc_ROI = acc_ROI
            best_acc_text = acc_text
            best_epoch = epoch
            torch.save(model.state_dict(), save_best_model_path)
        print("Best accuracy:  acc_ROI: {},  acc_text: {}, epoch: {}".format(best_acc_ROI, best_acc_text, best_epoch))
