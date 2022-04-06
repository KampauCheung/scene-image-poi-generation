
import torch
import math
import sys
import time
import os

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report

from models.utils import utils
from config import vlcls_config
from vlcls_dataset import VlClsDataset, collate_fn
# from faster_rcnn_feature import FasterRCNNFeatureCls
from vlcls_model import FasterRCNNFeatureMultiClsModel, LanguageMultiClsModel, VisualLanguageMultiClsModel


@torch.no_grad()
def evaluate(model, data_loader, loss_func, device,  out_filepath, mode='vl'):
    """
    :param model:
    :param data_loader:
    :param loss_func:
    :param device:
    :param mode:
    :return: image, text, or vl
    """

    outfile = open(out_filepath, 'w', encoding='utf-8')
    outfile.write('ROI_Label:Predict,Text_Label:Predict\n')

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

        ROI_label = ROI_labels.cpu().numpy()
        ROI_pred = pred.cpu().numpy()

        ROI_strs = ['{}:{}'.format(ROI_label[i], ROI_pred[i]) for i in range(len(ROI_label))]

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

            text_label = text_label.cpu().numpy()
            text_pred = pred.cpu().numpy()

            text_str = ['{}:{}'.format(text_label[i], text_pred[i]) for i in range(len(text_label))]
            text_str = '_'.join(text_str)
            outfile.write(ROI_strs[ti] + ',' + text_str + '\n')

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

    outfile.flush()
    outfile.close()
    return acc_ROI, acc_text


def evaluate_model():
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

    # if vlmc_config.resume_from_checkpoint:

    model.load_state_dict(torch.load(vlcls_config.resume_checkpoint))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.00001, weight_decay=0.00001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0005,
    #                             momentum=0.9, weight_decay=0.0005)
    loss_func = torch.nn.CrossEntropyLoss()

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)

    acc_ROI, acc_text = evaluate(model, data_loader_test, loss_func, device=device,
                                 out_filepath=vlcls_config.save_predict_label_dirpath, mode=model_mode)


def calculate_evaluate_metrics(predict_filepath):
    infile = open(predict_filepath, 'r', encoding='utf-8')
    infile.readline()
    _line = infile.readline()
    ROI_label = []
    ROI_pred = []
    text_label = []
    text_pred = []

    ROI_label_text_dict = {}

    while _line:
        ROI_label_predict = _line.split(',')[0].strip()
        text_label_predict = _line.split(',')[1].strip()

        _label = ROI_label_predict.split(':')[0]
        _pred = ROI_label_predict.split(':')[1]
        ROI_label.append(_label)
        ROI_pred.append(_pred)

        text_label_predict_list = text_label_predict.split('_')

        for pair in text_label_predict_list:
            _t_label = pair.split(':')[0]
            _t_pred = pair.split(':')[1]
            text_label.append(_t_label)
            text_pred.append(_t_pred)

            if _label not in ROI_label_text_dict:
                ROI_label_text_dict[_label] = [[_t_label], [_t_pred]]
            else:
                ROI_label_text_dict[_label][0].append(_t_label)
                ROI_label_text_dict[_label][1].append(_t_pred)

        _line = infile.readline()

    # print(classification_report(text_label, text_pred, digits=6))
    # print(confusion_matrix(text_label, text_pred))

    # for key in ROI_label_text_dict:
    #     label, pred = ROI_label_text_dict[key]
    #     print('\n================{}================'.format(key))
    #     print(classification_report(label, pred, digits=6))
    #     print(confusion_matrix(label, pred))


def calculate_POI_generation_metrics(predict_filepath):
    infile = open(predict_filepath, 'r', encoding='utf-8')
    infile.readline()
    _line = infile.readline()

    signboard_dict = {'tp': 0, 'fn': 0, 'fp': 0}  # label 0
    streetsign_dict = {'tp': 0, 'fn': 0, 'fp': 0}  # label 2

    need_label_list = ['0', '2']

    while _line:
        ROI_label_predict = _line.split(',')[0].strip()
        text_label_predict = _line.split(',')[1].strip()

        _label = ROI_label_predict.split(':')[0]
        _pred = ROI_label_predict.split(':')[1]

        if _label not in need_label_list and _pred not in need_label_list:
            _line = infile.readline()
            continue

        if _label != _pred:
            if _label == '0':
                signboard_dict['fn'] += 1
            elif _label == '2':
                streetsign_dict['fn'] += 1
            _line = infile.readline()
            continue

        text_label_predict_list = text_label_predict.split('_')

        is_correct = True
        need_text_list = ['0', '1', '2']
        for pair in text_label_predict_list:
            _t_label = pair.split(':')[0]
            _t_pred = pair.split(':')[1]

            if _t_label in need_text_list or _t_pred in need_text_list:
                if _t_label != _t_pred:
                    is_correct = False
                    break

        if _label == '0':
            if is_correct:
                signboard_dict['tp'] += 1
            else:
                signboard_dict['fn'] += 1
        else:
            if is_correct:
                streetsign_dict['tp'] += 1
            else:
                streetsign_dict['fn'] += 1

        _line = infile.readline()

    signboard_recall = signboard_dict['tp'] / (signboard_dict['tp'] + signboard_dict['fn'])
    streetsign_recall = streetsign_dict['tp'] / (streetsign_dict['tp'] + streetsign_dict['fn'])
    all_recall = (signboard_dict['tp'] + streetsign_dict['tp']) / (signboard_dict['tp'] + signboard_dict['fn'] + streetsign_dict['tp'] + streetsign_dict['fn'])

    print('signboard recall = {}, streetsign recall = {}, all recall = {}'.format(signboard_recall, streetsign_recall, all_recall))


if __name__ == "__main__":

    # evaluate_model()

    # calculate_evaluate_metrics('../../output/vl_cls_all_params/transformer-1layer-cat/predict_labels.csv')

    calculate_POI_generation_metrics('../../output/vl_cls_all_params/transformer-1layer-cat/predict_labels.csv')

