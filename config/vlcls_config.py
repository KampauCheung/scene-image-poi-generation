
import argparse
import json

ROI_LABEL_MAP = {
    "Signboard": 0,
    "Billboard": 1,
    "Streetsign": 2,
    "Others": 3
}

TEXT_LABEL_MAP = {
    "Prefix": 0,
    "Title": 1,
    "Subtitle": 2,
    "Address": 3,
    "Tag": 4,
    "Tel": 5,
    "Others": 6
}

train_data_dirpath = 'D:/vl_cls_dataset/vl_cls_dataset_train'
test_data_dirpath = 'D:/vl_cls_dataset/vl_cls_dataset_test'

simcse_model_dirpath = 'D:/py_project/SimCSE/result/rec-unsup-simcse-bert-base-chinese'
faster_rcnn_model_dirpath = '../../output/faster_rcnn_det_text/latest.pth'

max_seq_length = 32

model_mode = 'vl'  # image, text, or vl

batch_size = 1
save_latest_batch_num = 10
train_batch_num = 1000

save_model_dirpath = '../../output/vl_cls_all_params'
save_predict_label_dirpath = '../../output/vl_cls_all_params/transformer-1layer-cat/predict_labels.csv'
# save_predict_label_dirpath = '../output/language_cls_all_params/predict_labels.csv'

model_filepath = '../../output/vl_cls_all_params/best_accuracy.pth'

resume_from_checkpoint = True
resume_checkpoint = '../../output/vl_cls_all_params/best_accuracy.pth'

ROI_cls_num = len(ROI_LABEL_MAP)
text_cls_num = len(TEXT_LABEL_MAP)


# predict parameters
def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)

    # paras for ROI and text line classification
    parser.add_argument("--vlcls_checkpoint_file", type=str, default='../output/vl_cls_all_params/net.pkl')
    parser.add_argument("--roi_label_map", type=str, default=json.dumps(ROI_LABEL_MAP))
    parser.add_argument("--text_label_map", type=str, default=json.dumps(TEXT_LABEL_MAP))

    parser.add_argument("--drop_score", type=float, default=0.5)

    parser.add_argument("--img_dirpath", type=str,
                        default=r'Z:\Data\tencent_street_view_shenzhen_split\tencent_street_view_shenzhen_panorama_000')
    parser.add_argument("--save_dirpath", type=str,
                        default=r'Z:\Data\tencent_street_view_shenzhen_split_det_json\tencent_street_view_shenzhen_panorama_000')

    return parser.parse_args()



