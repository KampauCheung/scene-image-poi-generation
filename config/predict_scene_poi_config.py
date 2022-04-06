
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


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()

    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=500)

    # paras for filepath
    parser.add_argument("--img_path", type=str,
                        default='./demo/gt_131.jpg')
    parser.add_argument("--save_dirpath", type=str,
                        default='./demo/gt_131')

    # params for ROI detection
    parser.add_argument("--backbone", type=str, default='swin')
    parser.add_argument("--model_type", type=str, default='cascade_mask_rcnn')
    parser.add_argument("--config_file", type=str,
                        default='../output/cascade_mask_rcnn_swin_cls2/shopsign_cascade_mask_rcnn_swin.py')
    parser.add_argument("--checkpoint_file", type=str, default='../output/cascade_mask_rcnn_swin_cls2/latest.pth')
    parser.add_argument("--roi_threshold", type=float, default=0.5)

    # params for text detector
    parser.add_argument("--det_algorithm", type=str, default='DB')
    parser.add_argument("--det_model_dir", type=str, default='../output/det_r101_vd_db_ch/infer')
    parser.add_argument("--det_limit_side_len", type=float, default=960)
    parser.add_argument("--det_limit_type", type=str, default='max')

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="slow")
    # EAST parmas
    parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
    parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
    parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

    # SAST parmas
    parser.add_argument("--det_sast_score_thresh", type=float, default=0.5)
    parser.add_argument("--det_sast_nms_thresh", type=float, default=0.2)
    parser.add_argument("--det_sast_polygon", type=bool, default=False)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default='SRN')
    parser.add_argument("--rec_model_dir", type=str, default='../output/rec_r50_vd_srn_ch/infer')
    parser.add_argument("--rec_image_shape", type=str, default="1, 64, 256")
    parser.add_argument("--rec_char_type", type=str, default='ch')
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default="../models/ocr/PaddleOCR/ppocr/utils/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument(
        "--vis_font_path", type=str, default="../models/ocr/PaddleOCR/doc/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    parser.add_argument("--use_angle_cls", type=str2bool, default=False)

    # paras for ROI and text line classification
    parser.add_argument("--vlcls_checkpoint_file", type=str, default='../output/vl_cls_all_params/net.pkl')
    parser.add_argument("--roi_label_map", type=str, default=json.dumps(ROI_LABEL_MAP))
    parser.add_argument("--text_label_map", type=str, default=json.dumps(TEXT_LABEL_MAP))

    return parser.parse_args()
