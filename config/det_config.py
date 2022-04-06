
import argparse


# predict parameters
def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)

    parser.add_argument("--backbone", type=str, default='swin')
    parser.add_argument("--model_type", type=str, default='cascade_mask_rcnn')
    parser.add_argument("--config_file", type=str,
                        default='../output/cascade_mask_rcnn_swin_cls2/shopsign_cascade_mask_rcnn_swin.py')
    parser.add_argument("--checkpoint_file", type=str, default='../output/cascade_mask_rcnn_swin_cls2/latest.pth')

    parser.add_argument("--img_path", type=str,
                        default='../demo/000000_10041004150608152203900_left.jpg')
    parser.add_argument("--save_dirpath", type=str,
                        default='../demo')

    return parser.parse_args()
