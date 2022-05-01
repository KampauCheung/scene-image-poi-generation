# scene-image-poi-generation

This repository implements a deep learning-based three-stage framework to automatically generate POI data sets from scene images. This work has been accepted as the paper '[Deep-learning generation of POI data with scene images](https://www.sciencedirect.com/science/article/pii/S0924271622000995)' in ISPRS Journal of Photogrammetry and Remote Sensing. 

We thank [MMDetection](https://github.com/open-mmlab/mmdetection) and [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) for their implements for object detection and instance segmentation. Meanwhile, we also thank [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for their implements for scene text recognition. 

## Requirements
- MMDetection
- Paddle
- Transformers
- PyTorch

## Checkpoints
Checkpoints can be downloaded in [Google Driver](https://drive.google.com/file/d/1U_g8C7iTz-JSkof4L0Vtp0c1TJuM797R/view?usp=sharing).

## Usage
### Inference
- ROI sementation `tools\det_predict.py`
- Scene text recognition `tools\ocr_predict.py`
- ROI and text lines classification `tools\vlcls_predict.py`
- Entire framework `tools\scene_poi_predict.py`

### Training
- ROI sementation `models/det/tools/train.py`
- Scene text recognition `models/ocr/PaddleOCR/tools/train.py`
- ROI and text lines classification `models/vlcls/train_vlcls.py`
