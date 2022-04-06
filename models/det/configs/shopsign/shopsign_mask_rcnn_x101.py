# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py'

data_root = 'H:/sign_seg_dataset/sign_coco/'
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=3),
        mask_head=dict(num_classes=3)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('shopsign', 'name', 'streetsign',)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        img_prefix=data_root + 'train/',
        classes=classes,
        ann_file=data_root + 'train.json'),
    val=dict(
        img_prefix=data_root + 'val/',
        classes=classes,
        ann_file=data_root + 'val.json'),
    test=dict(
        img_prefix=data_root + 'val/',
        classes=classes,
        ann_file=data_root + 'val.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '../checkpoints/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth'
