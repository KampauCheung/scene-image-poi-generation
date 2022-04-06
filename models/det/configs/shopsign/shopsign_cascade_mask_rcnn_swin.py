# The new config inherits a base config to highlight the necessary modification
_base_ = '../swin/cascade_mask_rcnn_swin_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py'

data_root = 'Z:/Project/Street_view/project_data/sign_coco/'
# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        # bbox_head=dict(num_classes=2),
        mask_head=dict(num_classes=2)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('shopsign', 'name', 'streetsign',)
data = dict(
    samples_per_gpu=2,
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

runner = dict(type='EpochBasedRunner', max_epochs=10)
# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '../../../weights/det/cascade_mask_rcnn_swin.pth'
