# PointRCNN

To train and test the PointRCNN model we are going to use the code from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Data Setup

We are using the KITTI dataset.

Instructions to setup up the data for training and testing can be found [here](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#dataset-preparation). Notably the following download links are used:

* [calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
* [velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)
* [label_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)
* [image_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)

The files are setup in the directory as shown:

```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

## Package Installation

Running on WSL Ubuntu 22.04. Cuda is on 11.8.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install spconv-cu118
python -m pip install kornia==0.5.8
python -m pip install av2
python -m pip install open3d
```

## Train

The following command will train the second model. I have included the config file below. This includes the parameters used when training the model. The training took 7 hours and 46 minutes on a 3080Ti GPU and i9-11900K @ 3.5GHz. This model uses PointRCNN with an IoU-based loss function.

```bash
python -W ignore::DeprecationWarning train.py --cfg_file cfgs/kitti_models/pointrcnn_iou.yaml
```

```yaml
CLASS_NAMES: ['Car', 'Pedestrian', 'Cyclist']

DATA_CONFIG:
    _BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml

    DATA_PROCESSOR:
        -   NAME: mask_points_and_boxes_outside_range
            REMOVE_OUTSIDE_BOXES: True

        -   NAME: sample_points
            NUM_POINTS: {
                'train': 16384,
                'test': 16384
            }

        -   NAME: shuffle_points
            SHUFFLE_ENABLED: {
                'train': True,
                'test': False
            }

MODEL:
    NAME: PointRCNN

    BACKBONE_3D:
        NAME: PointNet2MSG
        SA_CONFIG:
            NPOINTS: [4096, 1024, 256, 64]
            RADIUS: [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
            NSAMPLE: [[16, 32], [16, 32], [16, 32], [16, 32]]
            MLPS: [[[16, 16, 32], [32, 32, 64]],
                   [[64, 64, 128], [64, 96, 128]],
                   [[128, 196, 256], [128, 196, 256]],
                   [[256, 256, 512], [256, 384, 512]]]
        FP_MLPS: [[128, 128], [256, 256], [512, 512], [512, 512]]

    POINT_HEAD:
        NAME: PointHeadBox
        CLS_FC: [256, 256]
        REG_FC: [256, 256]
        CLASS_AGNOSTIC: False
        USE_POINT_FEATURES_BEFORE_FUSION: False
        TARGET_CONFIG:
            GT_EXTRA_WIDTH: [0.2, 0.2, 0.2]
            BOX_CODER: PointResidualCoder
            BOX_CODER_CONFIG: {
                'use_mean_size': True,
                'mean_size': [
                    [3.9, 1.6, 1.56],
                    [0.8, 0.6, 1.73],
                    [1.76, 0.6, 1.73]
                ]
            }

        LOSS_CONFIG:
            LOSS_REG: WeightedSmoothL1Loss
            LOSS_WEIGHTS: {
                'point_cls_weight': 1.0,
                'point_box_weight': 1.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }

    ROI_HEAD:
        NAME: PointRCNNHead
        CLASS_AGNOSTIC: True

        ROI_POINT_POOL:
            POOL_EXTRA_WIDTH: [0.0, 0.0, 0.0]
            NUM_SAMPLED_POINTS: 512
            DEPTH_NORMALIZER: 70.0

        XYZ_UP_LAYER: [128, 128]
        CLS_FC: [256, 256]
        REG_FC: [256, 256]
        DP_RATIO: 0.0
        USE_BN: False

        SA_CONFIG:
            NPOINTS: [128, 32, -1]
            RADIUS: [0.2, 0.4, 100]
            NSAMPLE: [16, 16, 16]
            MLPS: [[128, 128, 128],
                   [128, 128, 256],
                   [256, 256, 512]]

        NMS_CONFIG:
            TRAIN:
                NMS_TYPE: nms_gpu
                MULTI_CLASSES_NMS: False
                NMS_PRE_MAXSIZE: 9000
                NMS_POST_MAXSIZE: 512
                NMS_THRESH: 0.8
            TEST:
                NMS_TYPE: nms_gpu
                MULTI_CLASSES_NMS: False
                NMS_PRE_MAXSIZE: 9000
                NMS_POST_MAXSIZE: 100
                NMS_THRESH: 0.85

        TARGET_CONFIG:
            BOX_CODER: ResidualCoder
            ROI_PER_IMAGE: 128
            FG_RATIO: 0.5

            SAMPLE_ROI_BY_EACH_CLASS: True
            CLS_SCORE_TYPE: roi_iou

            CLS_FG_THRESH: 0.7
            CLS_BG_THRESH: 0.25
            CLS_BG_THRESH_LO: 0.1
            HARD_BG_RATIO: 0.8

            REG_FG_THRESH: 0.55

        LOSS_CONFIG:
            CLS_LOSS: BinaryCrossEntropy
            REG_LOSS: smooth-l1
            CORNER_LOSS_REGULARIZATION: True
            LOSS_WEIGHTS: {
                'rcnn_cls_weight': 1.0,
                'rcnn_reg_weight': 1.0,
                'rcnn_corner_weight': 1.0,
                'code_weights': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            }

    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        SCORE_THRESH: 0.1
        OUTPUT_RAW_SCORE: False

        EVAL_METRIC: kitti

        NMS_CONFIG:
            MULTI_CLASSES_NMS: False
            NMS_TYPE: nms_gpu
            NMS_THRESH: 0.1
            NMS_PRE_MAXSIZE: 4096
            NMS_POST_MAXSIZE: 500


OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 3
    NUM_EPOCHS: 80

    OPTIMIZER: adam_onecycle
    LR: 0.01
    WEIGHT_DECAY: 0.01
    MOMENTUM: 0.9

    MOMS: [0.95, 0.85]
    PCT_START: 0.4
    DIV_FACTOR: 10
    DECAY_STEP_LIST: [35, 45]
    LR_DECAY: 0.1
    LR_CLIP: 0.0000001

    LR_WARMUP: False
    WARMUP_EPOCH: 1

    GRAD_NORM_CLIP: 10

```

## Test

## Qualitative Assesment

The `demo.py` file was modified to include an image with bounding boxes as well. The added code is below.

```bash
python demo.py --cfg_file cfgs/kitti_models/pointrcnn_iou.yaml --ckpt ../output/kitti_models/pointrcnn_iou/default/ckpt/checkpoint_epoch_80.pth --data_path ../data/kitti/testing/velodyne/000010.bin
python demo.py --cfg_file cfgs/kitti_models/pointrcnn_iou.yaml --ckpt ../output/kitti_models/pointrcnn_iou/default/ckpt/checkpoint_epoch_80.pth --data_path ../data/kitti/testing/velodyne/000061.bin
```

```python
def project_to_image(points_3d, P):
    """
    Projects 3D points to the image plane using the projection matrix P.
    """
    points_3d_homo = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  # Convert to homogeneous coordinates
    points_2d_homo = np.dot(P, points_3d_homo.T).T  # Apply projection
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:]  # Normalize by depth
    return points_2d

def draw_bounding_boxes(image, corners_2d, color=(0, 255, 0), thickness=2):
    """
    Draws 2D bounding boxes on the image.
    """
    corners_2d = corners_2d.astype(int)
    for i in range(4):
        # Draw lines between the front face corners
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[(i + 1) % 4]), color, thickness)
        # Draw lines between the back face corners
        cv2.line(image, tuple(corners_2d[i + 4]), tuple(corners_2d[(i + 1) % 4 + 4]), color, thickness)
        # Draw lines connecting front and back faces
        cv2.line(image, tuple(corners_2d[i]), tuple(corners_2d[i + 4]), color, thickness)

def load_calibration(calib_file):
    """
    Load calibration data from a KITTI calibration file.
    """
    with open(calib_file, 'r') as f:
        lines = f.readlines()

    # Extract P2, R0_rect, and Tr_velo_to_cam
    P2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
    R0_rect = np.array([float(x) for x in lines[4].split()[1:]]).reshape(3, 3)
    Tr_velo_to_cam = np.array([float(x) for x in lines[5].split()[1:]]).reshape(3, 4)

    # Add a row to make Tr_velo_to_cam a 4x4 matrix
    Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))

    return P2, R0_rect, Tr_velo_to_cam

def compute_3d_corners(box):
    """
    Compute the 8 corners of a 3D bounding box.
    """
    x, y, z, dx, dy, dz, heading = box
    # Rotation matrix around the Z-axis
    R = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading), np.cos(heading), 0],
        [0, 0, 1]
    ])

    # 8 corners in the box's local coordinate system
    corners = np.array([
        [dx / 2, dy / 2, dz / 2],
        [dx / 2, dy / 2, -dz / 2],
        [dx / 2, -dy / 2, dz / 2],
        [dx / 2, -dy / 2, -dz / 2],
        [-dx / 2, dy / 2, dz / 2],
        [-dx / 2, dy / 2, -dz / 2],
        [-dx / 2, -dy / 2, dz / 2],
        [-dx / 2, -dy / 2, -dz / 2]
    ])

    # Rotate and translate the corners
    corners = np.dot(corners, R.T) + np.array([x, y, z])
    return corners

# Load corresponding RGB image
image_path = f"../data/kitti/testing/image_2/000010.png"
image = cv2.imread(image_path)

if image is None:
    logger.warning(f"Image not found: {image_path}")
    continue

# Project 3D bounding boxes onto the image
calib_file = f"../data/kitti/testing/calib/000010.txt"
P2, R0_rect, Tr_velo_to_cam = load_calibration(calib_file)

pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
for box in pred_boxes:
    corners_3d_velo = compute_3d_corners(box)
    corners_3d_cam = np.dot(R0_rect, np.dot(Tr_velo_to_cam[:3, :3], corners_3d_velo.T) + Tr_velo_to_cam[:3, 3:4]).T
    corners_2d = project_to_image(corners_3d_cam, P2)
    draw_bounding_boxes(image, corners_2d)

# Display the image with bounding boxes
cv2.imwrite(f"./000010.png", image)
```

## Quantitative Assesment

The evaluation on the test set is run after training. Here are the raw results:

```
2025-05-01 05:35:27,222

Car AP@0.70, 0.70, 0.70:
bbox AP:90.5257, 89.2349, 88.9356
bev  AP:89.8123, 87.1214, 85.8392
3d   AP:88.3870, 78.1954, 77.7231
aos  AP:90.52, 89.13, 88.78

Car AP_R40@0.70, 0.70, 0.70:
bbox AP:95.9820, 92.2827, 90.0942
bev  AP:92.7414, 88.5143, 86.5560
3d   AP:90.5946, 80.0310, 77.8711
aos  AP:95.97, 92.17, 89.94

Car AP@0.70, 0.50, 0.50:
bbox AP:90.5257, 89.2349, 88.9356
bev  AP:90.4606, 89.4314, 89.2555
3d   AP:90.4573, 89.4017, 89.1987
aos  AP:90.52, 89.13, 88.78

Car AP_R40@0.70, 0.50, 0.50:
bbox AP:95.9820, 92.2827, 90.0942
bev  AP:95.9540, 94.5215, 92.4687
3d   AP:95.9384, 94.3293, 92.3538
aos  AP:95.97, 92.17, 89.94

Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:73.5444, 65.8273, 62.2745
bev  AP:67.5066, 60.2687, 54.0909
3d   AP:61.8402, 57.0153, 51.1470
aos  AP:71.29, 63.17, 59.48

Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:74.6165, 67.4874, 62.0952
bev  AP:66.6666, 59.3514, 52.8303
3d   AP:63.2602, 55.9949, 49.4335
aos  AP:72.02, 64.34, 58.92

Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:73.5444, 65.8273, 62.2745
bev  AP:80.5148, 73.8899, 66.3975
3d   AP:80.4604, 73.8003, 66.2768
aos  AP:71.29, 63.17, 59.48

Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:74.6165, 67.4874, 62.0952
bev  AP:81.2173, 74.9651, 68.0123
3d   AP:81.1624, 74.8505, 67.8606
aos  AP:72.02, 64.34, 58.92

Cyclist AP@0.50, 0.50, 0.50:
bbox AP:89.7502, 77.6679, 75.2861
bev  AP:88.3649, 74.4365, 71.0086
3d   AP:87.7197, 72.5666, 69.9434
aos  AP:89.67, 77.20, 74.70

Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:94.5122, 80.3424, 76.1598
bev  AP:92.7105, 75.6850, 71.2260
3d   AP:90.9061, 72.9923, 69.5031
aos  AP:94.41, 79.76, 75.58

Cyclist AP@0.50, 0.25, 0.25:
bbox AP:89.7502, 77.6679, 75.2861
bev  AP:89.1978, 75.8530, 72.6934
3d   AP:89.1978, 75.8530, 72.6934
aos  AP:89.67, 77.20, 74.70

Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:94.5122, 80.3424, 76.1598
bev  AP:93.7917, 78.0207, 73.5824
3d   AP:93.7917, 78.0207, 73.5824
aos  AP:94.41, 79.76, 75.58
```
