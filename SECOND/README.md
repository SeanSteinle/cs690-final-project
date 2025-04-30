# SECOND

To train and test the SECOND model we are going to use the code from [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

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

The following command will train the second model. I have included the config file below. This includes the parameters used when training the model.

```bash
python -W ignore::DeprecationWarning train.py --cfg_file cfgs/kitti_models/second.yaml
```

```yaml
CLASS_NAMES: ["Car", "Pedestrian", "Cyclist"]

DATA_CONFIG:
    _BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml

MODEL:
    NAME: SECONDNet

    VFE:
        NAME: MeanVFE

    BACKBONE_3D:
        NAME: VoxelBackBone8x

    MAP_TO_BEV:
        NAME: HeightCompression
        NUM_BEV_FEATURES: 256

    BACKBONE_2D:
        NAME: BaseBEVBackbone

        LAYER_NUMS: [5, 5]
        LAYER_STRIDES: [1, 2]
        NUM_FILTERS: [128, 256]
        UPSAMPLE_STRIDES: [1, 2]
        NUM_UPSAMPLE_FILTERS: [256, 256]

    DENSE_HEAD:
        NAME: AnchorHeadSingle
        CLASS_AGNOSTIC: False

        USE_DIRECTION_CLASSIFIER: True
        DIR_OFFSET: 0.78539
        DIR_LIMIT_OFFSET: 0.0
        NUM_DIR_BINS: 2

        ANCHOR_GENERATOR_CONFIG:
            [
                {
                    "class_name": "Car",
                    "anchor_sizes": [[3.9, 1.6, 1.56]],
                    "anchor_rotations": [0, 1.57],
                    "anchor_bottom_heights": [-1.78],
                    "align_center": False,
                    "feature_map_stride": 8,
                    "matched_threshold": 0.6,
                    "unmatched_threshold": 0.45,
                },
                {
                    "class_name": "Pedestrian",
                    "anchor_sizes": [[0.8, 0.6, 1.73]],
                    "anchor_rotations": [0, 1.57],
                    "anchor_bottom_heights": [-0.6],
                    "align_center": False,
                    "feature_map_stride": 8,
                    "matched_threshold": 0.5,
                    "unmatched_threshold": 0.35,
                },
                {
                    "class_name": "Cyclist",
                    "anchor_sizes": [[1.76, 0.6, 1.73]],
                    "anchor_rotations": [0, 1.57],
                    "anchor_bottom_heights": [-0.6],
                    "align_center": False,
                    "feature_map_stride": 8,
                    "matched_threshold": 0.5,
                    "unmatched_threshold": 0.35,
                },
            ]

        TARGET_ASSIGNER_CONFIG:
            NAME: AxisAlignedTargetAssigner
            POS_FRACTION: -1.0
            SAMPLE_SIZE: 512
            NORM_BY_NUM_EXAMPLES: False
            MATCH_HEIGHT: False
            BOX_CODER: ResidualCoder

        LOSS_CONFIG:
            LOSS_WEIGHTS:
                {
                    "cls_weight": 1.0,
                    "loc_weight": 2.0,
                    "dir_weight": 0.2,
                    "code_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                }

    POST_PROCESSING:
        RECALL_THRESH_LIST: [0.3, 0.5, 0.7]
        SCORE_THRESH: 0.1
        OUTPUT_RAW_SCORE: False

        EVAL_METRIC: kitti

        NMS_CONFIG:
            MULTI_CLASSES_NMS: False
            NMS_TYPE: nms_gpu
            NMS_THRESH: 0.01
            NMS_PRE_MAXSIZE: 4096
            NMS_POST_MAXSIZE: 500

OPTIMIZATION:
    BATCH_SIZE_PER_GPU: 4
    NUM_EPOCHS: 80

    OPTIMIZER: adam_onecycle
    LR: 0.003
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
python demo.py --cfg_file cfgs/kitti_models/second.yaml --ckpt ../output/kitti_models/second/default/ckpt/checkpoint_epoch_80.pth --data_path ../data/kitti/testing/velodyne/000010.bin
python demo.py --cfg_file cfgs/kitti_models/second.yaml --ckpt ../output/kitti_models/second/default/ckpt/checkpoint_epoch_80.pth --data_path ../data/kitti/testing/velodyne/000061.bin
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

2025-04-29 20:50:15,169   INFO

Car AP@0.70, 0.70, 0.70:
bbox AP:90.8121, 89.9794, 89.1778
bev  AP:89.8830, 87.8279, 86.4654
3d   AP:88.5152, 78.6068, 77.3290
aos  AP:90.80, 89.90, 89.01

Car AP_R40@0.70, 0.70, 0.70:
bbox AP:95.6312, 94.1471, 91.7841
bev  AP:92.1598, 89.3997, 87.5052
3d   AP:89.9717, 81.6250, 78.6227
aos  AP:95.62, 94.04, 91.59

Car AP@0.70, 0.50, 0.50:
bbox AP:90.8121, 89.9794, 89.1778
bev  AP:90.8258, 90.1000, 89.5272
3d   AP:90.8258, 90.0684, 89.4097
aos  AP:90.80, 89.90, 89.01

Car AP_R40@0.70, 0.50, 0.50:
bbox AP:95.6312, 94.1471, 91.7841
bev  AP:95.6666, 94.7488, 94.1398
3d   AP:95.6555, 94.6360, 93.8485
aos  AP:95.62, 94.04, 91.59

Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:69.2113, 66.1247, 63.4274
bev  AP:63.1115, 56.7746, 53.8335
3d   AP:58.6777, 53.9046, 49.7502
aos  AP:65.40, 61.91, 58.93

Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:70.3869, 66.7333, 63.1553
bev  AP:62.7921, 56.4844, 52.2657
3d   AP:58.6046, 52.5645, 48.1265
aos  AP:65.99, 61.98, 58.09

Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:69.2113, 66.1247, 63.4274
bev  AP:75.9657, 73.2218, 69.7225
3d   AP:75.6941, 72.9519, 69.3746
aos  AP:65.40, 61.91, 58.93

Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:70.3869, 66.7333, 63.1553
bev  AP:77.0275, 74.0073, 70.4700
3d   AP:76.7021, 73.6685, 70.1581
aos  AP:65.99, 61.98, 58.09

Cyclist AP@0.50, 0.50, 0.50:
bbox AP:86.6637, 76.5282, 72.6720
bev  AP:83.9493, 69.9241, 66.3441
3d   AP:80.7166, 66.5600, 62.2242
aos  AP:86.33, 75.98, 72.12

Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:89.5914, 78.2676, 74.0232
bev  AP:86.5923, 70.9884, 66.6470
3d   AP:82.0026, 66.9198, 62.8797
aos  AP:89.22, 77.66, 73.43

Cyclist AP@0.50, 0.25, 0.25:
bbox AP:86.6637, 76.5282, 72.6720
bev  AP:85.6653, 74.0072, 70.0826
3d   AP:85.6653, 74.0072, 70.0826
aos  AP:86.33, 75.98, 72.12

Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:89.5914, 78.2676, 74.0232
bev  AP:88.5071, 75.3610, 71.2955
3d   AP:88.5071, 75.3610, 71.2955
aos  AP:89.22, 77.66, 73.43
```
