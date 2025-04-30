# SECOND

To train and test the SECOND model we are going to leverage [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Data Setup

Setting up the data can be found [here](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#dataset-preparation). Notably the following download links are used:

* [calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
* [velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)
* [label_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)
* [image_2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)

## Package Installation

Running on WSL Ubuntu 22.04. Cuda is on 11.8.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install spconv-cu118
python -m pip install kornia==0.5.8
python -m pip install av2
```

## Train

```bash
python -W ignore::DeprecationWarning train.py --cfg_file cfgs/kitti_models/second.yaml
```

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
