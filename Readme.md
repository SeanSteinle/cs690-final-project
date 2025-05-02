# 1. Performance and Execution (Implementation details) ❌ ✅
a) PointPillar repo - https://github.com/zhulf0804/PointPillars

b) SECOND repo - https://github.com/open-mmlab/OpenPCDet

c) PointRCNN repo - https://github.com/sshaoshuai/PointRCNN

# 2. Report
## Abstract ❌
Over the years, numerous frameworks have emerged that make 3D object detection in the autonomous scenario easier and more robust. Despite the introduction of such frameworks, there are always areas that require improvement. In this project, we have explored a variety of end-to-end 3D object–detection pipelines on the KITTI benchmark to understand how different representations, network backbones, and native CUDA-accelerated modules impact 3D object detection accuracy and speed. The KITTI dataset consists of image and pointcloud data, which can be transformed into structured inputs in the form of voxels, pillars, or bird’s-eye-view (BEV) images, that can be processed by 2D or 3D convolutional backbones. We experimented with different frameworks and finalized three of these to be included in the final report. These are PointPillar, SECOND and PointRCNN. 

## Introduction ❌

## Related work ❌

## Technical Approach ❌
In 

## Results and Discussion ❌
### Peformance metrics
| Metric             | Definition                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------- |
| **AP\@X,Y,Z**      | Average Precision at IoU thresholds X (2D bbox), Y (BEV), Z (3D). Higher = better.          |
| **AP\_R40\@X,Y,Z** | The same, but computed with the “40‑point” recall setting (more stable for sparse objects). |
| **AOS\@X**         | Average Orientation Similarity at IoU = X: combines detection correctness + yaw accuracy.   |
| **“bbox” AP**      | 2D bounding‑box detection in image plane.                                                   |
| **“bev” AP**       | Bird’s‑Eye‑View (top‑down) 2D detection.                                                    |
| **“3d” AP**        | Full 3D bounding‑box detection (x,y,z + size + orientation).                                |
| **Difficulty**     | KITTI Easy / Moderate / Hard splits (increasing occlusion, truncation, distance).           |



Thresholds by class:

Cars use AP@0.70,0.70,0.70 (strict localization).
Pedestrians/Cyclists use AP@0.50,0.50,0.50 (and also looser AP@0.50,0.25,0.25).
We also report AP_R40 (40 recall positions) which reduces noise for small or sparse objects.

### Result analysis
We compare three models:

1. SECOND
2. PointRCNN
3. PointPillars

## Conclusions ❌

## References ❌


# 3. Presentation slides ❌
