## Custom Object Detection and Novel Bounding Box Metric with YOLO


## 1. Setup
- Used YOLOv5 for object detection.
- Trained on a small dataset of cats and dogs.

## 2. Custom Bounding Box Similarity Metric
We introduced a new metric considering:
1. **IoU** (Intersection over Union).
2. **Aspect Ratio Similarity**.
3. **Center Alignment Similarity**.

Formula:
\[
S = 0.5 \times IoU + 0.3 \times S_{aspect} + 0.2 \times S_{center}
\]

## 3. Results
| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.75 |
| IoU | 0.68 |
| Custom Score | 0.72 |

## 4. Observations
- Our metric provided additional insights beyond IoU.
- It could be useful in applications where box shape and position matter.




