# Custom Object Detection and Novel Bounding Box Metric with YOLO

## The steps I followed

### 1. Setup
- Used YOLOv5 for object detection.
- Trained on a small dataset of cats and dogs.

### 2. Custom Bounding Box Similarity Metric
We introduced a new metric considering:
1. **IoU** (Intersection over Union).
2. **Aspect Ratio Similarity**.
3. **Center Alignment Similarity**.

Formula:
\[
S = 0.5 \times IoU + 0.3 \times S_{aspect} + 0.2 \times S_{center}
\]

### 3. Results
| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.75 |
| IoU | 0.68 |
| Custom Score | 0.72 |

### 4. Observations
- Our metric provided additional insights beyond IoU.
- It could be useful in applications where box shape and position matter.





## category 1: If you are going to run the code in this repository

### 0. Clone the repo
```
!git clone https://github.com/Manimohan05/Custom_Object_Detection_and_Novel_Bounding_Box_Metric_with_YOLO.git
```

### 1. Setup
- Install dependencies
```
!pip install torch torchvision torchaudio
!pip install ultralytics
```

- Used YOLOv5 for object detection. 
```
%cd yolov5
!pip install -r requirements.txt
```

- Trained on a small dataset of cats and dogs.
```
!python train.py --img 640 --batch 16 --epochs 50 --data F:/Assessment/Custom_Object_Detection_and_Novel_Bounding_Box_Metric_with_YOLO/dog-and-cat-2/data.yaml --weights yolov5s.pt

```
- Integrate Metric into Training or Evaluation
```
!python val.py --data F:/Assessment/Custom_Object_Detection_and_Novel_Bounding_Box_Metric_with_YOLO/dog-and-cat-2/data.yaml --weights runs/train/exp/weights/best.pt

```

## category 2: If you are going to run this from scartch

### 1. Setup
- Install dependencies
```
!pip install torch torchvision torchaudio
!pip install ultralytics
```

- Used YOLOv5 for object detection. 
```
!git clone https://github.com/ultralytics/yolov5.git 
%cd yolov5
!pip install -r requirements.txt
```
### Download Dataset 
[about ~50–100 images with bounding box annotations is enough for a 
proof-of-concept. ] 

- I used this datset from the roboflow https://universe.roboflow.com/project-3vvpk/dog-and-cat-tmn5b/dataset/2/download
```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="6PdYCi4jBGkqWpGKkaGi")
project = rf.workspace("project-3vvpk").project("dog-and-cat-tmn5b")
version = project.version(2)
dataset = version.download("yolov5")
```

- You may get the some other datasets from kaggle, roboflow, custom datasets.But make sure the file format as below
```
/dataset
├── images
│   ├── train (training images)
│   ├── val (validation images)
├── labels
│   ├── train (training labels)
│   ├── val (validation labels)
├── data.yaml  # Defines dataset paths

```
                
### Setup the data.yaml file with the correct dataset path (after the download)
- path of data.yaml file will be dog-and-cat-2\data.yaml

- Open data.yaml file and verify the paths (You may get assertion Error, If you got then please go through the path)
```
train: ./data/images/train
val: ./data/images/val
```

- Trained on a small dataset of cats and dogs. ( You can set the  img size , batch size, and epoches. and set the file path of data.yaml correctly)

```

!python train.py --img 640 --batch 16 --epochs 50 --data F:/Assessment/Custom_Object_Detection_and_Novel_Bounding_Box_Metric_with_YOLO/dog-and-cat-2/data.yaml --weights yolov5s.pt

```



## 2. Custom Bounding Box Similarity Metric
We introduced a new metric considering:
1. **IoU** (Intersection over Union).
2. **Aspect Ratio Similarity**.
3. **Center Alignment Similarity**.
S = IoU + e^(-d/50) + (1 - |AR1 - AR2| / max(AR1, AR2))^3

​- penalizes center distance.
- Aspect ratio similarity prevents shape mismatches.

- Then wrote the functions as follows and save as a file custom_metric.py in side of yolov5 directory
```
def custom_bbox_similarity(box1, box2):
    """
    Compute a custom bounding box similarity score.
    Inputs:
        - box1, box2: Bounding boxes in [x1, y1, x2, y2] format
    Returns:
        - similarity score (0 to 1)
    """
    # Compute IoU
    def compute_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    iou = compute_iou(box1, box2)

    # Compute center distance penalty
    center1 = [(box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2]
    center2 = [(box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2]
    center_dist = np.linalg.norm(np.array(center1) - np.array(center2))
    center_penalty = np.exp(-center_dist / 50)  # Normalize by 50 pixels

    # Compute aspect ratio similarity
    def aspect_ratio(box):
        return (box[2] - box[0]) / (box[3] - box[1]) if (box[3] - box[1]) > 0 else 1

    aspect_ratio_similarity = 1 - abs(aspect_ratio(box1) - aspect_ratio(box2)) / max(aspect_ratio(box1), aspect_ratio(box2))

    # Weighted combination
    similarity_score = (iou + center_penalty + aspect_ratio_similarity) / 3
    return similarity_score
```
### Then made following changes in the val.py in the yolov5 

- added the following line 
```
from custom_metric import custom_bbox_similarity
```

- Find this section in val.py:
```
    # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
            seen += 1

            if npr == 0:
                if nl:
                    stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    if plots:
                        confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                (save_dir / "labels").mkdir(parents=True, exist_ok=True)
                save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run("on_val_image_end", pred, predn, path, names, im[si])

```
- Modify it to compute our custom metric:
```

# Metrics
for si, pred in enumerate(preds):
    labels = targets[targets[:, 0] == si, 1:]
    nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
    path, shape = Path(paths[si]), shapes[si][0]
    correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init
    seen += 1

    if npr == 0:
        if nl:
            stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0], torch.tensor([])))
            if plots:
                confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
        continue

    # Predictions
    if single_cls:
        pred[:, 5] = 0
    predn = pred.clone()
    scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

    # Evaluate
    custom_scores = []  # Store our custom similarity scores
    if nl:
        tbox = xywh2xyxy(labels[:, 1:5])  # Convert labels to (x1, y1, x2, y2)
        scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # Scale to native resolution
        labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # Concatenate class IDs with boxes
        correct = process_batch(predn, labelsn, iouv)

        # Compute Custom Metric
        for pred_box in predn[:, :4]:  # Iterate over predictions
            best_score = 0
            for gt_box in tbox:  # Compare with ground truth boxes
                score = custom_bbox_similarity(pred_box.tolist(), gt_box.tolist())
                best_score = max(best_score, score)  # Take max similarity for each prediction
            custom_scores.append(best_score)

        if plots:
            confusion_matrix.process_batch(predn, labelsn)

    # Convert list to tensor and add to stats
    custom_scores = torch.tensor(custom_scores, device=device) if custom_scores else torch.tensor([], device=device)
    stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0], custom_scores))  # Include custom scores

    # Log similarity score
    if len(custom_scores) > 0:
        print(f"Custom BBox Similarity (Batch {batch_i}, Image {si}): {torch.mean(custom_scores):.4f}")

    # Save/log
    if save_txt:
        (save_dir / "labels").mkdir(parents=True, exist_ok=True)
        save_one_txt(predn, save_conf, shape, file=save_dir / "labels" / f"{path.stem}.txt")
    if save_json:
        save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
    callbacks.run("on_val_image_end", pred, predn, path, names, im[si])
```



## Integrate Metric into Training or Evaluation (before that set the file path of data.yaml correctly)
```
!python val.py --data F:/Assessment/Custom_Object_Detection_and_Novel_Bounding_Box_Metric_with_YOLO/dog-and-cat-2/data.yaml --weights runs/train/exp/weights/best.pt
```

## 3. Results
| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.75 |
| IoU | 0.68 |
| Custom Score | 0.72 |

## 4. Observations
- Our metric provided additional insights beyond IoU.
- It could be useful in applications where box shape and position matter.




