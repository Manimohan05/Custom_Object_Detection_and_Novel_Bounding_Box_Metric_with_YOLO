# Custom Object Detection and Novel Bounding Box Metric with YOLO

## The steps I followed

### 1. Setup
- Used YOLOv5 for object detection.
- Trained on a small dataset of cats and dogs.

### 2. Custom Bounding Box Similarity Metric
I introduced a new metric considering the following factors:

- **IoU (Intersection over Union)**
- **Aspect Ratio Similarity**
- **Center Alignment Similarity**
- **Size Similarity**

The custom similarity score is computed as:

$$
\text{Similarity} = 0.5 \times \text{IoU} + 0.2 \times \text{ARS} + 0.2 \times \text{CA} + 0.1 \times \text{SS}
$$

Where:

- **IoU** (Intersection over Union) is calculated as:

$$
\text{IoU} = \frac{\text{Intersection Area}}{\text{Union Area}}
$$

- **ARS** (Aspect Ratio Similarity) is calculated as:

$$
\text{ARS} = 1 - \frac{|AR_1 - AR_2|}{\max(AR_1, AR_2)}
$$

- **CA** (Center Alignment) is calculated as:

$$
\text{CA} = 1 - \frac{\| \text{Center1} - \text{Center2} \|}{\text{Image Size}}
$$

- **SS** (Size Similarity) is calculated as:

$$
\text{SS} = 1 - \frac{|A_1 - A_2|}{\max(A_1, A_2)}
$$

Where \(AR_1, AR_2\) are the aspect ratios of box1 and box2, and \(A_1, A_2\) are the areas of box1(ground truth) and box2(prediction), respectively.

This metric combines these factors to provide a comprehensive similarity score between predicted and ground truth bounding boxes.



## Category 1: If you are going to run the code in this repository

### 0. Clone the repo
```
!git clone https://github.com/Manimohan05/Custom_Object_Detection_and_Novel_Bounding_Box_Metric_with_YOLO.git
```

- Open the notebook named as Assessment.ipynb.
- Then, do the folllowing steps

### 1. Setup
- Install dependencies
```
!pip install torch torchvision torchaudio
!pip install ultralytics
```

- Use YOLOv5 for object detection. 
```
%cd yolov5
!pip install -r requirements.txt
```
### 2. Training 
- Set the path of data.yaml file in the following command
- You can set the  img size , batch size, and epoches.
- Train on a small dataset of cats and dogs.
```
!python train.py --img 640 --batch 16 --epochs 50 --data F:/Assessment/Custom_Object_Detection_and_Novel_Bounding_Box_Metric_with_YOLO/dog-and-cat-2/data.yaml --weights yolov5s.pt

```

### 3. Evaluating 
- Set the path of data.yaml file in the following command
- Integrate Metric into Training or Evaluation
```
!python val.py --data F:/Assessment/Custom_Object_Detection_and_Novel_Bounding_Box_Metric_with_YOLO/dog-and-cat-2/data.yaml --weights runs/train/exp/weights/best.pt

```

## Category 2: If you are going to run this from scartch

### 1. Setup
- Install dependencies
```
!pip install torch torchvision torchaudio
!pip install ultralytics
```

- Use YOLOv5 for object detection. 
```
!git clone https://github.com/ultralytics/yolov5.git 
%cd yolov5
!pip install -r requirements.txt
```
### 2. Download Dataset 
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
                
### 3.Setup the data.yaml file with the correct dataset path (after the download)
- path of data.yaml file will be dog-and-cat-2\data.yaml

- Open data.yaml file and verify the paths (You may get assertion Error, If you got then please go through the path)
```
train: ./data/images/train
val: ./data/images/val
```

### 4. Training
- Train on a small dataset of cats and dogs. ( You can set the  img size , batch size, and epoches. and set the file path of data.yaml correctly)

```

!python train.py --img 640 --batch 16 --epochs 50 --data F:/Assessment/Custom_Object_Detection_and_Novel_Bounding_Box_Metric_with_YOLO/dog-and-cat-2/data.yaml --weights yolov5s.pt

```



### 5. Custom Bounding Box Similarity Metric
We introduced a new metric considering:
1. **IoU** (Intersection over Union).
2. **Aspect Ratio Similarity**.
3. **Center Alignment Similarity**.
S = IoU + e^(-d/50) + (1 - |AR1 - AR2| / max(AR1, AR2))^3

​- penalizes center distance.
- Aspect ratio similarity prevents shape mismatches.

- Then wrote the functions as follows and add that inside of file metrics.py in side of yolov5 directory
```
def custom_bbox_similarity(box1, box2, img_size=640):
    """
    Computes a custom bounding box similarity metric based on IoU, aspect ratio similarity,
    center alignment, and size similarity.

    Args:
        box1: Tensor of shape (N, 4) - Ground truth bounding boxes [x1, y1, x2, y2]
        box2: Tensor of shape (M, 4) - Predicted bounding boxes [x1, y1, x2, y2]
        img_size: (int) Image size (width = height) used for normalization

    Returns:
        Custom similarity score (0 to 1)
    """
    # Compute IoU
    xA = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0])  # Expand box1's x1 for element-wise comparison
    yA = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1])  # Expand box1's y1 for element-wise comparison
    xB = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2])  # Expand box1's x2 for element-wise comparison
    yB = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3])  # Expand box1's y2 for element-wise comparison

    inter_area = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    iou = inter_area / (box1_area.unsqueeze(1) + box2_area - inter_area + 1e-6)

    # Compute Aspect Ratio Similarity (ARS)
    ar1 = (box1[:, 2] - box1[:, 0]) / (box1[:, 3] - box1[:, 1] + 1e-6)
    ar2 = (box2[:, 2] - box2[:, 0]) / (box2[:, 3] - box2[:, 1] + 1e-6)
    ars = 1 - torch.abs(ar1.unsqueeze(1) - ar2) / torch.max(ar1.unsqueeze(1), ar2)

    # Compute Center Alignment (CA)
    # Calculate center for box1 and box2 (midpoints of (x1, y1) and (x2, y2))
    center1_x = (box1[:, 0] + box1[:, 2]) / 2
    center1_y = (box1[:, 1] + box1[:, 3]) / 2
    center2_x = (box2[:, 0] + box2[:, 2]) / 2
    center2_y = (box2[:, 1] + box2[:, 3]) / 2

    # Stack to create tensors of shape (N, 2) and (M, 2)
    center1_tensor = torch.stack((center1_x, center1_y), dim=-1)  # Shape (N, 2)
    center2_tensor = torch.stack((center2_x, center2_y), dim=-1)  # Shape (M, 2)

    # Ensure the tensors have the same shape before computing the norm
    ca = 1 - torch.norm(center1_tensor.unsqueeze(1) - center2_tensor, dim=-1) / img_size

    # Compute Size Similarity (SS)
    area1 = box1_area
    area2 = box2_area
    ss = 1 - torch.abs(area1.unsqueeze(1) - area2) / torch.max(area1.unsqueeze(1), area2)

    # Weighted Combination
    similarity = 0.5 * iou + 0.2 * ars + 0.2 * ca + 0.1 * ss
    return similarity
```
### 6. Make the following changes in the metrics.py in the yolov5 


- Find this section in metrics.py:
```
iou = box_iou(labels[:, 1:], detections[:, :4])

```
- Modify it to compute our custom metric:
```
iou = custom_bbox_similarity(labels[:, 1:], detections[:, :4])

```



### 7. Integrate Metric into Evaluation 
- Set the path of data.yaml file in the following command
```
!python val.py --data F:/Assessment/Custom_Object_Detection_and_Novel_Bounding_Box_Metric_with_YOLO/dog-and-cat-2/data.yaml --weights runs/train/exp/weights/best.pt
```


## Training Results
| Metric | Value |
|--------|-------|
| mAP@0.5|0.82698|
|--------|-------|
|   IOU  | 0.701 |


## Evaluation Results
| Metric | Value |
|--------|-------|
| mAP@0.5|0.759  |
|--------|-------|
|   IOU  | 0.681 |

## Observations
- Our metric provided additional insights beyond IoU.
- It could be useful in applications where box shape and position matter.




