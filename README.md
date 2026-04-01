# Computer Vision Project

This project performs object detection on an input image using **YOLO (You Only Look Once)**.

It supports both:
- **YOLOv3 (COCO dataset, 80 classes)**
- **YOLOv2 (VOC dataset, 20 classes)**

---

## Requirements

Make sure you have Python installed, then install dependencies:

```bash
pip install opencv-python numpy
```

---

## Model Files

The YOLO model files are **NOT included in this repository** due to their large size.

You must download them from the **OneDrive link**.

### Required Files

#### For YOLOv3:
- `yolov3.cfg`
- `yolov3.weights`
- `coco.names`

#### For YOLOv2:
- `yolov2.cfg`
- `yolov2.weights`
- `voc.names`

After downloading, place all files in the same directory as `yolo_test.py`.

---

## Project Structure

```
project_name/
├── yolo_test.py
├── result.jpg
├── yolov3.cfg
├── yolov3.weights
├── coco.names
├── yolov2.cfg
├── yolov2.weights
├── voc.names
```

---

## Usage

### Run YOLOv3

```bash
py yolo_test.py --input result.jpg --cfg yolov3.cfg --weights yolov3.weights --names coco.names
```

---

### Run YOLOv2

```bash
py yolo_test.py --input result.jpg --cfg yolov2.cfg --weights yolov2.weights --names voc.names
```

---

## How It Works

1. Loads YOLO model configuration and weights  
2. Preprocesses the input image (resizing + normalization)  
3. Runs forward pass through the neural network  
4. Applies **Non-Maximum Suppression (NMS)** to remove duplicate detections  
5. Draws bounding boxes and labels on detected objects  

---

## Output

- Displays detections in a window  
- Saves annotated image as:

```
output.jpg
```

---
