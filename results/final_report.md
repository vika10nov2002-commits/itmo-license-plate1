
# License plate detection and recognition homework

## 1. Detection data

For license plate detection, 73 images were annotated manually in YOLO format.

Class:

`0: license_plate`

Split:

- train: 51
- val: 10
- test: 12

## 2. Detection model

Model: YOLOv8n

Checkpoint:

`runs/detect/runs/detect/license_plate_detector_clean/weights/best.pt`

Metrics on test:

- Precision: 0.9642
- Recall: 0.7500
- mAP@0.5: 0.8779
- mAP@0.5:0.95: 0.2846

## 3. Recognition model

Model: CRNN + CTC Loss

Best CRNN experiment:

`crnn_exp1_width160`

Best checkpoint:

`checkpoints/recognition/crnn_exp1_width160_best.pt`

CRNN experiments are saved to:

`results/crnn_experiments.csv`

## 4. End-to-end pipeline

Pipeline:

1. Input car image.
2. YOLOv8 detector finds the license plate.
3. The license plate crop is passed to CRNN.
4. CRNN predicts the text string.

Output demo:

`results/e2e_prediction.jpg`
