# Colab command cells

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!pip install -r requirements.txt
```

```bash
!python -m src.anpr.train_recognizer --config configs/recognition.yaml --device auto
```

```bash
!python -m src.anpr.train_detector --data configs/detection.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --device 0 --eval
```
