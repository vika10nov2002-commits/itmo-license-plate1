# ITMO License Plate OCR — cleaned Colab version

Проект для распознавания автомобильных номеров:

1. **Detector**: YOLOv8/YOLO11 находит прямоугольник номера на машине.
2. **Recognizer**: CRNN + CTC распознаёт текст на crop номера.
3. **Pipeline**: YOLO crop -> CRNN text -> картинка с рамкой и предсказанием.

Код сделан под Google Colab и GitHub: нет абсолютных Windows-путей, датасеты и веса не должны попадать в репозиторий.

## Что взято из референсов и что улучшено

Из примеров сохранена базовая логика CRNN: `CNN -> BiLSTM -> Linear -> CTCLoss`, где `blank=0`, а символы начинаются с индекса 1. Улучшения:

- вместо одного большого ноутбука код разбит на модули;
- есть 2 архитектуры CRNN: `small` для стабильного старта и `deep` для более тяжёлого эксперимента;
- добавлены CER и exact-match accuracy, а не только train loss;
- добавлен `letterbox` resize вместо грубого растягивания номера;
- нет `RandomHorizontalFlip`, потому что зеркальный номер ломает OCR;
- добавлен `zero_infinity=True`, gradient clipping и пропуск non-finite loss;
- checkpoint хранит не только веса, но и конфиг + алфавит;
- есть отдельный end-to-end inference через YOLO + CRNN;
- добавлены скрипты подготовки данных: ручная разметка, pseudo-labeling для детекции, CSV-индексы для OCR.

## Установка в Colab

```bash
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
!pip install -r requirements.txt
```

Подключить Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 1. OCR / CRNN обучение

Вариант A: использовать AutoRia/Nomeroff структуру напрямую:

```text
OCR_ROOT/
  train/ann/*.json
  train/img/*.png
  test/ann/*.json
  test/img/*.png
```

Отредактируй `configs/recognition.yaml`:

```yaml
data_root: /content/drive/MyDrive/Datasets/OCR/autoriaNumberplateOcrRu
dataset_format: autoria_json
train_split: train
val_split: test
alphabet: ru_plate
architecture: small
```

Запуск:

```bash
python -m src.anpr.train_recognizer --config configs/recognition.yaml --device auto
```

Вариант B: сначала сделать CSV-индексы OCR, как в `data_prep/prep_ocr.py` из другого референса:

```bash
python tools/prepare_ocr_autoria_csv.py \
  --src /content/drive/MyDrive/Datasets/OCR/autoriaNumberplateOcrRu \
  --out /content/drive/MyDrive/Datasets/OCR/ocr_indexes
```

После этого можно поставить в конфиге:

```yaml
data_root: /content/drive/MyDrive/Datasets/OCR/ocr_indexes
dataset_format: csv
train_split: ocr_train
val_split: ocr_val
alphabet_path: /content/drive/MyDrive/Datasets/OCR/ocr_indexes/alphabet.txt
```

Для быстрой проверки можно временно включить:

```yaml
max_train_samples: 200
max_test_samples: 100
```

Но для нормального результата эти ограничения нужно убрать.

## 2. Детекция / YOLO

### Вариант A — правильный: ручная разметка

Если у тебя есть только изображения машин, сначала нужна разметка bbox номеров.

Подготовить картинки для разметки:

```bash
python tools/prepare_detection_images.py \
  --source /content/drive/MyDrive/number_car_detect.zip \
  --out data/detection_raw
```

Потом разметь номера в LabelImg/CVAT/Roboflow и экспортируй в YOLO format:

```text
data/detection_yolo/
  images/train/*.jpg
  labels/train/*.txt
  images/val/*.jpg
  labels/val/*.txt
  images/test/*.jpg
  labels/test/*.txt
  data.yaml
```

Проверить labels:

```bash
python tools/check_yolo_labels.py --root /content/drive/MyDrive/Datasets/license_plate_detection
```

### Вариант B — быстрый старт: pseudo-labeling

Это похоже на `data_prep/prep_detection.py` из референса Daniil-Nay: предобученный детектор номеров сам генерирует bbox-кандидаты, а потом их нужно открыть и поправить руками.

Важно: нужен **предобученный детектор номеров**, а не обычный COCO `yolov8n.pt`, потому что COCO-модель не знает класса `license_plate`.

```bash
python tools/pseudo_label_detection.py \
  --source /content/drive/MyDrive/number_car_detect.zip \
  --out /content/drive/MyDrive/Datasets/license_plate_detection \
  --weights /content/drive/MyDrive/weights/license_plate_detector.pt \
  --conf 0.30 \
  --preview 30
```

Потом обязательно посмотри папку:

```text
/content/drive/MyDrive/Datasets/license_plate_detection/preview
```

Если рамки кривые или номера пропущены, их надо исправить вручную. Для отчёта лучше честно написать: «первичная разметка была получена pseudo-labeling и затем проверена/скорректирована вручную».

### Обучение YOLO

В `configs/detection.yaml` поставь путь к датасету:

```yaml
path: /content/drive/MyDrive/Datasets/license_plate_detection
train: images/train
val: images/val
test: images/test
names:
  0: license_plate
```

Запуск:

```bash
python -m src.anpr.train_detector \
  --data configs/detection.yaml \
  --model yolov8n.pt \
  --epochs 50 \
  --imgsz 640 \
  --device 0 \
  --eval
```

## 3. End-to-end prediction

```bash
python -m src.anpr.predict_pipeline \
  --image /content/drive/MyDrive/test_car.jpg \
  --detector runs/detect/license_plate_detector/weights/best.pt \
  --recognizer checkpoints/recognition/crnn_small_ru_plate_best.pt \
  --output results/prediction.jpg
```

## Что пушить в GitHub

Пушить:

```text
src/
configs/
tools/
README.md
requirements.txt
.gitignore
```

Не пушить:

```text
data/
Datasets/
runs/
checkpoints/
*.pt
*.pth
*.ckpt
*.zip
```

## Мини-план для отчёта

1. Описать задачу: ANPR = detection + recognition.
2. Детекция: YOLO, метрики mAP50/mAP50-95, Precision, Recall.
3. OCR: CRNN + CTC, метрики CER и Accuracy.
4. Показать 5-10 end-to-end примеров.
5. Описать ошибки: плохое освещение, смаз, маленький номер, похожие символы.
