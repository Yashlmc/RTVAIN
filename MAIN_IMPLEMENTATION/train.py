from ultralytics import YOLO

# Load pretrained YOLOv11 model (choose small or nano)
model = YOLO("yolo11n.pt")  # or "yolov11n.pt" if you have very limited GPU

# Train
results = model.train(
    data="data.yaml",           # dataset config file
    epochs=10,                 # higher than usual (helps small datasets)
    imgsz=640,                  # input image size
    batch=8,                    # small batch for small data (adjust to GPU memory)
    patience=20,                # early stop if no improvement
    lr0=0.001,                  # initial learning rate
    lrf=0.01,                   # final learning rate factor (cosine decay)
    optimizer="AdamW",          # better for small datasets
    warmup_epochs=3,            # helps stabilize early learning
    mosaic=0.3,                 # partial mosaic augmentation (reduce from 1.0)
    mixup=0.1,                  # light mixup (synthetic blending)
    hsv_h=0.015,                # light color augmentations
    hsv_s=0.6,
    hsv_v=0.4,
    degrees=5,                  # small rotation
    translate=0.1,              # minor translation
    scale=0.2,                  # light scale jitter
    shear=2,                    # small shearing
    perspective=0.0005,         # subtle perspective distortion
    flipud=0.0,                 # vertical flip disabled (useful only for aerial images)
    fliplr=0.5,                 # horizontal flip (standard)
    weight_decay=0.0004,        # small L2 regularization
    device='cpu',                   # 0 for GPU, or 'cpu'
    name="yolov11_T2-n200",    # experiment name
)
