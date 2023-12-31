from ultralytics import YOLO

# Load a model
model = YOLO('yolov5n.yaml')  # build a new model from YAML
# model = YOLO('C:/Users/XIR1SBY/Desktop/camera/yolo/yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('C:/Users/XIR1SBY/Desktop/camera/runs/detect/train/args.yaml').load('C:/Users/XIR1SBY/Desktop/camera/yolo/yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
# results = model.train(data='VisDrone.yaml', epochs=20, imgsz=640)