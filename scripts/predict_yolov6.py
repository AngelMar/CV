from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"UDIT_test\train10\weights\best.pt")  # build a new model from scratch
    # Use the model
    model.predict(r"D:\NOBORRAR\video\LA.mp4",imgsz=640, save=True, conf=0.55, iou=0.7)  # train the model
