from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"D:\NOBORRAR\cfg\model\yolov6.yaml")  # build a new model from scratch
    # Use the model
    model.train(data = r"D:\NOBORRAR\SDC_dataset\dataset.yaml",cfg=r"D:\NOBORRAR\cfg\default.yaml")  # train the model
