from ultralytics import YOLO


model = YOLO("../models/nuwe.pt") # load pretrained model (recommended for training)


if __name__ == '__main__':
    results = model.val()
