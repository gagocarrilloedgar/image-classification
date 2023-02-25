from ultralytics import YOLO


model = YOLO("yolov8x-cls.yaml")

# https://docs.ultralytics.com/cfg/
if __name__ == '__main__':
    # Train the model
    model.train(data="/home/opc/git/ultralytics/datasets/reto/", epochs=300, patience=75, imgsz=640, verbose=True,
            augment=True,
                mosaic=0.0,
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                degrees=0,
                translate=0.1,
                scale=0.2,
                shear=0.0,
                flipud=0.0,
                fliplr=0.5,
                mixup=0.05
    )


