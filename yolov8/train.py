from ultralytics import YOLO

# Load a model
model = YOLO("/research/dept8/fyp22/pah2203/AIST4010/project/yolov8/runs/detect/Deepfashion2_yolov8m_cn_e50/weights/best.pt")  # build a new model from scratch

# Use the model
model.train(data="Deepfashion2.yaml", epochs=10, imgsz=640, batch=8, name="Deepfashion2_yolov8m_cn_e60")  # train the model
metrics = model.val()  # evaluate model performance on the validation set
#results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
success = model.export()  # export the model to ONNX format