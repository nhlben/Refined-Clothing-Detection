# Refined Clothing Detection with DeepFashion2 using YOLO

This project is about the individual project in AIST4010 - Refined Clothing Detection with DeepFashion2 using YOLO

## Introduction

This project aims to perform clothing detection, which is a kind of multiple object detection where given an image, all the clothing in the image will be detected, classified, and finally given a bounding box. This projects mainly uses the YOLO architecture, experimenting the result using YOLOv5, YOLOv7, YOLOv8. After finetuning the best YOLO model to get the final result, a further fashion attribute classification is proposed in order to predict more attributes of the clothing.

<p align="center">
  <img src="https://github.com/user-attachments/assets/867fc700-8dd9-43ae-9a27-55fadd11857d" width=500 alt><br>
  <em>Example of clothes detection output</em>
</p>

## Clothes Detection

### Dataset
For clothing detection, the DeepFashion2 dataset is used, which contains 491K annotated clothes images of 13 categories. 

<b>Offcial Website</b>:  https://github.com/switchablenorms/DeepFashion2

### Models

- YOLOv5 | Official Site: https://github.com/ultralytics/yolov5
- YOLOv7 | Official Site: https://github.com/WongKinYiu/yolov7
- YOLOv8 | Official Site: https://github.com/ultralytics/ultralytics

### Preliminary Results

Evaluation result on the validation set, all metrics are averaged over all classes:

| Model  | Number of Epochs |  mAP  | mAP_50 |
| :-------------: | :-------------: | :-------------: | :-------------: |
|  YOLOv5s  | 10  | 0.453 | 0.601 |
|  YOLOv5m  | 5  | 0.353 | 0.499 |
|  YOLOv7-tiny  | 10  | 0.391 | 0.547 |
|  YOLOv7  | 5  | 0.227 | 0.38 |
|  YOLOv8s  | 10  | 0.588 | 0.705 |
|  YOLOv8m  | 5  | 0.494 | 0.603 |
|  **YOLOv8m**  | **10**  | **0.629** | **0.738** |

The **YOLOv8m** model performs the best, achieving a over 70% mAP_50

### Finetuning Results
The YOLOv8m is further trained for 50 more epochs:

| Epoch |  mAP  | mAP_50 |
| :-------------: | :-------------: | :-------------: |
| 10 | 0.629|0.738|
|20 |0.695 |0.795|
|30 |0.712 |0.810|
|40 |0.721 |0.818|
|50 |0.725 |0.821|
|60 |0.725 |0.822|

Loss and Metrics:
<p align="center">
  <img src="https://github.com/user-attachments/assets/141f2bf6-dadc-4e30-94b8-e5c7ee8dd14f" width=800 alt><br>
</p>

Precision-Recall curve of final YOLOv8m model:
<p align="center">
  <img src="https://github.com/user-attachments/assets/9a57c1fd-2bd0-49d0-833f-e88fdcc3a343" width=600 alt><br>
</p>

Clothes detection result sample using final YOLOv8m model:
<p align="center">
  <img src="https://github.com/user-attachments/assets/f2d401a3-d8ad-48d0-918c-acafeb150acd" width=600 alt><br>
</p>

## Fashion Attribute Classification

There are 1000 different attributes in the dataset, therefore if all 1000 attributes are used, the performance of the model may not be very good. In this task, only 98 attributes that correspond to style, fabric, season, and type of the pattern are selected. After filtering out images that have the selected attributes, there are 137108 images in the dataset.

### Dataset
For fashion attribute classification, the DeepFashion dataset is used, which contains 289K clothes images labeled with 1000 different attributes.

<b>Offcial Website</b>: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

### Models

Resnet50 pretrained on ImageNet is used, and loaded using fastai API

### Training Result

After training the model for 10 epochs using RAdam as provided by fastai, F2-score is used to evaluate the result.

F2-score of Resnet50 in training:
<p align="center">
  <img src="https://github.com/user-attachments/assets/124a7599-20a8-4a0c-8b82-1895dd101336" width=400 alt><br>
</p>

A F2-score of 0.485126 is obtained. The result is not very outstanding, but sufficient to complete the task at a proper level.

## Final Result

After training both models, the clothing image can be detected with a more accurate attributes, as shown below.

<p float="left" align="center">
  <img src="https://github.com/user-attachments/assets/5f96c74d-b651-4d0d-9aca-0c20bad66a91" width=200 >
  <img src="https://github.com/user-attachments/assets/5059db8e-34c0-4e2d-aff3-f546250b6529" width=200 ><br>
  <em>Result of detected clothes without attribute classification</em>
</p>

<p float="left" align="center">
  <img src="https://github.com/user-attachments/assets/6f75be1c-7527-4538-92ed-e1871f1a3b0a" width=200 >
  <img src="https://github.com/user-attachments/assets/e7bcfea5-dee8-4ad4-b219-096ea127dbf7" width=200 ><br>
  <em>Final result of detected clothes with attribute classification</em>
</p>

<p float="left" align="center">
  <img src="https://github.com/user-attachments/assets/10603b11-fd05-49fb-9012-3b3b64bf119f" width=200 >
  <img src="https://github.com/user-attachments/assets/f07e8da8-a215-4022-802a-22ba53c8cab7" width=200 ><br>
  <em>Detection example of new images searched online</em>
</p>


