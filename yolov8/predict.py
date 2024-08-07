from __future__ import division
from ultralytics import YOLO
import cv2
import numpy as np
import argparse
from fastai.vision.all import *
import gc
import torch

model = YOLO("/research/dept8/fyp22/pah2203/AIST4010/project/yolov8/runs/detect/Deepfashion2_yolov8m_cn_e50/weights/best.pt")

# load model from fashion-ai
PATH = "/research/dept8/fyp22/pah2203/AIST4010/project/datasets/Deepfashion/"
TRAIN_PATH = "/research/dept8/fyp22/pah2203/AIST4010/project/fashion-ai/clothes-categories/multilabel-train.csv"
TEST_PATH = "/research/dept8/fyp22/pah2203/AIST4010/project/fashion-ai/clothes-categories/multilabel-test.csv"
CLASSES_PATH = "/research/dept8/fyp22/pah2203/AIST4010/project/fashion-ai/clothes-categories/attribute-classes.txt"

train_df = pd.read_csv(TRAIN_PATH)

def get_x(r): return PATH+r['image_name'].replace("(",")")
def get_y(r): return r['labels'].split(',')

def splitter(df):
    train = df.index[df['is_valid']==0].tolist()
    valid = df.index[df['is_valid']==1].tolist()
    return train,valid

dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=splitter,
                   get_x=get_x, 
                   get_y=get_y,
                   item_tfms=RandomResizedCrop(224, min_scale=0.8),
                   batch_tfms=aug_transforms())

dls = dblock.dataloaders(train_df, num_workers=5, device=torch.device('cuda'))

class LabelSmoothingBCEWithLogitsLossFlat(BCEWithLogitsLossFlat):
    def __init__(self, eps:float=0.1, **kwargs):
        self.eps = eps
        super().__init__(thresh=0.2, **kwargs)
    
    def __call__(self, inp, targ, **kwargs):
        # https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/166833#929222
        targ_smooth = targ.float() * (1. - self.eps) + 0.5 * self.eps
        return super().__call__(inp, targ_smooth, **kwargs)
    
    def __repr__(self):
        return "FlattenedLoss of LabelSmoothingBCEWithLogits()"

metrics=[FBetaMulti(2.0, 0.2, average='samples'), partial(accuracy_multi, thresh=0.2)]
wd      = 5e-7 #weight decay parameter
opt_func = partial(ranger, wd=wd)

learn = vision_learner(dls, resnet50, loss_func=LabelSmoothingBCEWithLogitsLossFlat(),
            metrics=metrics, opt_func=opt_func).to_fp16()
learn.load("/research/dept8/fyp22/pah2203/AIST4010/project/yolov8/fashion-ai/trained_model_resnet50")

#result= model("/research/dept8/fyp22/pah2203/AIST4010/project/datasets/Deepfashion2/val/images/018224.jpg")[0]
IMG_PATH = "/research/dept8/fyp22/pah2203/AIST4010/project/yolov8/real_example/"
img_ids = [str(i).zfill(6) for i in range(1,12)]  # use first 10 img
for img_id in img_ids:
    attr_list = []
    results = model(IMG_PATH+f"{img_id}.jpg")
    for result in results:
        res_plotted = result.plot(show_conf=False)

        #cv2.imwrite(f"/research/dept8/fyp22/pah2203/AIST4010/project/yolov8/real_example_detected/{img_id}_detected.jpg", res_plotted)
        cls = result.names[np.argmax(result.probs)]
        # Get box coordinates
        for box in result.boxes:
            xyxy = box.xyxy.cpu().numpy()
            x1 = round(xyxy[0][0])
            x2 = round(xyxy[0][2])
            y1 = round(xyxy[0][1])
            y2 = round(xyxy[0][3])

            # Crop img for testing
            ori_img = result[0].orig_img
            crop_img = ori_img[y1:y2, x1:x2]
            # Attribute Prediction from fashion-ai
            predicted = learn.predict(crop_img)
            print(predicted[0])
            attr = ",".join(predicted[0])
            attr_list.append(attr)
        res_plotted_with_attr = result.plot(show_conf=False, attr_list=attr_list)
        cv2.imwrite(f"/research/dept8/fyp22/pah2203/AIST4010/project/yolov8/real_example_detected/{img_id}_detected.jpg", res_plotted_with_attr)

        



