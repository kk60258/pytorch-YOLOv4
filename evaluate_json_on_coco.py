import argparse
import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw
from easydict import EasyDict as edict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from cfg import Cfg
from tool.darknet2pytorch import Darknet
from tool.utils import load_class_names
import tool.utils as utils
from tool.config import parse_cfg
from tqdm import tqdm

gt_annotations_path ='/home/jason/Downloads/annotations_trainval2017/annotations/instances_val2017.json'
annType = "bbox"
dataset_dir = '/home/jason/Downloads/val2017'

def get_class_name(cat):
    class_names = load_class_names("./data/coco.names")
    if cat >= 1 and cat <= 11:
        cat = cat - 1
    elif cat >= 13 and cat <= 25:
        cat = cat - 2
    elif cat >= 27 and cat <= 28:
        cat = cat - 3
    elif cat >= 31 and cat <= 44:
        cat = cat - 5
    elif cat >= 46 and cat <= 65:
        cat = cat - 6
    elif cat == 67:
        cat = cat - 7
    elif cat == 70:
        cat = cat - 9
    elif cat >= 72 and cat <= 82:
        cat = cat - 10
    elif cat >= 84 and cat <= 90:
        cat = cat - 11
    return class_names[cat]

cocoGt = COCO(gt_annotations_path)
cocoDt = cocoGt.loadRes('data/coco_val_outputs.json')

# with open(gt_annotations_path, 'r') as f:
#     gt_annotation_raw = json.load(f)
#     gt_annotation_raw_images = gt_annotation_raw["images"]
#     gt_annotation_raw_labels = gt_annotation_raw["annotations"]

# rgb_label = (255, 0, 0)
# rgb_pred = (0, 255, 0)
#
# print('123')
# with open('temp.json', 'r') as f:
#     sorted_annotations = json.load(f)
# print('456')
# reshaped_annotations = defaultdict(list)
# for annotation in sorted_annotations:
#     reshaped_annotations[annotation['image_id']].append(annotation)
#
#
# for i, image_id in enumerate(reshaped_annotations):
#     image_annotations = reshaped_annotations[image_id]
#     gt_annotation_image_raw = list(filter(
#         lambda image_json: image_json['id'] == image_id, gt_annotation_raw_images
#     ))
#     gt_annotation_labels_raw = list(filter(
#         lambda label_json: label_json['image_id'] == image_id, gt_annotation_raw_labels
#     ))
#     if len(gt_annotation_image_raw) == 1:
#         image_path = os.path.join(dataset_dir, gt_annotation_image_raw[0]["file_name"])
#         actual_image = Image.open(image_path).convert('RGB')
#         draw = ImageDraw.Draw(actual_image)
#
#         for annotation in image_annotations:
#             x1_pred, y1_pred, w, h = annotation['bbox']
#             x2_pred, y2_pred = x1_pred + w, y1_pred + h
#             cls_id = annotation['category_id']
#             label = get_class_name(cls_id)
#             draw.text((x1_pred, y1_pred), label, fill=rgb_pred)
#             draw.rectangle([x1_pred, y1_pred, x2_pred, y2_pred], outline=rgb_pred)
#         for annotation in gt_annotation_labels_raw:
#             x1_truth, y1_truth, w, h = annotation['bbox']
#             x2_truth, y2_truth = x1_truth + w, y1_truth + h
#             cls_id = annotation['category_id']
#             label = get_class_name(cls_id)
#             draw.text((x1_truth, y1_truth), label, fill=rgb_label)
#             draw.rectangle([x1_truth, y1_truth, x2_truth, y2_truth], outline=rgb_label)
#         actual_image.save("./data/outcome/predictions_{}".format(gt_annotation_image_raw[0]["file_name"]))
#     else:
#         print('please check')
#         break
#     if (i + 1) % 100 == 0: # just see first 100
#         break

imgIds = sorted(cocoGt.getImgIds())
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()