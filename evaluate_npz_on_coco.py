"""
A script to evaluate the model's performance using pre-trained weights using COCO API.
Example usage: python evaluate_on_coco.py -dir D:\cocoDataset\val2017\val2017 -gta D:\cocoDataset\annotatio
ns_trainval2017\annotations\instances_val2017.json -c cfg/yolov4-smaller-input.cfg -g 0
Explanation: set where your images can be found using -dir, then use -gta to point to the ground truth annotations file
and finally -c to point to the config file you want to use to load the network using.
"""

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
from PIL import Image, ImageDraw, ImageFont
from easydict import EasyDict as edict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from cfg import Cfg
from tool.darknet2pytorch import Darknet
from tool.utils import load_class_names
import tool.utils as utils
from tool.config import parse_cfg
from tqdm import tqdm
import cv2
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

def convert_cat_id(single_annotation):
    cat = single_annotation['category_id']
    if cat >= 1 and cat <= 11:
        cat = cat + 1
    elif cat >= 13 and cat <= 25:
        cat = cat + 2
    elif cat >= 27 and cat <= 28:
        cat = cat + 3
    elif cat >= 31 and cat <= 44:
        cat = cat + 5
    elif cat >= 46 and cat <= 65:
        cat = cat + 6
    elif cat == 67:
        cat = cat + 7
    elif cat == 70:
        cat = cat + 9
    elif cat >= 72 and cat <= 82:
        cat = cat + 10
    elif cat >= 84 and cat <= 90:
        cat = cat + 11
    single_annotation['category_id'] = cat
    return single_annotation

def convert_cat_id_and_reorientate_bbox(single_annotation):
    cat = single_annotation['category_id']
    bbox = single_annotation['bbox']
    # x, y, w, h = bbox
    x1, y1, x2, y2 = bbox#x - w / 2, y - h / 2, x + w / 2, y + h / 2
    w, h = x2 - x1, y2 - y1

    if 0 <= cat <= 10:
        cat = cat + 1
    elif 11 <= cat <= 23:
        cat = cat + 2
    elif 24 <= cat <= 25:
        cat = cat + 3
    elif 26 <= cat <= 39:
        cat = cat + 5
    elif 40 <= cat <= 59:
        cat = cat + 6
    elif cat == 60:
        cat = cat + 7
    elif cat == 61:
        cat = cat + 9
    elif 62 <= cat <= 72:
        cat = cat + 10
    elif 73 <= cat <= 79:
        cat = cat + 11



    # if cat >= 1 and cat <= 11:
    #     cat = cat + 1
    # elif cat >= 13 and cat <= 25:
    #     cat = cat + 2
    # elif cat >= 27 and cat <= 28:
    #     cat = cat + 3
    # elif cat >= 31 and cat <= 44:
    #     cat = cat + 5
    # elif cat >= 46 and cat <= 65:
    #     cat = cat + 6
    # elif cat == 67:
    #     cat = cat + 7
    # elif cat == 70:
    #     cat = cat + 9
    # elif cat >= 72 and cat <= 82:
    #     cat = cat + 10
    # elif cat >= 84 and cat <= 90:
    #     cat = cat + 11

    single_annotation['category_id'] = cat
    single_annotation['bbox'] = [x1, y1, w, h]
    return single_annotation



def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()
    else:
        return obj



def test(all_outout, annotations, cfg, length):
    if not annotations["images"]:
        print("Annotations do not have 'images' key")
        return
    images = annotations["images"]
    # images = images[:10]
    resFile = 'data/coco_val_outputs.json'


    boxes_json = []
    filename = [os.path.join(f'data/temp_{i}.json') for i in range(10)]
    interval = length // len(filename)
    for idx, dict in enumerate(all_outout()):
        image_name = list(dict.keys())[0]
        info = dict[image_name]
        start = time.time()

        layer_output = info['output']
        image_width, image_height = info['size']
        output = list(zip(*layer_output))
        output = [np.concatenate(t, axis=1) for t in output]
        boxes = utils.post_processing(None, 0.1, 0.4, output)
        finish = time.time()
        boxes_json_per_img = []
        if type(boxes) == list:
            # print(len(boxes[0]))
            for box in boxes[0]:
                box_json = {}
                category_id = box[-1]
                score = box[-2]
                bbox_normalized = box[:4]
                box_json["category_id"] = int(category_id)
                box_json["image_id"] = int(image_name.rsplit('.', 1)[0])
                bbox = []
                for i, bbox_coord in enumerate(bbox_normalized):
                    modified_bbox_coord = float(bbox_coord)
                    if i % 2:
                        modified_bbox_coord *= image_height
                    else:
                        modified_bbox_coord *= image_width
                    modified_bbox_coord = round(modified_bbox_coord, 2)
                    bbox.append(modified_bbox_coord)
                box_json["bbox_normalized"] = list(map(lambda x: round(float(x), 2), bbox_normalized))
                # print(f'{box_json["bbox_normalized"]}')
                box_json["bbox"] = bbox
                box_json["score"] = round(float(score), 2)
                box_json["timing"] = float(finish - start)
                boxes_json.append(box_json)
                boxes_json_per_img.append(box_json)
        else:
            print("warning: output from model after postprocessing is not a list, ignoring")
            return
        # img = info['image']
        # print(f'=============================={image_name}======================================')
        # for i, anno in enumerate(boxes_json_per_img):
        #     bbox = anno["bbox"]
        #     bbox_normalized = anno["bbox_normalized"]
        #     category = anno["category_id"]
        #     score = anno["score"]
        #     x1, y1, x2, y2 = bbox
        #     draw = ImageDraw.Draw(img)
        #     draw.rectangle(((x1, y1), (x2, y2)), outline='red')
        #
        #     draw.text((x1, y1), f"{category}_{score:.2f}")
        #     print(f'{i} {bbox_normalized}')
        # img.show()
        # time.sleep(5)
        # img.close()

        if (idx + 1) % interval == 0:
            # print("see box_json: ", box_json)
            with open(filename[idx // interval], 'w') as outfile:
                json.dump(boxes_json, outfile, default=myconverter)
            boxes_json = []

        # namesfile = 'data/coco.names'
        # class_names = load_class_names(namesfile)
        # plot_boxes(img, boxes, 'data/outcome/predictions_{}.jpg'.format(image_id), class_names)

    def merge_JsonFiles(filename, target):
        result = list()
        for f1 in filename:
            with open(f1, 'r') as infile:
                result.extend(json.load(infile))

        unsorted_annotations = result
        sorted_annotations = list(sorted(unsorted_annotations, key=lambda single_annotation: single_annotation["image_id"]))
        sorted_annotations = list(map(convert_cat_id_and_reorientate_bbox, sorted_annotations))
        # reshaped_annotations = defaultdict(list)
        # for annotation in sorted_annotations:
        #     reshaped_annotations[annotation['image_id']].append(annotation)

        with open(target, 'w') as output_file:
            json.dump(sorted_annotations, output_file, default=myconverter)
        return sorted_annotations

    merge_JsonFiles(filename, resFile)

    cocoGt = COCO(cfg.gt_annotations_path)
    cocoDt = cocoGt.loadRes(resFile)


    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

def get_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Test model on test dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, default='-1',
                        help='GPU', dest='gpu')
    parser.add_argument('-dir', '--data-dir', type=str, default=None,
                        help='dataset dir', dest='dataset_dir')
    parser.add_argument('-gta', '--ground_truth_annotations', type=str, default='instances_val2017.json',
                        help='ground truth annotations file', dest='gt_annotations_path')
    parser.add_argument('-w', '--weights_file', type=str, default='weights/yolov4.weights',
                        help='weights file to load', dest='weights_file')
    parser.add_argument('-c', '--model_config', type=str, default='cfg/yolov4.cfg',
                        help='model config file to load', dest='model_config')
    parser.add_argument('-o', type=str, default='',
                        help='onnx path', dest='onnx_path')
    args = vars(parser.parse_args())

    for k in args.keys():
        cfg[k] = args.get(k)
    return edict(cfg)


def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    import datetime
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def do_detect(session, img, conf_thresh, nms_thresh, use_cuda=1):
    t0 = time.time()

    # if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
    #     img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    # elif type(img) == np.ndarray and len(img.shape) == 4:
    #     img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    # else:
    #     print("unknow image type")
    #     exit(-1)

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32) / 255.0
    t1 = time.time()
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img})

    t2 = time.time()

    print('-----------------------------------')
    print('           Preprocess : %f' % (t1 - t0))
    print('      Model Inference : %f' % (t2 - t1))
    print('-----------------------------------')

    return utils.post_processing(img, conf_thresh, nms_thresh, outputs)
from tool.yolo_layer import YoloLayer

def create_yolo_layer(blocks):
    model = []
    strides = [8, 16, 32]
    for block in blocks:
        if block['type'] == 'yolo':
            yolo_layer = YoloLayer()
            anchors = block['anchors'].split(',')
            anchor_mask = block['mask'].split(',')
            yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
            yolo_layer.anchors = [float(i) for i in anchors]
            yolo_layer.num_classes = int(block['classes'])
            num_classes = yolo_layer.num_classes
            yolo_layer.num_anchors = int(block['num'])
            yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
            yolo_layer.stride = strides[len(model)]
            yolo_layer.scale_x_y = float(block['scale_x_y'])
            # yolo_layer.object_scale = float(block['object_scale'])
            # yolo_layer.noobject_scale = float(block['noobject_scale'])
            # yolo_layer.class_scale = float(block['class_scale'])
            # yolo_layer.coord_scale = float(block['coord_scale'])
            model.append(yolo_layer)
    return model

if __name__ == "__main__":
    logging = init_logger(log_dir='log')
    cfg = get_args(**Cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    import onnxruntime
    blocks = parse_cfg(cfg.cfgfile)
    models = create_yolo_layer(blocks)
    dir = '/home/jason/Downloads/val2017_512_output'
    img_dir = cfg.dataset_dir#'/home/jason/Downloads/val2017'
    npz_all = os.listdir(dir)
    length = len(npz_all)
# yolov4_leaky/conv95_centers shape is (10, 64, 64, 6)
# yolov4_leaky/conv95_scales shape is (10, 64, 64, 6)
# yolov4_leaky/conv95_obj shape is (10, 64, 64, 3)
# yolov4_leaky/conv95_probs shape is (10, 64, 64, 240)
    layer_name = ['conv95', 'conv103', 'conv110']
    net = 'yolov4-512'#'yolov4_leaky'
    # all_outout = {}
    def all_outout():
        for npz in tqdm(npz_all):
            npz_outputs = np.load(os.path.join(dir, npz))
            img_name = f'{npz.rsplit(".", 1)[0]}.jpg'
            img = Image.open(os.path.join(img_dir, img_name))
            size = img.size
            yolo_outputs = []
            for yolo, layer in zip(models, layer_name):
                # centers = npz_outputs[f'{net}/{layer}_centers']
                # scales = npz_outputs[f'{net}/{layer}_scales']
                # obj = npz_outputs[f'{net}/{layer}_obj']
                # probs = npz_outputs[f'{net}/{layer}_probs']
                # outputs = np.concatenate([centers, scales, obj, probs], axis=-1)
                outputs = npz_outputs[f'{net}/{layer}']
                outputs = np.expand_dims(outputs, axis=0)
                # if layer == 'conv95':
                #     print(centers[..., 0])
                outputs = np.transpose(outputs, (0, 3, 1, 2))
                outputs = torch.tensor(outputs, dtype=torch.float32)
                yolo_output = yolo.forward(outputs)
                yolo_outputs.append(yolo_output)
            # all_outout.update({img_name: {'output': yolo_outputs, 'size': size}})
            # print(f'{img_name} {size}')
            yield {img_name: {'output': yolo_outputs, 'size': size}}
        



    annotations_file_path = cfg.gt_annotations_path
    with open(annotations_file_path) as annotations_file:
        try:
            annotations = json.load(annotations_file)
        except:
            print("annotations file not a json")
            exit()
    test(all_outout,
         annotations=annotations,
         cfg=cfg, length=length)
