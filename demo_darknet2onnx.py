import pathlib
import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
import torch

from tool.utils import *
from tool.darknet2onnx import *


def main(cfg_file, namesfile, weight_file, image_path, batch_size, onnx_file_name=None, inference_torch=False):
    print(f'inference_torch {inference_torch}')
    if batch_size <= 0:
        onnx_path_demo, torchmodel = transform_to_onnx(cfg_file, weight_file, batch_size, onnx_file_name)
    else:
        # Transform to onnx as specified batch size
        # transform_to_onnx(cfg_file, weight_file, batch_size, onnx_file_name)

        # Transform to onnx as demo
        onnx_path_demo, torchmodel = transform_to_onnx(cfg_file, weight_file, 1, onnx_file_name)

    session = onnxruntime.InferenceSession(onnx_path_demo)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    if os.path.isdir(image_path):
        dir = os.path.dirname(image_path)
        dir = os.path.join(dir, 'pytorch-YOLOv4')
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        images = os.listdir(image_path)
        for img in images:
            img_path = os.path.join(image_path, img)
            image_src = cv2.imread(img_path)
            out_name = os.path.join(dir, img)
            if inference_torch:
                detect_torch(torchmodel, image_src, namesfile, conf=0.6, nms=0.213, out_name=out_name)
            else:
                detect(session, image_src, namesfile, conf=0.4, nms=0.5, out_name=out_name)
    else:
        image_src = cv2.imread(image_path)
        if inference_torch:
            detect_torch(torchmodel, image_src, namesfile)
        else:
            detect(session, image_src, namesfile)


def onnx_infer(image_path, onnx_file_name=None, namesfile='data/coco.names'):
    print(f'onnx_infer {onnx_file_name}')
    session = onnxruntime.InferenceSession(onnx_file_name)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    if os.path.isdir(image_path):
        dir = os.path.dirname(image_path)
        dir = os.path.join(dir, 'pytorch-YOLOv4')
        pathlib.Path(dir).mkdir(parents=True, exist_ok=True)
        images = os.listdir(image_path)
        for img in images:
            img_path = os.path.join(image_path, img)
            image_src = cv2.imread(img_path)
            out_name = os.path.join(dir, img)
            detect(session, image_src, namesfile, conf=0.4, nms=0.5, out_name=out_name)
    else:
        image_src = cv2.imread(image_path)
        detect(session, image_src, namesfile)

def resize(img, new_size, letter_box=False, interpolation=cv2.INTER_LINEAR, color=(0, 0, 0)):
    if not letter_box:
        return cv2.resize(img, new_size, interpolation=interpolation), new_size
    else:
        shape = img.shape[:2]  # current shape [height, width]

        r = min(new_size[1] / shape[0], new_size[0] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_size[0] - new_unpad[0], new_size[1] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if r != 1:  # resize
            img = cv2.resize(img, new_unpad, interpolation=interpolation)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, new_unpad

def detect(session, image_src, namesfile, conf=0.4, nms=0.6, out_name='predictions_onnx_ciou.jpg'):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized, unpad_size = resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR, letter_box=True)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})

    boxes = post_processing(img_in, conf, nms, outputs)

    class_names = load_class_names(namesfile)

    plot_boxes_cv2(image_src, boxes[0], savename=out_name, class_names=class_names, input_size=(img_in.shape[3], img_in.shape[2]), unpad_size=unpad_size)


def detect_torch(darknet, image_src, namesfile, conf=0.4, nms=0.6, out_name='predictions_onnx_ciou.jpg'):
    IN_IMAGE_H = darknet.width
    IN_IMAGE_W = darknet.height

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    print("Shape of the network input: ", img_in.shape)

    # Compute
    darknet.eval()
    img_in = torch.Tensor(img_in)
    outputs = darknet(img_in)
    # for o in outputs:
    #     o_1 = torch.sigmoid(o[:, 4::6, :, :])
    #     o_2 = torch.sigmoid(o[:, 5::6, :, :])
    #
    #     o_1xo2 = o_1*o_2
    #     print(f'{o.shape[-1]}, o1 {torch.max(o_1)}, o2 {torch.max(o_2)}, o1xo2 {torch.max(o_1xo2)}')
    conf_max = torch.max(outputs[1])
    print(conf_max)
    boxes = post_processing(img_in, conf, nms, outputs)

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(image_src, boxes[0], savename=out_name, class_names=class_names)


if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    if len(sys.argv) >= 6:
        cfg_file = sys.argv[1]
        namesfile = sys.argv[2]
        weight_file = sys.argv[3]
        image_path = sys.argv[4]
        batch_size = int(sys.argv[5])
        onnx_file_name = sys.argv[6] if len(sys.argv) >=7 else None
        torch_inference = sys.argv[7] if len(sys.argv) >=8 else False

        main(cfg_file, namesfile, weight_file, image_path, batch_size, onnx_file_name, torch_inference)
    elif len(sys.argv) == 3:
        image_path = sys.argv[1]
        onnx_file_name = sys.argv[2]
        onnx_infer(image_path, onnx_file_name=onnx_file_name)
    else:
        print('Please run this way:\n')
        print('  python demo_onnx.py <cfgFile> <namesFile> <weightFile> <imageFile> <batchSize>')
