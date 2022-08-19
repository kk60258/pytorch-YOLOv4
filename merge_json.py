import json
import numpy as np
import datetime
import os

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

def merge_JsonFiles(filename, target):
    result = list()
    for f1 in filename:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    with open(target, 'w') as output_file:
        json.dump(result, output_file, default=myconverter)

filename = [os.path.join(f'data/temp_{i}.json') for i in range(10)]
resFile = 'data/coco_val_outputs.json'
merge_JsonFiles(filename, resFile)