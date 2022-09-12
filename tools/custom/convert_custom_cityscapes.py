# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
File: convert_cityscapes.py
This file is based on https://github.com/mcordts/cityscapesScripts to generate **labelTrainIds.png for training.
Before running, you should download the cityscapes form https://www.cityscapes-dataset.com/ and make the folder
structure as follow:
cityscapes
|
|--leftImg8bit
|  |--train
|  |--val
|  |--test
|
|--gtFine
|  |--train
|  |--val
|  |--test
"""

import os
import argparse
from multiprocessing import Pool, cpu_count
import glob

from pathlib import Path
ROOT = Path(__file__).parent.parent.parent.resolve().absolute().__str__()
from json2labelImg_custom import json2labelImg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate **labelTrainIds.png for training')
    parser.add_argument(
        '--cityscapes_path',
        dest='cityscapes_path',
        help='cityscapes path',
        type=str,
        default='data/custom')

    parser.add_argument(
        '--num_workers',
        dest='num_workers',
        help='How many processes are used for data conversion',
        type=int,
        default=cpu_count())
    return parser.parse_args()


def gen_labelTrainIds(json_file):
    # label_file = json_file.replace("annotations","labels").replace(".json", "_labelTrainIds.png")
    label_file = json_file.replace("annotations","labels").replace(".json", ".png")
    label_dir = os.path.dirname(label_file)
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
    json2labelImg(json_file, label_file, "trainIds")

def gen_labelColors(json_file):
    # label_file = json_file.replace("annotations","labels").replace(".json", "_labelTrainIds.png")
    label_file = json_file.replace("annotations","colors").replace(".json", ".png")
    label_dir = os.path.dirname(label_file)
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
    json2labelImg(json_file, label_file, "color")

def main():
    args = parse_args()
    fine_path = args.cityscapes_path
    json_files = glob.glob(os.path.join(fine_path, 'annotations', '*.json'))
    label_map_path = os.path.join(fine_path, 'labels')
    color_map_path = os.path.join(fine_path, 'colors')
    if not os.path.exists(label_map_path):
        os.mkdir(label_map_path)
    if not os.path.exists(color_map_path):
        os.mkdir(color_map_path)

    print('generating **_labelTrainIds.png')
    p = Pool(args.num_workers)
    for f in json_files:
        f = os.path.join(ROOT,f)
        p.apply_async(gen_labelTrainIds, args=(f, ))
        p.apply_async(gen_labelColors, args=(f, ))
    p.close()
    p.join()


if __name__ == '__main__':
    main()
