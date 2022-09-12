import json
import os
import glob
from tqdm import tqdm

def find_nolabel(img_dir = 'data/custom/images'):
    imgfile = glob.glob(os.path.join(img_dir,'*'))
    postfix = ''
    for path in tqdm(imgfile):
        label = path.replace('images','labels').replace('.jpg',f'{postfix}.png')
        if not os.path.exists(label):
            print(path)

def checktxt(files_path):
    postfix = ''
    for file_path in files_path:
        with open(file_path,'r') as f:
            for line in tqdm(f.readlines(),desc=file_path):
                line = line.strip()
                img_file,label_file = line.split(' ')
                if img_file.replace('images','labels').replace('.jpg',f'{postfix}.png') != label_file:
                    print(img_file)
                    print(label_file)
                    print('')

def check_rate(file):
    collect_num = 0
    crawl_num = 0
    with open(file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            img = line.split(' ')[0]
            img = img.split('/')[-1]
            if img.startswith('2022'):
                collect_num += 1
            else:
                crawl_num += 1
    print('collect number: {}\ncrawl number: {}\n'.format(collect_num,crawl_num))

def check_with_floor(file):
    data_root = '/home/gwc/gwc/code/PaddleSeg/data/custom'
    with_floor = 0
    no_floor = 0
    with open(file,'r',encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            img_name = line.split(' ')[0]
            img_path = os.path.join(data_root,img_name)
            json_path = img_path.replace('images','annotations').replace('.jpg','.json')
            with open(json_path) as f:
                data = json.load(f)
                objects = data['objects']
                if len(objects)>3:
                    with_floor += 1
                else:
                    no_floor += 1
    print(f"with floor: {with_floor}, no floor: {no_floor}")


if __name__ == "__main__":
    # find_nolabel()
    # checktxt(['data/custom/train.txt','data/custom/test.txt','data/custom/val.txt'])
    # check_rate('data/custom/train.txt')
    # check_rate('data/custom/val.txt')
    # check_rate('data/custom/test.txt')
    check_with_floor('data/custom/train.txt')
    check_with_floor('data/custom/val.txt')
    check_with_floor('data/custom/test.txt')
