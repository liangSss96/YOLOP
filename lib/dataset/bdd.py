'''
Descripti
Version: 2.0
Autor: ls
Date: 2023-06-12 10:54:11
LastEditors: ls
LastEditTime: 2023-07-19 14:33:22
'''
import numpy as np
import json

from termcolor import colored

import torchvision.transforms as transforms

from .convert import convert, id_dict, id_dict_single
from rich.progress import track

from .AutoDriveDataset import AutoDriveDataset
import cv2

class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        # self.cfg = cfg
        self.class_dict = cfg.DATASET.CLASS_NAMES
        self.single_cls = cfg.TRAIN.SINGLE_CLS
        self.db = self._get_db_ls()
  

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes  # ORG_IMG_SIZE
        for mask in track(list(self.mask_list)):
            mask_path = str(mask)
            label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".json")
            image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
            lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
            with open(label_path, 'r') as f:
                label = json.load(f)
            data = label['frames'][0]['objects']
            data = self.filter_data(data)
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['category']
                if category == "traffic light":
                    color = obj['attributes']['trafficLightColor']
                    category = "tl_" + color
                if category in id_dict.keys():
                    x1 = float(obj['box2d']['x1'])
                    y1 = float(obj['box2d']['y1'])
                    x2 = float(obj['box2d']['x2'])
                    y2 = float(obj['box2d']['y2'])
                    cls_id = id_dict[category]
                    if self.single_cls:
                         cls_id=0
                    gt[idx][0] = cls_id
                    box = convert((width, height), (x1, x2, y1, y2))
                    gt[idx][1:] = list(box)
                

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def _get_db_ls(self):
        """
        get database from the annotation file 

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print(colored('Start building the database....','green'))
        gt_db = []
        # 图片的大小是动态的。需要更改
        height, width = self.shapes  # ORG_IMG_SIZE
        for image_file in track(list(self.img_root.iterdir())):
            image_path = str(image_file)
            mask_path = image_path.replace(str(self.img_root), str(self.mask_root)).replace(".jpg", ".png")
            label_path = image_path.replace(str(self.img_root), str(self.label_root)).replace(".jpg", ".json")
            lane_path = image_path.replace(str(self.img_root), str(self.lane_root)).replace(".jpg", ".png")
            gt = np.zeros((0, 5))   # 某一个标注文件的gt列表  目标识别
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            print(height, '     ', width)
            if self.cfg.DATASET.LABELISAVAILABLE:
                with open(label_path, 'r') as f:
                    label = json.load(f)
                data = label['frames'][0]['objects']
                # 过滤出2D目标
                data = self.filter_data(data)
                # 对一个标注文件来说的GT
                gt = np.zeros((len(data), 5)) 
                
                for idx, obj in enumerate(data):
                    category = obj['category']
                    # kitti数据集中的红绿灯有两级的属性
                    if category == "traffic light":
                        color = obj['attributes']['trafficLightColor']
                        category = "tl_" + color
                    if category in id_dict.keys():
                        x1 = float(obj['box2d']['x1'])
                        y1 = float(obj['box2d']['y1'])
                        x2 = float(obj['box2d']['x2'])
                        y2 = float(obj['box2d']['y2'])
                        cls_id = id_dict[category]
                        if self.single_cls:
                            cls_id=0
                        gt[idx][0] = cls_id
                        box = convert((width, height), (x1, x2, y1, y2))   # 归一化后的xywh
                        gt[idx][1:] = list(box)
                

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path
            }]

            # rec ['image': image_path,'label': gt [[cls_id, xywh], [cls_id, xywh], ..],'mask': mask_path,'lane': lane_path]

            gt_db += rec
        print(colored('Database building completed....','green'))
        return gt_db

    # 过滤类别使用
    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if self.single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass


if __name__ == "__main__":
    from lib.config import cfg
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    print(cfg.TRAIN.SINGLE_CLS)
    dataset = BddDataset(cfg, True, [640, 640], transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    print(len(dataset))
    img, target, img_path, shapes = dataset[2]

