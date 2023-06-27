'''
Descripti
Version: 2.0
Autor: ls
Date: 2023-06-12 10:54:11
LastEditors: ls
LastEditTime: 2023-06-27 18:47:29
'''
import numpy as np
import json

import sys
import os
# sys.path.append(os.getcwd())

from .convert import convert, id_dict, id_dict_single
from rich.progress import track

from .AutoDriveDataset import AutoDriveDataset

class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.cfg = cfg
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
        print('building database...')
        gt_db = []
        height, width = self.shapes  # ORG_IMG_SIZE
        for image_file in track(list(self.img_root.iterdir())):
            image_path = str(image_file)
            mask_path = image_path.replace(str(self.img_root), str(self.mask_root)).replace(".jpg", ".png")
            label_path = image_path.replace(str(self.img_root), str(self.label_root)).replace(".jpg", ".json")
            lane_path = image_path.replace(str(self.img_root), str(self.lane_root)).replace(".jpg", ".png")
            gt = np.zeros((0, 5))   # 某一个标注文件的gt列表
            if self.cfg.DATASET.LABELISAVAILABLE:
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
                        box = convert((width, height), (x1, x2, y1, y2))   # 归一化后的xywh
                        gt[idx][1:] = list(box)
                

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path
            }]

            # rec ['image': image_path,'label': gt [[idx, [xywh]], [idx, [xywh]], ..],'mask': mask_path,'lane': lane_path]

            gt_db += rec
        print('database build finish')
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
    print(cfg.TRAIN.SINGLE_CLS)
    dataset = BddDataset(cfg, True, [640, 640])
    print(len(dataset))
    print(dataset[0])