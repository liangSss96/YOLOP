'''
Description: 
Version: 2.0
Autor: ls
Date: 2023-06-19 14:41:25
LastEditors: ls
LastEditTime: 2023-06-29 09:51:10
'''
from lib.dataset import BddDataset
import torch
from lib.config import cfg 
from lib.models import get_net

import sys
sys.path.insert(0, '/home/ls/ls_disk/project/yolo_family/yolov7')

def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}

if __name__ == "__main__":
    aa = torch.load("./weights/End-to-end.pth", map_location=torch.device('cpu'))


    # device = torch.device('cuda', 0)
    model = get_net(cfg)
    aa['state_dict'] = intersect_dicts(model.state_dict(), aa['state_dict'])
    model.load_state_dict(aa['state_dict'], strict=False)
    for k, v in model.named_parameters():
        print(k)
