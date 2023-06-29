import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout


class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """
    def __init__(self, cfg, is_train, inputsize=[640, 640], transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize   # 输入网络图片大小
        self.Tensor = transforms.ToTensor()
        # 数据集的一些路径
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)
        # 训练和验证的路径名称
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.lane_root = lane_root / indicator
        # self.label_list = self.label_root.iterdir()
        # self.img_root = self.img_root.iterdir()

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR   # 缩放
        self.rotation_factor = cfg.DATASET.ROT_FACTOR  # 旋转
        self.flip = cfg.DATASET.FLIP   # 翻转
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)
    
    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

#     def __getitem__(self, idx):
#         """
#         Get input and groud-truth from database & add data augmentation on input

#         Inputs:
#         -idx: the index of image in self.db(database)(list)
#         self.db(list) [a,b,c,...]
#         a: (dictionary){'image':, 'information':}

#         Returns:
#         -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
#         -target: ground truth(det_gt,seg_gt)

#         function maybe useful
#         cv2.imread
#         cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
#         cv2.warpAffine
#         """
#         data = self.db[idx]
#         img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         if self.cfg.num_seg_class == 3:
#             seg_label = cv2.imread(data["mask"])
#         else:
#             seg_label = cv2.imread(data["mask"], 0)  # 单通道的灰度图像
#         lane_label = cv2.imread(data["lane"], 0)
#         print(img.shape)
#         print(lane_label.shape)
#         print(seg_label.shape)
#         resized_shape = self.inputsize   # 模型的输入图像大小
#         if isinstance(resized_shape, list):
#             resized_shape = max(resized_shape)   # 输入图像的长边
#         h0, w0 = img.shape[:2]  # orig hw
#         r = resized_shape / max(h0, w0)  
#         if r != 1:  # always resize down, only resize up if training with augmentation
#             interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
#             img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
#             seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
#             lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
#         h, w = img.shape[:2]
        
#         (img, seg_label, lane_label), ratio, pad = letterbox((img, seg_label, lane_label), resized_shape, auto=True, scaleup=self.is_train)
#         shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
#         # ratio = (w / w0, h / h0)
#         # print(resized_shape)
        
#         det_label = data["label"]
#         labels=[]
#         print("new ratio: ", ratio)
        
#         if det_label.size > 0:
#             # Normalized xywh to pixel xyxy format
#             labels = det_label.copy()
#             labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
#             labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
#             labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
#             labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]
            
#         if self.is_train:
#             combination = (img, seg_label, lane_label)
#             (img, seg_label, lane_label), labels = random_perspective(
#                 combination=combination,
#                 targets=labels,
#                 degrees=self.cfg.DATASET.ROT_FACTOR,
#                 translate=self.cfg.DATASET.TRANSLATE,
#                 scale=self.cfg.DATASET.SCALE_FACTOR,
#                 shear=self.cfg.DATASET.SHEAR
#             )
#             #print(labels.shape)
#             augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)
#             # img, seg_label, labels = cutout(combination=combination, labels=labels)

#             if len(labels):
#                 # convert xyxy to xywh
#                 labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

#                 # Normalize coordinates 0 - 1
#                 labels[:, [2, 4]] /= img.shape[0]  # height
#                 labels[:, [1, 3]] /= img.shape[1]  # width

#             # if self.is_train:
#             # random left-right flip
#             # 翻转问题
#             lr_flip = True
#             if lr_flip and random.random() < 0.5:
#                 img = np.fliplr(img)
#                 seg_label = np.fliplr(seg_label)
#                 lane_label = np.fliplr(lane_label)
#                 if len(labels):
#                     labels[:, 1] = 1 - labels[:, 1]

#             # random up-down flip
#             ud_flip = False
#             if ud_flip and random.random() < 0.5:
#                 img = np.flipud(img)
#                 seg_label = np.filpud(seg_label)
#                 lane_label = np.filpud(lane_label)
#                 if len(labels):
#                     labels[:, 2] = 1 - labels[:, 2]
        
#         else:
#             if len(labels):
#                 # convert xyxy to xywh
#                 labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

#                 # Normalize coordinates 0 - 1
#                 labels[:, [2, 4]] /= img.shape[0]  # height
#                 labels[:, [1, 3]] /= img.shape[1]  # width

#         labels_out = torch.zeros((len(labels), 6))
#         if len(labels):
#             labels_out[:, 1:] = torch.from_numpy(labels)
#         # Convert
#         # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
#         # img = img.transpose(2, 0, 1)
#         # 返回一个连续的array，其内存是连续的
#         img = np.ascontiguousarray(img)
#         # seg_label = np.ascontiguousarray(seg_label)
#         # if idx == 0:
#         #     print(seg_label[:,:,0])

#         if self.cfg.num_seg_class == 3:
#             _,seg0 = cv2.threshold(seg_label[:,:,0],128,255,cv2.THRESH_BINARY)
#             _,seg1 = cv2.threshold(seg_label[:,:,1],1,255,cv2.THRESH_BINARY)
#             _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
#         else:
#             _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)
#             _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV)
#         _,lane1 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY)
#         _,lane2 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY_INV)
# #        _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
#         # # seg1[cutout_mask] = 0
#         # # seg2[cutout_mask] = 0
        
#         # seg_label /= 255
#         # seg0 = self.Tensor(seg0)
#         if self.cfg.num_seg_class == 3:
#             seg0 = self.Tensor(seg0)
#         seg1 = self.Tensor(seg1)
#         seg2 = self.Tensor(seg2)
#         # seg1 = self.Tensor(seg1)
#         # seg2 = self.Tensor(seg2)
#         lane1 = self.Tensor(lane1)
#         lane2 = self.Tensor(lane2)

#         # seg_label = torch.stack((seg2[0], seg1[0]),0)
#         if self.cfg.num_seg_class == 3:
#             seg_label = torch.stack((seg0[0],seg1[0],seg2[0]),0)
#         else:
#             seg_label = torch.stack((seg2[0], seg1[0]),0)
            
#         lane_label = torch.stack((lane2[0], lane1[0]),0)
#         # _, gt_mask = torch.max(seg_label, 0)
#         # _ = show_seg_result(img, gt_mask, idx, 0, save_dir='debug', is_gt=True)
        

#         target = [labels_out, seg_label, lane_label]
#         img = self.transform(img)

#         return img, target, data["image"], shapes

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # print(" load img *****************", img[0][0])

        if self.cfg.DATASET.SEGISAVAILABLE:
            if self.cfg.num_seg_class == 3:
                seg_label = cv2.imread(data["mask"])
            else:
                seg_label = cv2.imread(data["mask"], 0)  # 单通道的灰度图像
        else:
            if self.cfg.num_seg_class == 3:
                seg_label = np.zeros_like(img, dtype=np.uint8)
            else:
                # print("&&&&&&&&&&&&&&&&&&&&&&&&")
                seg_label = np.zeros(img.shape[:2], dtype=np.uint8)
                # print(seg_label)

        if self.cfg.DATASET.LLISAVAILABLE:  
            lane_label = cv2.imread(data["lane"], 0)
        else:
            lane_label = np.zeros(img.shape[:2], dtype=np.uint8)


        # print(img.shape)
        # print(lane_label.shape)
        # print(seg_label.shape)

        resized_shape = self.inputsize   # 模型的输入图像大小
        # if isinstance(resized_shape, list):
        #     resized_shape = max(resized_shape)   # 输入图像的长边
        h0, w0 = img.shape[:2]  # orig hw
        r_h, r_w = resized_shape[1] / h0, resized_shape[0] / w0
        r = min(r_h, r_w)
        # r = resized_shape / max(h0, w0)  
        if r != 1:  # always resize down, only resize up if training with augmentation
            # cv2.INTER_AREA 插值方法来最小化细节损失。而较大的缩放比例可能需要更平滑的插值，因此选择 cv2.INTER_LINEAR 插值方法来实现更平滑的放大效果。
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]  # 新的尺寸，图像的长宽比前后是不变的
        # print(" resize img *****************", img[0][0])
        # print("h, w: ", h, w)
        '''
        TODO
        '''
        (img, seg_label, lane_label), ratio, pad = letterbox((img, seg_label, lane_label), resized_shape, auto=self.cfg.TRAIN.AUTOFILL, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # print(" letterbox img *****************", img[0][0])
        # ratio = (w / w0, h / h0)
        # print(resized_shape)
        
        det_label = data["label"]
        labels=[]
        # print("new ratio: ", ratio)
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]
            
        if self.is_train:
            combination = (img, seg_label, lane_label)
            (img, seg_label, lane_label), labels = random_perspective(
                self.cfg,
                combination=combination,
                targets=labels,
                degrees=self.cfg.DATASET.ROT_FACTOR,  # 旋转
                translate=self.cfg.DATASET.TRANSLATE,  # 裁剪
                scale=self.cfg.DATASET.SCALE_FACTOR,  # 缩放
                shear=self.cfg.DATASET.SHEAR  
            )
            # print(" random_perspective img *****************", img[0][0])
            #print(labels.shape)
            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)
            # img, seg_label, labels = cutout(combination=combination, labels=labels)
            # print(" augment_hsv img *****************", img[0][0])
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            # if self.is_train:
            # random left-right flip
            # 翻转问题
            img_name = data["image"].split("/")[-1]
            # print(img_name)
            if not img_name.startswith("tl"):
                lr_flip = self.cfg.DATASET.LR_FLIP
                if lr_flip and random.random() < 0.5:
                    img = np.fliplr(img)
                    seg_label = np.fliplr(seg_label)
                    lane_label = np.fliplr(lane_label)
                    if len(labels):
                        labels[:, 1] = 1 - labels[:, 1]

                # random up-down flip
                ud_flip = self.cfg.DATASET.UD_FLIP
                if ud_flip and random.random() < 0.5:
                    img = np.flipud(img)
                    seg_label = np.filpud(seg_label)
                    lane_label = np.filpud(lane_label)
                    if len(labels):
                        labels[:, 2] = 1 - labels[:, 2]
        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width
        # [img_index, class, x, y, w, h]
        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        # 返回一个连续的array，其内存是连续的这通常能够提高计算效率，特别是在涉及大量数据的情况下。
        # 在计算机视觉中，经常需要对图像进行处理，而某些操作要求图像是连续的。
        img = np.ascontiguousarray(img)
        # seg_label = np.ascontiguousarray(seg_label)
        # if idx == 0:
        #     print(seg_label[:,:,0])
        
        # 前后背景的图
        if self.cfg.num_seg_class == 3:
            _,seg0 = cv2.threshold(seg_label[:,:,0],128,255,cv2.THRESH_BINARY)
            _,seg1 = cv2.threshold(seg_label[:,:,1],1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        else:
            _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV)
        _,lane1 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY)
        _,lane2 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY_INV)
#        _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        # # seg1[cutout_mask] = 0
        # # seg2[cutout_mask] = 0
        
        # seg_label /= 255
        # seg0 = self.Tensor(seg0)
        if self.cfg.num_seg_class == 3:
            seg0 = self.Tensor(seg0)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        # seg1 = self.Tensor(seg1)
        # seg2 = self.Tensor(seg2)
        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)

        # seg_label = torch.stack((seg2[0], seg1[0]),0)
        if self.cfg.num_seg_class == 3:
            seg_label = torch.stack((seg0[0],seg1[0],seg2[0]),0)
        else:
            seg_label = torch.stack((seg2[0], seg1[0]),0)
            
        lane_label = torch.stack((lane2[0], lane1[0]),0)
        # _, gt_mask = torch.max(seg_label, 0)
        # _ = show_seg_result(img, gt_mask, idx, 0, save_dir='debug', is_gt=True)
        

        target = [labels_out, seg_label, lane_label]
        img = self.transform(img)

        return img, target, data["image"], shapes

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_det, label_seg, label_lane = [], [], []
        for i, l in enumerate(label):
            l_det, l_seg, l_lane = l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
            label_lane.append(l_lane)
        # torch.cat()对tensors沿指定维度拼接，但返回的Tensor的维数不会变 (2,3)*2 -> (4,3)
        # torch.stack()同样是对tensors沿指定维度拼接，但返回的Tensor会多一维 (2,3)*2 -> (2,2,3)
        return torch.stack(img, 0), [torch.cat(label_det, 0), torch.stack(label_seg, 0), torch.stack(label_lane, 0)], paths, shapes

