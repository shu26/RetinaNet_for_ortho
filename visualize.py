import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from modules.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from model import resnet50
from modules.nms_pytorch import NMS
from modules.anchors import Anchors
from modules.utils import BBoxTransform, ClipBoxes
from modules.ortho_util import adjust_for_ortho, unite_images
from modules import losses

assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    # parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    # parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='coco')
    # parser.add_argument('--coco_path', help='Path to COCO directory', default='./data')
    # parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    # parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    # parser.add_argument('--model', help='Path to model (.pt) file.', default='./coco_resnet_50_map_0_335.pt')

    # parser = parser.parse_args(args)
    params = {
            'dataset': 'csv',
            'coco_path': '',
            'csv_classes': './csv_data/anchi/annotations/class_id.csv',
            'csv_val': './csv_data/anchi/annotations/annotation.csv',
            'model': './saved_models/model_anchi_0518_100epochs.pth',
            'num_class': 3
            }
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print(":::::::::::::::::")
    print(device)
    print(torch.cuda.get_device_name(1))
    print(":::::::::::::::::")


    if params['dataset'] == 'coco':
        dataset_val = CocoDataset(params['coco_path'], set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif params['dataset'] == 'csv':
        dataset_val = CSVDataset(train_file=params['csv_val'], class_list=params['csv_classes'], transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    nms = NMS(BBoxTransform, ClipBoxes)

    retinanet = resnet50(num_classes=params['num_class'], pretrained=True)
    retinanet.load_state_dict(torch.load(params['model']), strict=False)
    retinanet.eval()

    retinanet = retinanet.to(device)

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


    scores_list = []
    labels_list = []
    boxes_list = []
    images_list = []
    p_idxs = []
    positions = []
    div_nums = []
    with torch.no_grad():
        for idx, data in enumerate(dataloader_val):
            # ここのどこかで画像に対して黒を判別してreturnする処理をかける
            # RGBを抽出
            # img1d = np.sum(img,axis=-1)
            # 黒を抽出
            # imgblack = np.where(img1d=0,1,0)
            # 黒の部分を計算
            # sum = np.sum(img1d=1の部分)
            # 画像に対する黒の割合を計算
            # ratio = sum / (h * w) 

            #st = time.time()
            # scores, classification, transformed_anchors = retinanet(data['img'].to(device).float())
            print("device:", device)
            input = data['img'].to(device).float()
            regression, classification, anchors = retinanet(input)
            scores, labels, boxes = nms.calc_from_retinanet_output(
                    input, regression, classification, anchors)

            data['p_idx'] = data['p_idx'][0]
            data['position'] = data['position'][0]
            data['div_num'] = data['div_num'][0]
            if boxes.shape[0] != 0:
                adjusted_boxes = adjust_for_ortho(boxes, data['position'], data['div_num'])
                scores_list.append(scores.to(torch.float).to(device))
                labels_list.append(labels.to(torch.long).to(device))
                boxes_list.append(adjusted_boxes.to(torch.float).to(device))

            p_idxs.append(data['p_idx'])
            positions.append(data['position'])
            div_nums.append(data['div_num'])

            # image denomalization
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
            img = img[:,:600,:600]
            #print(img.shape)
            img[img<0] = 0
            img[img>255] = 255
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            images_list.append(img)
            print("idx", idx)
            #cv2.imwrite('imgs/test_img{}.png'.format(idx), img)

        # if scores and labels is torch tensor
        scores_list = torch.cat(tuple(scores_list), 0).cpu()
        labels_list = torch.cat(tuple(labels_list), 0).cpu()
        boxes_list = torch.cat(tuple(boxes_list), 0).cpu()

        # ----------------------------------------
        # apply nms calcuraiton to entire bboxes
        entire_scores, entire_labels, entire_boxes = nms.entire_nms(scores_list, labels_list, boxes_list)
        # ----------------------------------------


        #
        # テスト終了後にmAPスコアの評価を行う必要あり
        #


        # ----------------------------------------
        # unite image parts
        ortho_img = unite_images(images_list, p_idxs, positions, div_nums)
        # ----------------------------------------

        #print('Elapsed time: {}'.format(time.time()-st))

        print(boxes.shape)
        idxs = np.where(entire_scores>0.5)
        for j in range(idxs[0].shape[0]):
            bbox = entire_boxes[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = dataset_val.labels[int(entire_labels[idxs[0][j]])]
            draw_caption(ortho_img, (x1, y1, x2, y2), label_name)

            cv2.rectangle(ortho_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

        cv2.imwrite('vis.png', ortho_img)
        cv2.waitKey(0)



if __name__ == '__main__':
    main()
