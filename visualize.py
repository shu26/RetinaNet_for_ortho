import numpy as np
import torchvision
import os
import copy

import sys
import cv2
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from modules.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer, Grayscale
from model import resnet50
from modules.nms_pytorch import NMS
from modules.anchors import Anchors
from modules.utils import BBoxTransform, ClipBoxes
from modules.ortho_util import adjust_for_ortho, adjust_for_ortho_for_vis, adjust_for_ortho_for_test, adjust_for_ortho_for_vis_for_test, unite_images, unite_images_for_test
from modules import losses
from modules import csv_eval

print('CUDA available: {}'.format(torch.cuda.is_available()))


#def main(args=None):
def main(model_path, epoch_num):
    
    def evaluate(epoch_num, dataset_val, retinanet, nms, device):
        print('-------------------------------------')
        if params["dataset"] == 'csv' and params["csv_val"] is not None:

            print('Evaluating dataset csv')

            recall, precision, mAP = csv_eval.evaluate(dataset_val, retinanet, nms, device)
            metrics = {
                    'precision': precision,
                    'recall': recall,
                    'mAP': mAP[0][0]
                    }

            print("precision: ", precision)
            print("recall: ", recall)
            print("mAP: ", mAP[0][0])

            #self.experiment.log_metrics(metrics, step=epoch_num)


    params = {
            'dataset': 'csv',
            'coco_path': '',
            'csv_classes': './csv_data/split_dataset/makiya/annotations/pet_class_id.csv',
            'csv_val': './csv_data/split_dataset/kudeken/annotations/only_pet_annotation.csv',
            'model': model_path,
            'num_class': 1,
            'prediction': True,
            'test': False,
            }
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    print(":::::::::::::::::")
    print("Now visualizing...")
    print(device)
    print(torch.cuda.get_device_name(1))
    print(":::::::::::::::::")


    if params['dataset'] == 'coco':
        dataset_val = CocoDataset(params['coco_path'], set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif params['dataset'] == 'csv':
        dataset_val = CSVDataset(train_file=params['csv_val'], class_list=params['csv_classes'], transform=transforms.Compose([Normalizer(), Grayscale(), Resizer()]))
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
    #TODO:
    evaluate(epoch_num, dataset_val, retinanet, nms, device)
    sys.exit(0)

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

            input = data['img'].to(device).float()
            regression, classification, anchors = retinanet(input)
            scores, labels, boxes = nms.calc_from_retinanet_output(
                input, regression, classification, anchors)
            data['p_idx'] = data['p_idx'][0]
            data['position'] = data['position'][0]
            data['div_num'] = data['div_num'][0]

            if boxes.shape[0] != 0:
                global adjusted_boxes
                if params['test'] == False:
                    adjusted_boxes = adjust_for_ortho(boxes, data['position'], data['div_num'])
                else:
                    adjusted_boxes = adjust_for_ortho_for_test(boxes, data['position'], data['div_num'])
                
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
            print("idx", idx, end='\r')
            #cv2.imwrite('imgs/test_img{}.png'.format(idx), img)

        idxs = np.array(p_idxs)

        # ----------------------------------------
        # unite image parts
        global ortho_img
        if params['test'] == False:
            print("images_list: ", len(images_list))
            print("p_idxs: ", len(p_idxs))
            ortho_img = unite_images(images_list, p_idxs, positions, div_nums)
        else:
            ortho_img = unite_images_for_test(images_list, p_idxs, positions, div_nums)
        # ----------------------------------------

        #print(scores_list)

        # if scores and labels is torch tensor
        scores_list = torch.cat(tuple(scores_list), 0).cpu()
        labels_list = torch.cat(tuple(labels_list), 0).cpu()
        boxes_list = torch.cat(tuple(boxes_list), 0).cpu()
        
        # ----------------------------------------
        # apply nms calcuraiton to entire bboxes
        entire_scores, entire_labels, entire_boxes = nms.entire_nms(scores_list, labels_list, boxes_list)
        # ----------------------------------------

        idxs = np.where(entire_scores>0.5)
         
        if params['prediction'] == False:
        
            vis_label = []
            vis_idxs = []
            vis_bbox = []
            vis_pos = []
            vis_div = []
            vis_adjust = []
        
            with open(params['csv_val'], 'r') as f:
                reader = csv.reader(f)
                idx = 0
                for row in reader:
                    print(row)
                    # When there are some boxes, do below
                    if row[1] != '': 
                        path = row[0]
                        img_name = path.split("/")[4] #change idx 4 to 5 if you use small dataset
                        #img_idx = img_name.split("_")[0]
                        img_pos = img_name.split("_")[1]
                        img_div = img_name.split("_")[2].split(".")[0]

                        pos_x = int(img_pos.split("x")[0].replace("'", " "))
                        pos_y = int(img_pos.split("x")[1].replace("'", " "))
                        div_x = int(img_div.split("x")[0].replace("'", " "))
                        div_y = int(img_div.split("x")[1].replace("'", " "))
                    
                        pos = [pos_x, pos_y]
                        div = [div_x, div_y]
                    
                        x1 = float(row[1].replace("'", " "))
                        x2 = float(row[2].replace("'", " "))
                        y1 = float(row[3].replace("'", " "))
                        y2 = float(row[4].replace("'", " "))
                        label = row[5]
                        bbox = [x1, x2, y1, y2]
                        if label == "tree":
                            label = 1
                        elif label == "rope":
                            label = 2
                        elif label == "plasticbottle":
                            label = 0
                        elif label == "net":
                            labee = 3
                        elif label == "spraycan":
                            label = 4
                        elif label == "bucket":
                            label = 5
                        elif label == "buoy":
                            label = 6

        
                        vis_bbox.append(bbox)
                        vis_label.append(label)
                        vis_idxs.append(idx)
                        vis_pos.append(pos)
                        vis_div.append(div)
                        idx+=1
            for i, bbox in enumerate(vis_bbox):
                adjusted_boxes = adjust_for_ortho_for_vis(bbox, vis_pos[i], vis_div[0])
                #global adjusted_boxes
                #if params['test'] == False:
                #    adjusted_boxes = adjust_for_ortho(bbox, vis_pos[i], vis_div[0])
                #else:
                #    adjusted_boxes = adjust_for_ortho_for_test(bbox, vis_pos[i], vis_div[0])
                #vis_adjust.append(adjusted_boxes.to(torch.float).to(device))
                vis_adjust.append(adjusted_boxes)
            vis_adjust = torch.tensor(vis_adjust)
            #vis_adjust = torch.cat(tuple(vis_adjust), 0).cpu()
            idxs = (np.array(vis_idxs),)
            entire_boxes = vis_adjust
            entire_labels = torch.tensor(vis_label)

        for j in range(idxs[0].shape[0]):
            bbox = entire_boxes[idxs[0][j], :]  # Tensor
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            label_name = dataset_val.labels[int(entire_labels[idxs[0][j]])]
            draw_caption(ortho_img, (x1, y1, x2, y2), label_name)
            if label_name == "tree":
                cv2.rectangle(ortho_img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            if label_name == "rope":
                cv2.rectangle(ortho_img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            if label_name == "plasticbottle":
                cv2.rectangle(ortho_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            if label_name == "net":
                cv2.rectangle(ortho_img, (x1, y1), (x2, y2), color=(100, 100, 0), thickness=2)
            if label_name == "spraycan":
                cv2.rectangle(ortho_img, (x1, y1), (x2, y2), color=(0, 100, 100), thickness=2)
            if label_name == "bucket":
                cv2.rectangle(ortho_img, (x1, y1), (x2, y2), color=(100, 0, 100), thickness=2)
            if label_name == "buoy":
                cv2.rectangle(ortho_img, (x1, y1), (x2, y2), color=(50, 100, 200), thickness=2)
        #dataset_val = set_dataset()
        #evaluate(epoch_num, dataset_val)

        print("Now saving...")
        global ortho_img
        if params["test"] == True:
            # clip original size from ortho_img made in RetinaNet
            ortho_img = ortho_img[0:9704,0:11522] 
        ortho_img = ortho_img[0:7298,0:10938] 
        cv2.imwrite('./visualized_images/split_dataset/kudeken/graysacle_vis_{}epochs.png'.format(epoch_num), ortho_img)
        #cv2.imwrite('./visualized_images/vis_test_1016.png', ortho_img)
        print("Finish saving")
        #cv2.waitKey(0)


        def set_dataset():
            dataset_val = CSVDataset(train_file=params["csv_val"], class_list=paranms["csv_classes"], transform=transforms.Compose([Normalizer(), Resizer()]))
            return dataset_val

    #def evaluate(epoch_num, dataset_val, retinanet, nms, device):
    #    print('-------------------------------------')
    #    if self.dataset == 'csv' and self.csv_val is not None:

    #        print('Evaluating dataset csv')

    #        recall, precision, mAP = csv_eval.evaluate(dataset_val, retinanet, nms, device)
    #        metrics = {
    #                'precision': precision,
    #                'recall': recall,
    #                'mAP': mAP[0][0]
    #                }

    #        print("precision: ", precision)
    #        print("recall: ", recall)
    #        print("mAP: ", mAP[0][0])

    #        self.experiment.log_metrics(metrics, step=epoch_num)


if __name__ == '__main__':
    main("./saved_models/split_dataset/makiya/grayscale_gamma_model_199epochs.pth", 200)
