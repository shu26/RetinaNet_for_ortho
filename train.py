import os
import sys
import cv2
import copy
import collections
import numpy as np
from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms

from modules.anchors import Anchors
from modules import losses
from modules.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer
from modules import coco_eval
from modules import csv_eval
from modules.nms_pytorch import NMS
from modules.utils import BBoxTransform, ClipBoxes, AverageMeter
from visualize import main as visualize

import model

#assert torch.__version__.split('.')[1] == '4'

print('CUDA available: {}'.format(torch.cuda.is_available()))


class Trainer:
    def __init__(self):
        # Simple training script for training a RetinaNet network.

        # Dataset type, must be one of csv or coco.
        self.dataset = 'csv'

        # Path to COCO directory
        self.coco_path = './data'

        # Path to file containing training annotations (see readme)
        self.csv_train ='./csv_data/kudeken_makiya/annotations/annotation.csv'

        # Path to file containing class list (see readme)
        self.csv_classes = './csv_data/kudeken_makiya/annotations/pet_class_id.csv'

        # Path to file containing validation annotations (optional, see readme)
        self.csv_val = './csv_data/kudeken_makiya/annotations/annotation.csv'

        # Resnet depth, must be one of 18, 34, 50, 101, 152
        self.depth = 50

        # batch_size
        self.bs = 6

        # learning rate
        self.lr = 6e-5

        # Number of epochs
        self.epochs = 600

        # Number of save epochs
        #self.save_freq = 5

        # set device
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        # set focal loss
        self.focal_loss = losses.FocalLoss()

        # module calcurating nms
        self.nms = NMS(BBoxTransform, ClipBoxes)

        # index of the saving model
        self.save_name = 0

        # use comet_ml
        self.cml = True

        # classification_loss
        self.cls_loss_meter = AverageMeter()

        # regression_loss
        self.rgrs_loss_meter = AverageMeter()

        self.set_comet_ml()

        self.unnormalize = UnNormalizer()
    
    def set_comet_ml(self):
        params = {
        'epochs': self.epochs,
        'batch_size': self.bs,
        'lr': self.lr,
        'resnet_depth': self.depth,
        'save_name': self.save_name,
        }

        if self.cml:
            self.experiment = Experiment(api_key="XgC28yk6LIhqha2yicN0vohwm",
                        project_name="general", workspace="shu26")
            print("send comet_ml")
            self.experiment.log_parameters(params)
        else:
            self.experiment = None

    def set_dataset(self):
        # Create the data loaders
        if self.dataset == 'coco':
            if self.coco_path is None:
                raise ValueError('Must provide --coco_path when training on COCO,')
            dataset_train = CocoDataset(self.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
            dataset_val = CocoDataset(self.coco_path, set_name='val2017', transform=transforms.Compose([Normalizer(), Resizer()]))

        elif self.dataset == 'csv':
            if self.csv_train is None:
                raise ValueError('Must provide --csv_train when training on COCO,')
            if self.csv_classes is None:
                raise ValueError('Must provide --csv_classes when training on COCO,')
            dataset_train = CSVDataset(train_file=self.csv_train, class_list=self.csv_classes, transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
            if self.csv_val is None:
                dataset_val = None
                print('No validation annotations provided.')
            else:
                dataset_val = CSVDataset(train_file=self.csv_val, class_list=self.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))

        else:
            raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

        return dataset_train, dataset_val


    def set_models(self, dataset_train):
        # Create the model
        if self.depth == 18:
            retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
        elif self.depth == 34:
            retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
        elif self.depth == 50:
            retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
        elif self.depth == 101:
            retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
        elif self.depth == 152:
            retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

        self.retinanet = retinanet.to(self.device)
        self.retinanet.training = True
        self.optimizer = optim.Adam(self.retinanet.parameters(), lr=self.lr)
        # This lr_shceduler reduce the learning rate based on the models's validation loss
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)
        self.loss_hist = collections.deque(maxlen=500)

    
    def iterate(self):
        print('GPU:{} is used'.format(self.device))
        dataset_train, dataset_val = self.set_dataset()
        sampler = AspectRatioBasedSampler(dataset_train, batch_size=self.bs, drop_last=False)
        dataloader_train = DataLoader(dataset_train, num_workers=0, collate_fn=collater, batch_sampler=sampler)

        print('Num training images: {}'.format(len(dataset_train)))

        self.set_models(dataset_train)

        for epoch_num in range(self.epochs):
            epoch_loss = []

            metrics = {
                    'classification_loss': self.cls_loss_meter.avg,
                    'regression_loss': self.rgrs_loss_meter.avg,
                    'entire_loss': self.cls_loss_meter.avg + self.rgrs_loss_meter.avg
                    }

            if self.experiment is not None:
                self.experiment.log_metrics(metrics, step=epoch_num)
            self.retinanet.train()
            self.retinanet.freeze_bn()
            epoch_loss = self.train(epoch_num, epoch_loss, dataloader_train)
            self.retinanet.eval()
            # 評価
            #self.evaluate(epoch_num, dataset_val)

            # save the model & visualize the image
            self.scheduler.step(np.mean(epoch_loss))	
            self.retinanet.eval()
            if (epoch_num+1) % 20 == 0:
                self.evaluate(epoch_num, dataset_val)

            if (epoch_num+1) % 100 == 0:# or epoch_num == 10:
                #self.evaluate(epoch_num, dataset_val)
                model_path = os.path.join('./saved_models/kudeken_makiya/', 'pet_model_{}epochs.pth'.format(epoch_num))
                torch.save(self.retinanet.state_dict(), model_path)
            
            if (epoch_num+1) % 10000 == 0:
                visualize(model_path, epoch_num)
                #self.experiment.log_image(image_data=vis_img)


        #torch.save(self.retinanet.state_dict(), './saved_models/model_anchi_0522_1000epochs_pet.pth')


    def train(self, epoch_num, epoch_loss, dataloader_train):
        for iter_num, data in enumerate(dataloader_train):
            try:

                self.optimizer.zero_grad()
                input = data['img'].to(self.device).float()
                annot = data['annot'].to(self.device)

                # unnormalization
                img = np.array(255 * self.unnormalize(data['img'][0, :, :, :])).copy()
                img = img[:,:600,:600]
                img[img<0] = 0
                img[img>255] = 255
                img = np.transpose(img, (1, 2, 0))
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

                img1d = np.sum(img, axis=-1) # sum rgb values -> 0 means that the color is black
                imgblack = np.where(img1d==0, 1, 0)
                black_count = np.sum(imgblack)  # calcurate the number of the black image
                ratio = black_count / (img1d.shape[0] * img1d.shape[1])

                # When about 80% of the image is dark, do not process it
                if ratio >= 0.8:
                    continue

                regression, classification, anchors = self.retinanet(input)
                
                classification_loss, regression_loss = self.focal_loss.calcurate(classification, regression, anchors, annot)
                
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                self.cls_loss_meter.update(classification_loss)
                self.rgrs_loss_meter.update(regression_loss)
                loss = classification_loss + regression_loss
                if bool(loss == 0):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.retinanet.parameters(), 0.1)
                self.optimizer.step()
                self.loss_hist.append(float(loss.item()))
                epoch_loss.append(float(loss.item()))
                torch.nn.utils.clip_grad_norm_(self.retinanet.parameters(), 0.1)
                print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(self.loss_hist)))
                
                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

            # if iter_num == 10:
            #     break

        return epoch_loss


    def evaluate(self, epoch_num, dataset_val):
        print('------------------------------------')
        if self.dataset == 'coco':

            print('Evaluating dataset coco')

            coco_eval.evaluate_coco(dataset_val, self.retinanet, self.nms, self.device)

        elif self.dataset == 'csv' and self.csv_val is not None:

            print('Evaluating dataset csv')
            
            recall, precision, mAP = csv_eval.evaluate(dataset_val, self.retinanet, self.nms, self.device)

            metrics = {
                    'precision': precision,
                    'recall': recall,
                    'mAP': mAP[0][1]
                    }

            self.experiment.log_metrics(metrics, step=epoch_num)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.iterate()
