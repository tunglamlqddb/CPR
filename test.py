from multiprocessing.connection import wait
import os, sys, glob
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from torch import Tensor
from PIL import Image

def split_cub200_loader(root, seg_root, seg, cropped, type):
    
    data = {}
    for i in range(10):
        data[i] = {}
        data[i]['name'] = 'split_cub200-{:d}'.format(i)
        data[i]['ncla'] = 20
        data[i]['train'] = {'x': [], 'y': []}
        data[i]['test'] = {'x': [], 'y': []}
        data[i]['valid'] = {'x': [], 'y': []}

    folders = sorted(os.listdir(root+'/images'))
    seg_folders = sorted(os.listdir(seg_root))

    with open(root+'/bounding_boxes.txt', 'r') as f:
        bounding_boxes = f.read().splitlines()
        bounding_boxes = [item.split() for item in bounding_boxes]
        bounding_boxes = [[int(float(i)) for i in item] for item in bounding_boxes]
    
    mean = np.array([[[123.77, 127.55, 110.25]]])
    std = np.array([[[59.16, 58.06, 67.99]]])
    
    idx = -1
    label_true = -1
    
    for folder in folders:
        
        folder_path = os.path.join(root+'/images', folder)
        seg_folder_path = os.path.join(seg_root, folder)
        img_list = sorted(os.listdir(folder_path))
        seg_list = sorted(os.listdir(seg_folder_path))
        label_true += 1
        
        img_len = len(img_list)
        tr_num = int(len(img_list) * 0.7)
        val_num = int(len(img_list) * 0.1)
        te_num = int(len(img_list) * 0.2)
        
        folder_img_idx = 0
        
        for (ims, seg) in zip(img_list, seg_list):
            idx += 1
            img_path = os.path.join(folder_path, ims)
            img = plt.imread(img_path)
            try:
                H,W,C = img.shape
                
                if not cropped and H < 224 or W < 224:
                    continue
            except:
                continue
            
            if seg:
                seg_path = os.path.join(seg_folder_path, seg)
                seg = plt.imread(seg_path).astype(int)
                seg = np.expand_dims(seg, 2)
                img = img*seg
                plt.imshow(img)
                plt.show()
            if idx>10: quit()

            if cropped:
                x,y,w,h = bounding_boxes[idx][1], bounding_boxes[idx][2], bounding_boxes[idx][3], bounding_boxes[idx][4]
                print(w<=W and h<=H)
                img = img[y:y+h, x:x+w]
                plt.imshow(img)
                plt.show()
                if h < 224 or w < 224:
                    if type=='padding':
                        pass
                        # create new image of desired size and color (blue) for padding
                        # new_image_width = 300
                        # new_image_height = 300
                        # color = (255,0,0)
                        # result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

                        # # compute center offset
                        # x_center = (new_image_width - old_image_width) // 2
                        # y_center = (new_image_height - old_image_height) // 2

                        # # copy img image into center of result image
                        # result[y_center:y_center+old_image_height, 
                        #     x_center:x_center+old_image_width] = img
                # if type=='resize':
                #     img = Image.fromarray(img)
                #     img.resize(size=(224, 224))
                #     img = np.array(img)
                #     
                #     print(img.size)
                #     plt.imshow(img)
                #     plt.show()
                if idx > 10: quit() 


            if folder_img_idx < tr_num:
                s = 'train'
                print ('train', img.shape)
            elif folder_img_idx < tr_num + val_num:
                s = 'valid'
                print ('valid', img.shape)
            else:
                s = 'test'
                print ('test', img.shape)
                
            img = (img - mean) / std
            img_tensor = Tensor(img).float()
            task_idx = label_true // 20
            label = label_true % 20
            data[task_idx][s]['x'].append(img_tensor)
            data[task_idx][s]['y'].append(label) 
            
            folder_img_idx += 1
    
    return data

split_cub200_loader(root='../data/CUB200(2011)/CUB_200_2011/CUB_200_2011', seg_root='../data/CUB200(2011)/segmentations', seg=True, cropped=False, type='resize')