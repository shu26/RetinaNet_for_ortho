import cv2
import torch
import numpy as np
from PIL import Image


def adjust_for_ortho(boxes, position, div_num):
    for idx, box in enumerate(boxes):
        tl_x = box[0]
        tl_y = box[1]
        br_x = box[2]
        br_y = box[3]
        # start position from 0 not 1
        adj_x = (position[1] - 1) * 600
        adj_y = (position[0] - 1) * 600

        out_box = torch.tensor([
            tl_x + adj_x,
            tl_y + adj_y,
            br_x + adj_x,
            br_y + adj_y
            ]).unsqueeze(0)

        if idx == 0:
            out_boxes = out_box
        else:
            out_boxes = torch.cat(
                (out_boxes, out_box), 0)

    return out_boxes


def adjust_for_ortho_for_test(boxes, position, div_num):
    for idx, box in enumerate(boxes):
        tl_x = box[0]
        tl_y = box[1]
        br_x = box[2]
        br_y = box[3]
        # start position from 0 not 1
        adj_x = (position[1] - 1) * 480
        adj_y = (position[0] - 1) * 480

        out_box = torch.tensor([
            tl_x + adj_x,
            tl_y + adj_y,
            br_x + adj_x,
            br_y + adj_y
            ]).unsqueeze(0)

        if idx == 0:
            out_boxes = out_box
        else:
            out_boxes = torch.cat(
                (out_boxes, out_box), 0)

    return out_boxes


# this is tmporal implementation term presentation
# so, this function is stricted 3 x 3 ortho image
def unite_images(images, idxs, positions, div_nums):
    div_num_x = div_nums[0][0]
    div_num_y = div_nums[0][1]

    sorted_images = images.copy()
    for i in idxs:
        sorted_images[i] = images[idxs.index(i)]

    horizontal_concatenated_images = []
    default=0
    div_num_y_val = div_num_y #38

    for i in range(div_num_x-1):  #28
        print("i:default:div::::", i,default,div_num_y_val)
        temp_image = np.hstack(sorted_images[default:div_num_y_val]) #38こずつ横に連結
        horizontal_concatenated_images.append(temp_image)
        default+=div_num_y
        div_num_y_val+=div_num_y

    img = np.vstack((horizontal_concatenated_images[0:div_num_x])) #28こ縦に連結

    return img


def unite_images_for_test(images, idxs, positions, div_nums):
    
    div_num_x = div_nums[0][0]  # 34 line
    div_num_y = div_nums[0][1]  # 47 row

    sorted_images = images.copy()
    
    for i in idxs:
        print(i)
        sorted_images[i] = images[idxs.index(i)]

    count=0
    # all_width & all_height are the size used for image2new_img expansion
    # 合成の時はひとまず大きい画像にする，nmsを適用させた後，最終的に出力する際に元の画像サイズに合わせるとかでもいいかも 
    all_width = (div_num_y+1)*480
    all_height = (div_num_x+1)*480

    # make a big image to paste each image.
    new_img = np.zeros((all_height, all_width, 3), np.uint8)
    cv2.rectangle(new_img, (0, 0), (all_width, all_height), (255, 0, 0), -1)

    for i in range(div_num_x):  #34
        for j in range(div_num_y): #47
            print(i,j)
            w,h,_ = sorted_images[count].shape
            new_img[480*i:480*i+w:, 480*j:480*j+h] = sorted_images[count] # [y, x]
            count+=1
        
    return new_img
