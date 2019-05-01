import torch
import numpy as np


def adjust_for_ortho(boxes, position, div_num):
    for idx, box in enumerate(boxes):
        tl_x = box[0]
        tl_y = box[1]
        br_x = box[2]
        br_y = box[3]
        # start position from 0 not 1
        # TODO: Edit for test code: 4/8 kaiho
        if idx == 0:
            adj_x = (position[1] - 1) * 600 #-11
            adj_y = (position[0] - 1) * 600 #-8
        else:
            adj_x = (position[1] - 1) * 480 + 600#-11
            adj_y = (position[0] - 1) * 480 + 600 #-8

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


def unite_images(images, idxs, positions, div_nums):
    # all element in div_nums(list) is same.
    positions_x = positions[0]
    positions_y = positions[1]
    div_num_x = div_nums[0]
    div_num_y = div_nums[1]

    print("----------")
    print(idxs)
    print("----------")
    print(positions)
    print("----------")
    print(div_nums)
    print("----------")

    sorted_images = images.copy()
    for i in idxs:
        print(i)
        sorted_images[i] = images[idxs.index(i)]

    # 28x38
    # 横に38こ連結を縦に28回繰り返す
    # 横 hstack = (concat axis=1)
    '''
    positions_xはxの値ではなく，ある要素のpositionのxとy両方が含まれている
    '''
    
    horizontal_concatenated_images = []
    default=0
    div_num_y_val = div_num_y[1] #38

    print("==============================================")
    print(div_num_y_val)
    print("==============================================")

    for i in range(div_num_y[0]-1):  #28
        print("i:default:div::::", i,default,div_num_y_val)
        temp_image = np.hstack(sorted_images[default:div_num_y_val]) #38こずつ横に連結
        horizontal_concatenated_images.append(temp_image)
        default+=div_num_y[1]
        div_num_y_val+=div_num_y[1]

    img = np.vstack((horizontal_concatenated_images[0:div_num_y[0]])) #28こ縦に連結

    return img

