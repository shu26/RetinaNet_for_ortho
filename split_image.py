import cv2,matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

def main():
    size = 600
    count = 0
    v_count = 0
    h_count = 0

    img_default = tifffile.imread("ortho_image/ortho_anchi.tif")
    img = cv2.cvtColor(img_default, cv2.COLOR_BGR2RGB)
    v_size = img.shape[0]
    h_size = img.shape[1]
    v_size_clopped = img.shape[0] // size * size
    h_size_clopped = img.shape[1] // size * size
    
    img_clopped = img[:v_size_clopped, :h_size_clopped]
    img_right = img[:v_size_clopped, h_size_clopped:h_size]
    img_bottom = img[v_size_clopped:v_size, :h_size_clopped]
    img_fraction = img[v_size_clopped:v_size, h_size_clopped:h_size]
    
    v_split = img_clopped.shape[0] // size
    h_split = img_clopped.shape[1] // size
    
    out_imgs = []
    out_places = []
    out_imgs_right=[]
    # 右部分の端数を縦に分割
    for out_img_right in np.vsplit(img_right, v_split):
        out_imgs_right.append(out_img_right)
    # 垂直方向に分割する。 -> 行
    for h_img in np.vsplit(img_clopped, v_split):
        h_count=0
        v_count+=1
        # 水平方向に分割する。 -> 列
        for v_img in np.hsplit(h_img, h_split):
            h_count+=1
            out_imgs.append(v_img)
            out_place = "{0}x{1}".format(v_count,h_count)
            out_places.append(out_place)
        # 各行の最後に端数の部分を追加（画像の右部）
        if v_size != v_size_clopped:
            h_count+=1
            out_img_right = out_imgs_right[v_count-1]
            out_imgs.append(out_img_right)
            out_place = "{0}x{1}".format(v_count,h_count)
            out_places.append(out_place)
    # 各列の最後に端数の部分を追加（画像の下部）
    if h_size != h_size_clopped:
        h_count=0
        v_count+=1
        for v_img in np.hsplit(img_bottom, h_split):
            h_count+=1
            out_imgs.append(v_img)
            out_place = "{0}x{1}".format(v_count,h_count)
            out_places.append(out_place)
        # 行にも端数があれば追加（画像の右下）
        if v_size != v_size_clopped:
            h_count+=1
            out_imgs.append(img_fraction)
            out_place = "{0}x{1}".format(v_count,h_count)
            out_places.append(out_place)
    # 最終的に何個に分割したかカウント
    split_count = "{0}x{1}".format(v_count,h_count)
    out_imgs = np.array(out_imgs)
    # 出力する
    for i in out_imgs:
        out_place = out_places[count]
        file_name = "./anchi/images/{0}_{1}_{2}.png".format(count,out_place,split_count)
        cv2.imwrite(file_name, i)
        count+=1


if __name__ == "__main__":
    main()
