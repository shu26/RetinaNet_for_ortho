import cv2,matplotlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tifffile

import sklearn
from sklearn.cluster import KMeans

def main():
    size = 600
    slide_size = int(size * 0.8)
    count = 0
    v_count = 1
    h_count = 1

    img_default = tifffile.imread("../../Desktop/Orthor_experiment/original_dataset/1004_Pix4d/20m.tif")
    img = cv2.cvtColor(img_default, cv2.COLOR_BGR2RGB)

    # 元画像のサイズ
    v_size = img.shape[0]
    h_size = img.shape[1]
    # 元の画像をsizeで割り切れる値
    v_size_clopped = img.shape[0] // size * size
    h_size_clopped = img.shape[1] // size * size

    img_clopped = img[:v_size_clopped, :h_size_clopped]   # sizeで割り切れるところまでの値
    img_right = img[:v_size_clopped, h_size_clopped:h_size]   # 右端
    img_bottom = img[v_size_clopped:v_size, :h_size_clopped]    # 下端
    img_fraction = img[v_size_clopped:v_size, h_size_clopped:h_size]    # 右下

    # sizeでの分割数
    v_split = img_clopped.shape[0] // size
    h_split = img_clopped.shape[1] // size
    # slide_sizeでのスライド数
    v_slide_count = (img_clopped.shape[0]) // slide_size + 1
    h_slide_count = (img_clopped.shape[1]) // slide_size + 1

    out_imgs = []
    out_places = []
    out_imgs_right=[]

    for i in range(int(v_slide_count)):
        v_count+=1
        h_count=1
        for j in range(int(h_slide_count)):
            h_count+=1
            im_crop = img[slide_size * i : slide_size * i + size,
                          slide_size * j : slide_size * j + size]
            out_imgs.append(im_crop)
            out_place = "{0}x{1}".format(i,j)
            out_places.append(out_place)

    # 最終的に何個に分割したかカウント
    split_count = "{0}x{1}".format(v_count,h_count)
    out_imgs = np.array(out_imgs)

    # 出力する
    for i in out_imgs:
        out_place = out_places[count]
        """
        出力の前に，各画像で黒い部分が100%のものを端数としてテストには使用しないようにする
        そのため，画像サイズごとで場合分けをしたあと，
        色を検出し，黒かそうでないか分類して保存する
        """

        # 画像のテンソルを（縦と横にフラット化させた値，カラーチャンネル）の２次元に変換
        #img_reshaped = i.reshape((i.shape[0] * i.shape[1], 3))
        # クラスタリング
        #cluster = KMeans(n_clusters=1)
        #cluster.fit(X=img_reshaped)
        #cluster_centers_arr = cluster.cluster_centers_.astype(int,copy=False)[0]
        # 黒の画像のみ別のディレクトリに保存
        #black = np.array([0,0,0])
        #if (cluster_centers_arr==black).all():
        #  file_name = "./anchi/images_for_test_0406/black/{0}_{1}_{2}.png".format(count,out_place,split_count)
        #else:
        file_name = "../../Desktop/Orthor_experiment/test_dataset/komesu/images/{0}_{1}_{2}.png".format(count,out_place,split_count)
        cv2.imwrite(file_name, i)
        count+=1

if __name__ == "__main__":
    main()
