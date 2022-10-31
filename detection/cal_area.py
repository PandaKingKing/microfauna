import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

from otsu_th import otsu_th


def cal_area(src,
             mask_th=10,
             std_th=40,
             otsu_mod=30,
             area_th1=100,
             area_th2=50000,
             ksize1=15,
             ksize2=5,
             ksize3=9,
             hwr_th=2.5,
             hull_th=5,
             vis=False):
    """
    用于计算图像中黑泥部分面积

    :param src: 输入图像
    :param mask_th: 视为圆形边框的阈值
    :param std_th: 图像标准差阈值
    :param otsu_mod: otsu阈值的修正量
    :param area_th1: 轮廓面积阈值1
    :param area_th2: 轮廓面积阈值2
    :param ksize1: 形态学滤波核1
    :param ksize2: 形态学滤波核2
    :param ksize3: 形态学滤波核3
    :param hwr_th: 轮廓宽高比阈值
    :param hull_th: 轮廓凸性缺陷阈值
    :param vis: 标志是否可视化
    :return: 返回计算得到的黑泥面积值
    """
    # pre
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # calculate target mask
    ret, binary = cv2.threshold(blur, np.min(blur) + mask_th, 255, cv2.THRESH_BINARY)
    kernel = np.ones((ksize1, ksize1), 'i1')
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel)
    cnts, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt_area = [cv2.contourArea(cnt) for cnt in cnts]
    aim_cnt = cnts[cnt_area.index(max(cnt_area))]
    mask = np.zeros(binary.shape)
    ellipse = cv2.fitEllipse(aim_cnt)
    cv2.ellipse(mask, ellipse, 1, -1)

    # split with otsu
    bool_mask = np.asarray(mask, 'b1')
    std = np.std(blur[bool_mask])
    # ret_otsu = otsu_th(blur[bool_mask])
    ret_otsu = cv2.threshold(blur[bool_mask], 0, 255, cv2.THRESH_OTSU)[0]
    if std < std_th:
        ret_otsu -= otsu_mod
    ret, th = cv2.threshold(blur, ret_otsu, 255, cv2.THRESH_BINARY_INV)
    th = np.asarray(th * mask, 'u1')
    kernel = np.ones((ksize2, ksize2), 'i1')
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    # remove noise cnt
    contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # good_cnt = []
    res = th.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_th1:
            (x, y), (w, h), theta = cv2.minAreaRect(cnt)
            if (abs(np.log2(h / w)) < hwr_th) and ((area > area_th2) or (h * w / area < hull_th)):
                # good_cnt.append
                continue
        cv2.drawContours(res, [cnt], -1, 0, -1)
    # res = np.zeros(th.shape)
    # cv2.drawContours(res, good_cnt, -1, 1, -1)
    kernel = np.ones((ksize3, ksize3), 'i1')
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

    # calculate target result
    mask_area = np.count_nonzero(mask)
    target_area = np.count_nonzero(res)
    target_rate = target_area / mask_area

    # visualize if needed
    if vis:
        imgs = [src, blur, binary, mask, th, res]
        names = ["origin", "blur", "binary", "mask_fit", "OTSU_thresh", "rm_noise"]
        plt.figure(figsize=(6, 10), dpi=100)
        for i in range(len(imgs)):
            plt.subplot(3, 2, i + 1)
            plt.imshow(imgs[i])
            plt.title(names[i])
            plt.axis('off')
        # plt.subplot(321)
        # plt.imshow(src)
        # plt.title()
        # plt.subplot(322)
        # plt.imshow(blur)
        # plt.title()
        # plt.subplot(323)
        # plt.imshow(binary)
        # plt.title()
        # plt.subplot(324)
        # plt.imshow(mask)
        # plt.title()
        # plt.subplot(325)
        # plt.imshow(th)
        # plt.title()
        # plt.subplot(326)
        # plt.imshow(res)
        # plt.title()
        plt.show()
        print(f"{std=:.2f}")
        print(f"The area of the target: {target_area}")
        print(f"The area of the mask: {mask_area}")
        print(f"The rate of the target: {target_rate:.2f}")

    return target_rate


if __name__ == "__main__":
    sample_path = Path('sample/with_edge')
    # ds_path = Path('traintest/1')
    # img_path_list = list(ds_path.glob('*.jpg'))
    # rand_idx = np.random.permutation(len(img_path_list))
    # # print(img_path_list)
    # for idx in rand_idx:
    #     # rand_idx = np.random.randint(len(pic_list)) if idx < 0 else idx
    #     # print(f"pic_idx={idx}")
    #     img = cv2.imread(str(img_path_list[idx]))
    #     rate = cal_area(img, vis=True)
    #     aa = input('Type the new command: ')
    #     if 's' in aa:
    #         i = 0
    #         while (sample_path / f'sample{i}.jpg').exists(): i += 1
    #         save_path = sample_path / f'sample{i}.jpg'
    #         cv2.imwrite(str(save_path), img)
    #     if 'q' in aa:
    #         break

    for img_path in sample_path.glob('sample[0-9]*.jpg'):
        img = cv2.imread(str(img_path))
        rate = cal_area(img, vis=True)
        # aa = input('Type the new command: ')
        # if 'q' in aa:
        #     break
