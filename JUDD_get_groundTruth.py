# coding:utf-8
"""
Python2

JUDD

Requirs:

conda env: py2_tf
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat


JUDD_ROOT = "/home/lixiang/Desktop/personal-code/dataset/JUDD"
ALL_JUDD_SUBNAMES = ['CNG', 'ajs', 'emb', 'ems', 'ff', 'hp', 'jcw', 'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb']


def get_fix_by_imgName_subName(img_id, sub_name):
    """
    inputs:
        Fix/img_id.mat, which are the JUDD matlab program outputs.
    """
    if img_id.endswith('.jpeg'):
        img_id = img_id[:-5]
    fix_infile = os.path.join(JUDD_ROOT, "Fix", sub_name, img_id+'.mat')
    fix_data = loadmat(fix_infile)
    return fix_data


def fixdata_filter(fix_data):
    """

    """
    position = fix_data['position']
    new_pos = []
    for idx, old_pos in enumerate(position):
        if idx == 0:
            # 从第二个点开始计算
            continue
        if old_pos[0]<0 or old_pos[1]<0:
            # 忽略画幅外的注释点
            continue
        new_pos.append([math.floor(old_pos[0]), math.floor(old_pos[1])])
    fix_data['position'] = np.array(new_pos)
    return fix_data


def scanpath_filter(fix_data, cut_length):
    """
    符合标准则返回对应数据;
    不符合标准则返回None.
    """
    position = fix_data['position']
    if len(position) < cut_length:
        return None
    else:
        position = position[:cut_length]
        fix_data['position'] = position
        return fix_data


def ground_truth_by_imgName_subNames(img_id, sub_names, cut_length, verbose=False):
    if img_id.endswith('.jpeg'):
        img_id = img_id[:-5]

    ### 根据subnames获取多个单subject数据，同时进行fixation_filter
    fix_datas = []
    for sub_name in sub_names:
        fix_data = get_fix_by_imgName_subName(img_id, sub_name)
        # position = fix_data['position'][:max(len(fix_data['position']), 6)]
        fix_data = fixdata_filter(fix_data)
        fix_datas.append(fix_data)

    ### 根据scanpath_filter过滤掉过短等路径
    fix_datas_cut = []
    for fix_data in fix_datas:
        fix_data = scanpath_filter(fix_data, cut_length)
        if fix_data is None:
            continue
        else:
            fix_datas_cut.append(fix_data)

    fix_datas_out = fix_datas_cut
    ### fix_datas_out 的可视化检查
    if verbose:
        img_file = os.path.join(JUDD_ROOT, "ALLSTIMULI", img_id+'.jpeg')
        print(img_file)

        img_data = cv2.imread(img_file) # (height, weight, channel)=(1024, 768, 3)
        print(np.shape(img_data))
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(img_data)

        color_mat = loadmat('cmat.mat')
        color_set = color_mat['cmat']
        for fix_data in fix_datas_out:
            print(fix_data['position'])
            color_tmp = color_set[random.randint(1, 99)]
            x = fix_data['position'][:, 0]
            y = fix_data['position'][:, 1]
            ax1.plot(x, y, color=color_tmp, linewidth=1)
            for i in range(len(fix_data['position'][:, 0])):
                x = fix_data['position'][i, 0]
                y = fix_data['position'][i, 1]
                ax1.text(x, y, s="{}-{}".format(fix_data['sub_name'], i), color=color_tmp, ha = 'center',va = 'bottom',fontsize=7)
        
        plt.show()
    
    return fix_datas_out


if __name__ == "__main__":
    # get_fix_by_imgName_subName(
    #     'ya',
    #     'istatic_submitted_saxena_chung_ng_nips2005_learningdepth_img_math14_p_313t0.jpeg'
    #     )

    img_name = 'i05june05_static_street_boston_p1010764'
    sub_names = ['CNG', 'ajs', 'emb', 'ems', 'ff', 'hp', 'jcw', 'jw', 'kae', 'krl', 'po', 'tmj', 'tu', 'ya', 'zb']
    # sub_names = ['hp']
    cut_length = 5

    ground_truth_by_imgName_subNames(img_name, sub_names, cut_length, verbose=True)