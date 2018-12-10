# coding:utf-8
"""
python version: 2 / 3
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import random
import sys
import copy
import math
import pickle

import pandas as pd
import numpy as np
from PIL import Image
import scipy.io as sio
import matplotlib.pyplot as plt


VERBOSE = False
def eprint(input_str):
    if VERBOSE:
        print(input_str, file=sys.stderr)


class Trans_tools(object):

    def __init__(self):
        """
        x / col / weight
        y / row / height
        """
        pass

    def trans_rcIdx2scanIdx(self, row_idx, col_idx, row_num=14):
        """
        trans:
            row_idx * 14 + col_idx -> scan_idx
        example of input：
            output = input
            [29] = [2][1]
            [59] = [4][3]
        """
        scan_idx = row_idx * row_num + col_idx
        return scan_idx

    def trans_scanIdx2rcIdx(self, scan_idx, row_num=14):
        """
        trans: 
            scan_idx -> row_idx * 14 + col_idx
        example of input：
            output = input
            [2][1] = [29]
            [4][3] = [59]
        """
        row_idx = scan_idx // row_num
        col_idx = scan_idx % row_num
        return row_idx, col_idx

    def trans_rcIdx2pos(self, row_idx, col_idx, img_height=224, img_weight=224, row_num=14, col_num=14):
        """
        inputs: 
            row_idx, col_idx
        return: 
            the center pos of block ( row_pos, col_pos )
        """
        block_height = img_height // row_num
        block_weight = img_weight // col_num
        row_pos = row_idx * block_height + block_height // 2
        col_pos = col_idx * block_weight + block_weight // 2
        return row_pos, col_pos
    
    def trans_scanIdx2rcIdx_seq(self, scan_idx_seq, row_num=14):
        """
        trans: 
            scan_idx -> row_idx * 14 + col_idx
        example of input：
            output = input
            [[2][1], [4][3]] = [29, 59]
        """
        res = []
        for scan_idx in scan_idx_seq:
            row_idx, col_idx = self.trans_scanIdx2rcIdx(scan_idx)
            res.append([row_idx, col_idx])
        return res


class Show_tools(object):

    def __init__(self):
        pass

    def show_scanpath_compare(self, img, scan_pos=None, scan_idxs=None, title_str=None, save_path=None):
        """
        inputs:
            img: img_file or img_data
            scan_pos: list[[pos_x, pos_y], [pos_x, pos_y], ...]
            scan_idxs: list[idx1, idx2, idx3]
        """
        assert (scan_pos is not None) or (scan_idxs is not None)

        IMG_HEIGHT = 480
        IMG_WEIGHT = 640

        if isinstance(img, str):
            img = Image.open(img)
        fig = plt.figure()

        if title_str is not None:
            plt.title(title_str)

        ax1 = fig.add_subplot(121)
        ax1.imshow(img)
        ax2 = fig.add_subplot(122)
        ax2.imshow(img)

        line_color = random.randint(1, 99)

        if scan_pos:
            x1 = []
            y1 = []
            for i, scan_pos in enumerate(scan_pos):
                col_pos, row_pos = scan_pos
                x1.append(col_pos)
                y1.append(row_pos)
                ax1.text(col_pos, row_pos, "{}".format(i+1), color='r')
            color_mat = sio.loadmat("cmat.mat")
            color_set = color_mat['cmat']
            ax1.plot(x1, y1, color=color_set[line_color], linewidth=1)

        trans_tools = Trans_tools()
        if scan_idxs:
            x2 = []
            y2 = []
            for i, scan_idx in enumerate(scan_idxs):
                row_idx, col_idx = trans_tools.trans_scanIdx2rcIdx(scan_idx)
                row_pos, col_pos = trans_tools.trans_rcIdx2pos(row_idx, col_idx, IMG_HEIGHT, IMG_WEIGHT)
                x2.append(col_pos)
                y2.append(row_pos)
                ax2.text(col_pos, row_pos, "{}".format(i+1), color='r')
            color_mat = sio.loadmat("cmat.mat")
            color_set = color_mat['cmat']
            ax2.plot(x2, y2, color=color_set[line_color], linewidth=1)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    def show_review(self, img, label_scanpaths=None, pred_scanpath=None, title_str=None, save_path=None):
        """
        inputs:
            img: img_file or img_data
            label_scanpaths: list[[pos_x, pos_y], [pos_x, pos_y], ...]
            pred_scanpath: list[idx1, idx2, idx3]
        """
        assert (label_scanpaths is not None) or (pred_scanpath is not None)

        IMG_HEIGHT = 480
        IMG_WEIGHT = 640

        if isinstance(img, str):
            img = Image.open(img)
        fig = plt.figure()

        if title_str is not None:
            plt.title(title_str)

        ax1 = fig.add_subplot(121)
        ax1.imshow(img)
        ax2 = fig.add_subplot(122)
        ax2.imshow(img)

        line_color = random.randint(1, 99)

        if label_scanpaths:
            for i, label_scanpath in enumerate(label_scanpaths):
                x1 = []
                y1 = []
                for j, point in enumerate(label_scanpath):
                    col_pos, row_pos = point
                    x1.append(col_pos)
                    y1.append(row_pos)
                    ax1.text(col_pos, row_pos, "{}".format(j+1), color='r')
                color_mat = sio.loadmat("cmat.mat")
                color_set = color_mat['cmat']
                ax1.plot(x1, y1, color=color_set[line_color], linewidth=1)

        trans_tools = Trans_tools()
        if pred_scanpath:
            x2 = []
            y2 = []
            for i, scan_idx in enumerate(pred_scanpath):
                row_idx, col_idx = trans_tools.trans_scanIdx2rcIdx(scan_idx)
                row_pos, col_pos = trans_tools.trans_rcIdx2pos(row_idx, col_idx, IMG_HEIGHT, IMG_WEIGHT)
                x2.append(col_pos)
                y2.append(row_pos)
                ax2.text(col_pos, row_pos, "{}".format(i+1), color='r')
            color_mat = sio.loadmat("cmat.mat")
            color_set = color_mat['cmat']
            ax2.plot(x2, y2, color=color_set[line_color], linewidth=1)

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    def show_scanpaths(self, img, scanpaths, title_str=None, save_path=None):
        """
        inputs:
            img: 
            scanpaths: 
            title_str: 
        """

        if isinstance(img, str):
            img = Image.open(img)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.imshow(img)

        color_mat = sio.loadmat("cmat.mat")
        color_set = color_mat['cmat']

        if title_str is not None:
            plt.title(title_str)

        tmp_color = random.randint(1, 99)
        tmp_color = 99
        color_val = int(99 // len(scanpaths))

        for scanpath in scanpaths:
            # tmp_color = random.randint(1, 99)
            x = []
            y = []
            for i, point in enumerate(scanpath):
                _x = point[0]
                _y = point[1]
                x.append(_x)
                y.append(_y)
                ax1.text(_x, _y, "{}".format(i+1), color=color_set[tmp_color])
            
            ax1.plot(x, y, color=color_set[tmp_color], linewidth=1.5)

            tmp_color += color_val
            tmp_color = tmp_color % 99
        
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)


class Analysis_tools(object):

    def __init__(self):
        pass

    def load_img_score(self, file_path):
        """
        return:
            raw_datas: dict(key=img_id, value=[score1, score2])
            scores: dict(key=1or2, value=[score, score, score])
        """
        raw_datas = {}
        scores = {1: [], 2:[]}
        with open(file_path, 'r') as fh:
            for line in fh:
                items = line.strip().split('\t')
                img_id = items[0]
                score = list(map(float, items[1:]))
                raw_datas[img_id] = score
                scores[1].append(score[0])
                scores[2].append(score[1])
        return raw_datas, scores
    
    def load_raw_score(self, raw_dir=None, raw_file=None):
        if raw_dir is None and raw_file is None:
            # assert FileNotFoundError
            assert Exception
        elif raw_dir is not None:
            pass
        elif raw_file is not None:
            return np.load(raw_file)
        else:
            pass


class Model_quota(object):

    def __init__(self):
        """
        precision and recall
        """
        pass
    
    def precision_and_recall(self, pred, label):
        assert len(pred) == len(label)
        # print("pred:\t{}".format(pred))
        # print("label:\t{}".format(label))
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i in range(len(pred)):
            if pred[i]==1 and label[i]==1:
                tp += 1
            elif pred[i]==1 and label[i]==0:
                fp += 1
            elif pred[i]==0 and label[i]==1:
                fn += 1
            elif pred[i]==0 and label[i]==0:
                tn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return precision, recall


class Mat_tools(object):
    
    def __init__(self):
        """
        The usage tool of .mat file.
        """
        pass
    
    def save_mat(self, out_file, dict_obj):
        """
        save list/ndarray obj to mat
        """
        sio.savemat(out_file, dict_obj)
        return True
    
    def load_mat(self, in_file):
        """
        load mat to ndarray
        """
        out_dict = {}
        dict_obj = sio.loadmat(in_file)
        for k, v in dict_obj.items():
            if k[:2] == "__":
                continue
            out_dict[k] = v
        return out_dict


class Pkl_tools(object):

    def __init__(self):
        """
        The usage tool of .pkl file.
        """
        pass
    
    def save_pkl(self, out_file, out_dict):
        with open(out_file, 'wb') as fh:
            pickle.dump(out_dict, fh, protocol=2)

    def load_pkl(self, in_file):
        with open(in_file, 'rb') as fh:
            out_dict = pickle.load(fh)
        return out_dict


class Data_process(object):

    def __init__(self):
        """
        Scanpath data preprocess
        """
        pass
    
    def pre_process_raw_data(self, file_name):
        """
        inputs:
            file_name: 眼动仪输出的一个xlsx文件
            一个xlsx文件，对应一次眼动实验，对应一个被试观看多个刺激
        outputs:
            scanpaths: list[scanpath_1, scanpath_2, ...]
            len(scanpaths)为刺激数
            其中:
                每个scanpath为fixation构成的list，
                每个fixation为[x, y, duration]。
            labels: list[cls_str1, cls_str2, ...]
        """
        ### df: all data
        df = pd.DataFrame(pd.read_excel(file_name))
        df = df.interpolate(method = 'linear', axis = 0)
        ### df1: some columns
        df1= df[['ParticipantName', 'MediaName', 'FixationIndex', 'GazeEventDuration', 'GazePointX (MCSpx)', 'GazePointY (MCSpx)']]
        ### df2: some rows
        df2 = df1.loc[df1['MediaName'] != '2-interval.jpg']
        df2 = df2.loc[df2['MediaName'] != '4-have_a_rest.jpg']

        ### init record
        labels = []
        last_idx = None 
        scanpaths = []
        ### get data
        for i in range(0, len(df2)):
            line = df2.iloc[i]
            MediaName = line['MediaName']
            FixationIndex = line['FixationIndex']
            GazeEventDuration = line['GazeEventDuration']
            GazePointX = line['GazePointX (MCSpx)']
            GazePointY = line['GazePointY (MCSpx)']
            fixation = [GazePointX, GazePointY, GazeEventDuration]
            
            if "stimulate" not in MediaName:
                if labels == []:
                    labels.append(MediaName)
                elif MediaName != labels[-1]:
                    labels.append(MediaName)
                continue

            if last_idx is None:
                last_idx = FixationIndex
                last_scanpath = []
                last_scanpath.append(fixation)
            elif FixationIndex - last_idx == 1:
                last_idx = FixationIndex
                last_scanpath.append(fixation)
            else:
                ### output
                tmp_scanpath = copy.deepcopy(last_scanpath)
                scanpaths.append(tmp_scanpath)
                ### reset
                last_idx = FixationIndex
                last_scanpath = []
                last_scanpath.append(fixation)
        ### output
        tmp_scanpath = copy.deepcopy(last_scanpath)
        scanpaths.append(tmp_scanpath)
        assert len(scanpaths) == len(labels)
        return scanpaths, labels


class Label_tools(object):
    """
    Label trans tools.
    Get the label matrix from the original label
    example:
        original_label: 
            [1, 2, 1, 2]
        label_matrix: 
            [[-1, 0, 1, 0],
            [0, -1, 0, 1],
            [1, 0, -1, 0],
            [0, 1, 0, -1]]
    """
    def __init__(self):
        pass

    def trans_truth_label(self, labels):
        """
        inputs:
            labels: ground truth
        """
        label_num = len(labels)
        label_onehot = np.zeros((label_num, label_num))
        for label_idx in range(label_num):
            cur_label = labels[label_idx]
            for cmp_idx in range(label_num):
                cmp_label = labels[cmp_idx]
                if label_idx == cmp_idx:
                    label_onehot[label_idx][cmp_idx] = -1
                elif cur_label == cmp_label:
                    label_onehot[label_idx][cmp_idx] = 1
                else:
                    continue
        return label_onehot

    def trans_pred_label(self, labels):
        """
        inputs:
            labels: predict results from kmeans
        """
        label_num = len(labels)
        label_onehot = np.zeros((label_num, label_num))
        for label_idx in range(label_num):
            if np.sum(labels[label_idx]) > (label_num//2):
                """exchange kmean_label_1 and kmean_label_0"""
                for cmp_idx in range(label_num):
                    if label_idx == cmp_idx:
                        label_onehot[label_idx][cmp_idx] = -1
                    elif labels[label_idx][cmp_idx] == 0:
                        label_onehot[label_idx][cmp_idx] = 1
                    else:
                        label_onehot[label_idx][cmp_idx] = 0
            else:
                for cmp_idx in range(label_num):
                    if label_idx == cmp_idx:
                        label_onehot[label_idx][cmp_idx] = -1
                    elif labels[label_idx][cmp_idx] == 0:
                        label_onehot[label_idx][cmp_idx] = 0
                    else:
                        label_onehot[label_idx][cmp_idx] = 1           
        return label_onehot