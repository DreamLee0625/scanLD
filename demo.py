# coding:utf-8
"""
demo of scanLD method
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import utils
import scanLD

# Number of scanpaths per condition
SCANPATH_NUM = 20

class Filter(object):
    
    def __init__(self):
        """
        1.Preprocesse original scanpaths from eye-tracker 
        2.Resample scanpaths in temporal dimension
        """
        pass
    
    def filter_by_duration(self, scanpaths, min_duration=200):
        """
        Reject fixation points whose duration is too short.
        """
        new_scanpaths = []
        for scanpath in scanpaths:
            tmp_new_scanpath = []
            for fixation in scanpath:
                if fixation[2] < min_duration:
                    continue
                tmp_new_scanpath.append(fixation)
            new_scanpaths.append(tmp_new_scanpath)
        return new_scanpaths

    def filter_clip(self, scanpaths):
        """
        Reject the first fixation point and the last one.
        Reason: 
            The position of the first fixation point is strongly related to the previous stimulus(image).
        A little experiment:
            The baseline group: keep the first fixation point.
            The experimental group: reject the first fixation point.
            The experimental group received higher similarity scores than the baseline group
        """
        new_scanpaths = []
        for scanpath in scanpaths:
            tmp_new_scanpath = copy.deepcopy(scanpath[1: -2])
            new_scanpaths.append(tmp_new_scanpath)
        return new_scanpaths

    def filter_temporalResample(self, scanpaths, resample_val=None):
        """
        Resample scanpaths in temporal dimension
        """
        new_scanpaths = []
        for scanpath in scanpaths:
            tmp_new_scanpath = []
            for fixation in scanpath:
                if resample_val is None:
                    resample_count = 1
                else:
                    resample_count = int(fixation[2] / resample_val)
                for _ in range(resample_count):
                    tmp_new_scanpath.append([fixation[0], fixation[1]])
            new_scanpaths.append(tmp_new_scanpath)
        return new_scanpaths


class Label_tools(object):
    
    def __init__(self):
        """
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
        pass

    def get_onehot_label(self, labels):
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

    def trans_kmeans_label(self, labels):
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


def load_raw_data(file_name):
    """
    inputs:
        file_name: original scanpaths output by eye-tracker
    outputs:
        scanpaths: list[scanpath_1, scanpath_2, ...]
        Among them:
            Each scanpath is a list of fixation points
            The data structure of each fixation point is [x, y, duration].
        labels: list[cls_str1, cls_str2, ...]
    """
    ### df: all data
    df = pd.DataFrame(pd.read_excel(file_name))
    df = df.interpolate(method = 'linear', axis = 0)
    ### df1: some columns
    df1= df[['MediaName', 'FixationIndex', 'GazeEventDuration', 'GazePointX (MCSpx)', 'GazePointY (MCSpx)']]
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
    assert len(scanpaths) == SCANPATH_NUM, print(len(scanpaths))
    assert len(labels) == SCANPATH_NUM, print("{}\t{}".format(len(labels), labels))
    return scanpaths, labels


def eval_onePerson(scanpaths, method_tool):
    """
    The pairwise similarity between scanpaths of each subject was compared with that of each subject as a unit.
    inputs:
        scanpaths:
        method_tool:
    returns:
        Similarity matrix
    """

    scanpath_num = len(scanpaths)
    score_mat = np.zeros((scanpath_num, scanpath_num))

    for i in range(0, scanpath_num-1):
        for j in range(i+1, scanpath_num):
            tmp_score = method_tool.calc_sim_score(
                scanpaths[i], scanpaths[j],
            )
            score_mat[i][j] = tmp_score
            score_mat[j][i] = tmp_score

    return score_mat


def kmean_analysis(score_mat):
    """
    inputs:
        Cluster analysis is carried out for each row in score matrix.
    returns:
        Returns the cluster center and the cluster label
    """
    center_value = []
    preds = []
    for i, line in enumerate(score_mat):
        # 排除自己与自己相比的情况
        new_line = []
        zers_idx = None
        for j, item in enumerate(line):
            if j == i:
                zers_idx = j
                continue
            new_line.append(item)
        new_line = np.array(new_line)
        # reshape数据并且进行kmean聚类
        new_line = new_line.reshape(-1, 1)
        estimator = KMeans(n_clusters=2)
        estimator.fit(new_line)
        # 获取2个聚类中心 [max_value, min_value]
        value_min = min(estimator.cluster_centers_)[0]
        value_max = max(estimator.cluster_centers_)[0]
        center_value.append([value_max, value_min])
        # 将聚类结果补充上自己与自己相比的情况，并且值为-1
        tmp_pred = list(copy.deepcopy(estimator.labels_))
        tmp_pred.insert(zers_idx, -1)
        preds.append(np.array(tmp_pred))
    return center_value, preds


def my_anova(score_mat, truth):
    """
    anova analysis
    """

    label_1_idx = np.where(truth==1)
    label_0_idx = np.where(truth==0)

    same_group_value = list(score_mat[label_1_idx])
    diff_group_value = list(score_mat[label_0_idx])
    same_label = list(np.ones(len(same_group_value)))
    diff_label = list(np.zeros(len(diff_group_value)))
    
    same_mean = np.mean(np.array(same_group_value))
    same_std = np.std(np.array(same_group_value))

    diff_mean = np.mean(np.array(diff_group_value))
    diff_std = np.std(np.array(diff_group_value))

    df_dic = {
        "group": same_label + diff_label,
        "score": same_group_value + diff_group_value
    }

    df = pd.DataFrame(df_dic)
    model = ols("score ~ group", df).fit()
    anovat = anova_lm(model)
    print(anovat)

    print("same: {}\t{}".format(same_mean, same_std))
    print("diff: {}\t{}".format(diff_mean, diff_std))


def check_data(scanpaths, labels, idxs=None):
    """ check input data """
    show_tools = utils.Show_tools()
    img_file = "exp4-stimulate/3-stimulate-1.jpg"
    for PATH1 in range(len(scanpaths)):
        if idxs is not None and PATH1 not in idxs:
            continue
        show_scanpaths = [scanpaths[PATH1]]
        title_str = "{}: {}".format(PATH1, labels[PATH1])
        show_tools.show_scanpaths(img_file, show_scanpaths, title_str)


def check_result(preds, labels):
    """ check output data """
    model_quota = utils.Model_quota()
    precisions = []
    recalls = []
    for i in range(len(preds)):
        precision, recall = model_quota.precision_and_recall(pred=preds[i], label=labels[i])
        precisions.append(precision)
        recalls.append(recall)
    return np.mean(precisions), np.mean(recalls), precisions, recalls


def main(raw_data, method_tool):
    showdata=False
    ###
    label_tools = Label_tools()
    filter_tools = Filter()
    ###
    scanpaths, labels = load_raw_data(raw_data)
    # scanpaths = scanpaths[4:]
    # labels = labels[4:]
    # ###
    # check_data(scanpaths, labels)
    ###
    scanpaths = filter_tools.filter_clip(scanpaths) # [1:-2]
    scanpaths = filter_tools.filter_by_duration(scanpaths, min_duration=200) # duration>200
    scanpaths = filter_tools.filter_temporalResample(scanpaths, resample_val=None)
    if showdata:
        scanpaths = adjust_scanpath(scanpaths)
    ###
    score_mat = eval_onePerson(scanpaths, method_tool)

    if showdata:
        plt.imshow(score_mat)
        plt.show()
        mat_tools = utils.Mat_tools()
        mat_tools.save_mat("./showdata/exp7-0ms.mat", {'scoreMat': score_mat})
        return
    ###
    center_value, label_raw = kmean_analysis(score_mat)
    ###
    print("calc precision and recall...")
    truth = label_tools.get_onehot_label(labels)
    pred = label_tools.trans_kmeans_label(label_raw)
    precision_mean, recall_mean, precisions, recalls = check_result(preds=pred, labels=truth)
    ###
    my_anova(score_mat, truth)
    ###
    center_value = np.array(center_value)
    print("detail...")
    print("p means precision, and r means recall")
    for idx, label in enumerate(label_raw):
        if sum(label) > 10:
            recall_idx = np.where(label==0)
        else:
            recall_idx = np.where(label==1)
        print("{}\tp: {:.2f}\tr: {:.2f}\tcenter: [{:.2f}, {:.2f}]\t{}".format(idx, precisions[idx], recalls[idx], center_value[idx][0], center_value[idx][1], recall_idx))
    print("center_mean:\t{}".format(np.mean(center_value, axis=0)))
    print("precision_mean:\t{}".format(precision_mean))
    print("recall_mean:\t{}".format(recall_mean))

    # show_tools = utils.Show_tools()
    # PATH1 = 10
    # img_file = "exp4-stimulate/3-stimulate-1.jpg"
    # for PATH2 in range(len(scanpaths)):
    #     show_scanpaths = [scanpaths[PATH1], scanpaths[PATH2]]
    #     sim_score = score_mat[PATH1][PATH2]
    #     title_str = "{}: {}\n{}: {}\nscore: {}".format(PATH1, labels[PATH1], PATH2, labels[PATH2], sim_score)
    #     # out_file = "showdata/{}-{}-{}".format("scanLD_exp2_Rec_07", PATH1, PATH2)
    #     out_file = None
    #     show_tools.show_scanpaths(img_file, show_scanpaths, title_str, save_path=out_file)


def adjust_mat(score_mat):
    """
    inputs:
    1234 1234 1234 1234 1234
    outputs:
    11111 22222 33333 44444
    """
    group_num = 4
    line_val = group_num

    output_mat = None
    for group_start in range(group_num):
        group_idxs = [epoch_idx*group_num+group_start for epoch_idx in range(5)]
        group_score = score_mat[group_idxs, :]
        if output_mat is None:
            output_mat = group_score
        else:
            output_mat = np.vstack((output_mat, group_score))
    output_mat_2 = None
    for group_start in range(group_num):
        group_idxs = [epoch_idx*group_num+group_start for epoch_idx in range(5)]
        group_score = score_mat[:, group_idxs]
        if output_mat_2 is None:
            output_mat_2 = group_score
        else:
            output_mat_2 = np.hstack((output_mat_2, group_score))
    print(output_mat_2)
    print(np.shape(output_mat_2))
    return output_mat_2


def adjust_scanpath(scanpaths):
    group_num = 4
    output_mat = []
    for group_start in range(group_num):
        group_idxs = [epoch_idx*group_num+group_start for epoch_idx in range(5)]
        for row_idx in group_idxs:
            output_mat.append(scanpaths[row_idx])
    return output_mat


if __name__ == "__main__":
    method_tool = scanLD.ScanLD(
        same_range=64,
        inception_range=64,
        calc_mode=1
    )

    main(
        raw_data="data/scanLD_exp-real-data-1_Rec 01.xlsx",
        method_tool=method_tool,
    )