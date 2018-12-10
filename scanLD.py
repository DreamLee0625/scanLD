# coding:utf-8
"""
python version: 2 / 3
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys

import numpy as np
import copy


class ScanLD(object):

    def __init__(self, same_range, inception_range, calc_mode=1, is_reversed=False, debug_info=False):
        """
        _calc_distance(self, point1, point2)
        _inplace_cost(self, point1, point2)
        _is_same_block(self, p1, p2)

        levenshtein_distance(self, in_path1, in_path2)

        reference params:
        if same_range is None -> grid-based mode (also called block mode)
        if same_range is not None -> point mode
        if grid-based inception_range = 2
        if point inception_range = 32
        """
        self.same_range = same_range
        self.inception_range = inception_range
        self.calc_mode = calc_mode
        self.is_reversed = is_reversed
        self.debug_info = debug_info
    
    def _eprint(self, instr):
        if self.debug_info == True:
            print(instr, file=sys.stderr)

    def _calc_distance(self, point1, point2):
        """
        inputs:
            if reversed == False:
                point1: [x, y]
                point2: [x, y]
            if reversed == True:
                point1: [y, x]
                point2: [y, x]
        """
        if self.is_reversed:
            point1 = [point1[1], point1[0]]
            point2 = [point2[1], point2[0]]
        point1 = np.array(point1)
        point2 = np.array(point2)
        res = np.sqrt(np.sum((point1 - point2)**2))
        return res

    def _inplace_cost(self, point1, point2):
        """
        inputs:
            point: [posX, posY]
            reverse == True -> [x, y]
            same_range = None -> grid-based method (also called block mode)
            same_range = 0 -> point method

        inception-range is the boundary of the maximum cost of the substitution operation, 
        maximum cost = 1
        When the distance between fixation points is greater than the maximum boundary, 
        they are no longer related to each other, 
        so the editing cost of the substitution operation between them is 1.
        """
        dist = self._calc_distance(point1, point2)
        if self.same_range is not None:
            """ mode == point """
            res = min((dist-self.same_range) / self.inception_range, 1)
        else:
            """ mode == block """
            res = min(dist/self.inception_range, 1)
        self._eprint("{} and {} inplace cost: {}".format(point1, point2, res))
        return res

    def _is_same_block(self, p1, p2):
        """
        'same_range = None' is different from 'same_range = 0', but they have same results.
        """
        if self.same_range is not None:
            dist = self._calc_distance(p1, p2)
            if dist <= self.same_range:
                res = True
            else:
                res = False
            self._eprint("{} and {} is_same_block: {}".format(p1, p2, res))
            return res
        else:
            return p1 == p2
    
    def levenshtein_distance(self, in_path1, in_path2):
        """
        inputs:
            in_path: sequence of idx or sequence of pos [idx: same_range=None，pos: same_range=int]
            same_range:
                same_range, also called match_range.
                Static partitioning of an area may result in two adjacent points being in adjacent areas. So dynamically partition the region.
        difference between block and point:
        1. inception_range
        2._is_same_block()
        3._inplace_cost()

        difference between block and point(same_range=0): i
        nception_range.

        example of input：
        input 1：
            [[5, 1], [3, 3], [2, 5], [2, 6]], 
            [[4, 1], [3, 3], [2, 5], [2, 6], [3, 6]],
            same_range=None
        input 2：
            [[88, 24], [56, 56], [40, 88], [40, 104]],
            [[72, 24], [56, 56], [40, 88], [40, 104], [56, 104]],
            same_range=0
        input 3：
            [[88, 24], [56, 56], [40, 88], [40, 104]],
            [[72, 24], [56, 56], [40, 88], [40, 104], [56, 104]],
            same_range=8
        """
        path1 = copy.deepcopy(in_path1)
        path2 = copy.deepcopy(in_path2)
        
        path1_len = len(path1) # 4
        path2_len = len(path2) # 5

        dist_mat = [[0]*(path2_len+1) for _ in range(path1_len+1)] # row*col = 5*6

        for i in range(1, path1_len+1):
            dist_mat[i][0] = i
        for j in range(1, path2_len+1):
            dist_mat[0][j] = j
        for i in range(1, path1_len+1):
            for j in range(1, path2_len+1):
                if( self._is_same_block(path1[i-1], path2[j-1]) ):
                    dist_mat[i][j] = dist_mat[i-1][j-1]
                else:
                    self.is_reversed = True
                    inplace_cost = self._inplace_cost(path1[i-1], path2[j-1])
                    dist_mat[i][j] = min(
                        dist_mat[i-1][j]+1, # delete
                        dist_mat[i][j-1]+1, # insert
                        dist_mat[i-1][j-1]+inplace_cost, # inplace
                    )
        return dist_mat[path1_len][path2_len]

    def longest_common_subsequence(self, path1, path2, same_range=None):
        """
        inputs:
            path：sequence of idx or sequence of pos
            same_range decide the mode of function "is_same_block":
                if idx(same_range=None) -> equal;
                if pos(same_range=int) -> threshold。

        example of input：
            [[88, 24], [56, 56], [40, 88], [40, 104]],
            [[72, 24], [56, 56], [40, 88], [40, 104], [56, 104]],
        """
        len1 = len(path1)
        len2 = len(path2)
        dp = [[0]*(len2+1) for _ in range(len1+1)]
        for i in range(1, len1+1):
            for j in range(1, len2+1):
                if self._is_same_block(path1[i-1], path2[j-1]):
                    dp[i][j] = dp[i-1][j-1]+1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

    def first_diff_pos(self, path1, path2, same_range=None):
        """
        inputs:
            path：sequence of idx or sequence of pos
            same_range decide the mode of function "is_same_block":
                if idx(same_range=None) -> equal;
                if pos(same_range=int) -> threshold。
        """
        res = min(len(path1), len(path2))
        for i in range(min(len(path1), len(path2))):
            if not self._is_same_block(path1[i], path2[i]):
                res = i
                break
        return res

    def calc_sim_score(self, path1, path2):
        if self.calc_mode == 1:
            ld = self.levenshtein_distance(path1, path2)
            maxlen = max(len(path1), len(path2))
            score = 1 - ld / maxlen
        elif self.calc_mode == 2:
            ld = self.levenshtein_distance(path1, path2)
            lcs = self.longest_common_subsequence(path1, path2)
            try:
                score = lcs / (ld + lcs)
            except:
                print("debug info...")
                print(path1)
                print(path2)
                print(ld)
                print(lcs)
                raise Exception
        else: # calc_mode = 0
            ld = self.levenshtein_distance(path1, path2)
            maxlen = max(len(path1), len(path2))
            score_1 = 1 - ld / maxlen
            lcs = self.longest_common_subsequence(path1, path2)
            score_2 = lcs / (ld + lcs)
            score = (score_1, score_2)
        return score


if __name__ == "__main__":
    ### test case: point
    scanLD = ScanLD(same_range=8, inception_range=16, calc_mode=1, debug_info=False)
    print(scanLD.calc_sim_score(
        [[88, 24], [56, 56], [40, 88], [40, 104], [56, 104]],
        [[72, 24], [56, 56], [40, 88], [40, 104], [56, 104]],
    ))

    ### test case: block
    # scanLD = ScanLD(same_range=None, inception_range=2, calc_mode=1)
    # print(scanLD.calc_sim_score(
    #     [[72, 24], [56, 56], [40, 88], [40, 104], [56, 104]],
    #     [[72, 24], [56, 56], [40, 88], [40, 104], [56, 104]],
    # ))