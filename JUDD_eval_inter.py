# coding:utf-8
"""
Python2

Requirs:

conda env: py2_tf
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import cv2
import numpy as np

import JUDD_get_groundTruth as JUDD
import scanLD

JUDD_ROOT = "/home/lixiang/Desktop/personal-code/dataset/JUDD"
OUT_FILE = "/home/lixiang/Desktop/personal-code/scanLD/result/JUDD_inter_{}"

def main(cut_length):
	# init
	method_tool = scanLD.ScanLD(
        same_range=64,
        inception_range=64,
        calc_mode=1
    )
	print("cut_length: {} begin...".format(cut_length))
	#
	img_scores = {}
	img_count = 0
	img_sum = 0
	err_count = 0
	for img_name in os.listdir(os.path.join(JUDD_ROOT, "ALLSTIMULI/")):
		if not img_name.endswith('.jpeg'):
			continue
		# # debug >>>
		# if img_name != "i1280528250.jpeg":
		# 	continue	
		# # debug <<<
		img_file = os.path.join(JUDD_ROOT, "ALLSTIMULI", img_name)
		img_data = cv2.imread(img_file)
		y_resolution, x_resolution, channel_num = np.shape(img_data)
		# print(y_resolution, x_resolution)
		# cv2.imshow('img_data', img_data)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		img_id = img_name[:-5]
		fix_datas = JUDD.ground_truth_by_imgName_subNames(
			img_id,
			JUDD.ALL_JUDD_SUBNAMES,
			cut_length=cut_length,
			verbose=False
		)
		print("[{}] fix_datas get. len: [{}]".format(img_id, len(fix_datas)))
		nums_scanpath = len(fix_datas)
		scores = []
		for i in range(0, nums_scanpath-1):
			for j in range(i, nums_scanpath):
				if i == j:
					continue
				data1 = fix_datas[i]['position']
				data2 = fix_datas[j]['position']
				try:
					# score = scanMatch_tool.calc_score(data1, data2, x_resolution, y_resolution)
					score = method_tool.calc_sim_score(data1, data2)
				except:
					err_count += 1
					print('error occur')
					print(data1)
					print(data2)
					print(x_resolution)
					print(y_resolution)
					continue
				scores.append(score)
		img_score = np.mean(scores)

		img_scores[img_id] = img_score
		img_count += 1
		img_sum += img_score
		print("{}\t{}\t{}".format(img_count, img_id, img_score))
		# # debug >>>
		# if img_count == 10:
		# 	break
		# # debug <<<

	JUDD_score = img_sum / img_count
	print("cut_length: {}\tdataset_score: {}".format(cut_length, JUDD_score))
	print('err count: {}'.format(err_count))
	with open(OUT_FILE.format(cut_length), 'w') as fh:
		for img_id, img_score in img_scores.items():
			fh.write("{}\t{}\n".format(img_id, img_score))
		fh.write("dataset\t{}\t{}\n".format(cut_length, JUDD_score))
		fh.write("errinfo\tcount\t{}".format(err_count))

if __name__ == "__main__":
	for cut_length in range(1, 7):
		main(cut_length)
