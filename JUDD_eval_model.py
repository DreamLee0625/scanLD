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
import json

import cv2
import numpy as np

import JUDD_get_groundTruth as JUDD
import scanLD

JUDD_ROOT = "/home/lixiang/Desktop/personal-code/dataset/JUDD"
JUDD_ALLSTIMULI = os.path.join(JUDD_ROOT, 'ALLSTIMULI')

# 1998 ltti by lixiang
# PRED_FILE = "/home/lixiang/Desktop/personal-code/1998-ltti/result_scanpath/result_scanpath#{}"
# OUT_FILE = "/home/lixiang/Desktop/personal-code/1998-ltti/result_score/JUDD_ltti_{}"
# 2007 JUDD	by lixiang
# PRED_FILE = "/home/lixiang/Desktop/personal-code/2009-JUDD/result_scanpath/path_point_svm#f4#std/result_scanpath#{}"
# OUT_FILE = "/home/lixiang/Desktop/personal-code/2009-JUDD/result_score/score_point_svm#f4#std/JUDD_JUDD_{}"
# 1998 ltti by uthor
# PRED_FILE = "/home/lixiang/Desktop/code/1998-ltti/result_scanpath/result_scanpath#{}"
# OUT_FILE = "/home/lixiang/Desktop/code/1998-ltti/result_score/JUDD_ltti_{}"
# 2007 JUDD	by uthor
# PRED_FILE = "/home/lixiang/Desktop/code/2009-JUDD/our_saliency_model/result_scanpath/result_scanpath#{}"
# OUT_FILE = "/home/lixiang/Desktop/code/2009-JUDD/our_saliency_model/result_score/JUDD_ltti_{}"
# 2007 JUDD	by uthor
PRED_FILE_MAXLEN = "/home/lixiang/Desktop/personal-code/scanpath-NN-torch/result_scanpath/result_scanpath"
OUT_FILE = "/home/lixiang/Desktop/personal-code/scanpath-NN-torch/result_score/JUDD_nn0226_{}"
def main(cut_length):
	### init eval_tool
	method_tool = scanLD.ScanLD(
        same_range=64,
        inception_range=64,
        calc_mode=1
    )
	print("cut_length: {} begin...".format(cut_length))
	###
	pred_scanpaths = {}
	img_resolution = {}
	# with open(PRED_FILE.format(cut_length), 'r') as fh:
	# 	for line in fh:
	# 		items = line.strip().split('\t')
	# 		img_id = items[0]
	# 		resolution = json.loads(items[1])
	# 		scanpath = json.loads(items[2])
	# 		pred_scanpaths[img_id] = scanpath
	# 		img_resolution[img_id] = resolution
	with open(PRED_FILE_MAXLEN, 'r') as fh:
		for line in fh:
			items = line.strip().split('\t')
			img_id = items[0]
			resolution = json.loads(items[1])
			scanpath = json.loads(items[2])
			scanpath = scanpath[:cut_length]
			pred_scanpaths[img_id] = scanpath
			img_resolution[img_id] = resolution
	###
	img_scores = {}
	img_count = 0
	img_sum = 0
	err_count = 0
	for img_name in os.listdir(os.path.join(JUDD_ALLSTIMULI)):
		if not img_name.endswith('.jpeg'):
			continue
		### debug >>>
		# if img_name != "i1280528250.jpeg":
		# 	continue	
		### debug <<<
		
		# img_file = os.path.join(JUDD_ALLSTIMULI, img_name)
		# img_data = cv2.imread(img_file)
		# y_resolution, x_resolution, channel_num = np.shape(img_data)

		img_id = img_name[:-5]
		### get groundTruth
		fix_datas = JUDD.ground_truth_by_imgName_subNames(
			img_id,
			JUDD.ALL_JUDD_SUBNAMES,
			cut_length=cut_length,
			verbose=False
		)
		print("[{}] fix_datas get. len: [{}]".format(img_id, len(fix_datas)))
		### get predict
		data1 = pred_scanpaths[img_id]

		### calc score
		nums_scanpath = len(fix_datas)
		scores = []
		for i in range(nums_scanpath):
			data2 = fix_datas[i]['position']
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
