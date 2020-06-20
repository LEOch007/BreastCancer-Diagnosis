'''
detect and segement potential nuclei in miscropic images (H&E stained)
@author: Kemeng Chen 
'''
import os
import numpy as np 
import cv2
from time import time
from util import*
import re
from FeatureCalculation import *
import matplotlib.pyplot as plt
import pandas as pd

def process(data_folder, model_name, format):
	patch_size=128
	stride=16
	file_path=os.path.join(os.getcwd(), data_folder)
	name_list=os.listdir(file_path)
	print(str(len(name_list)), ' files detected')
	model_path=os.path.join(os.getcwd(), 'models')
	model=restored_model(os.path.join(model_path, model_name), model_path)
	print('Start time:')
	print_ctime()

	features = []

	for index, temp_name in enumerate(name_list):
		ts=time()
		print('process: ', str(index), ' name: ', temp_name)
		temp_path=os.path.join(file_path, temp_name)
		print(temp_path)
		if not temp_path.endswith('png') or ('label' in temp_path) or ('mask' in temp_path):
			continue
		# result_path=os.path.join(temp_path, 'mask.png')
		temp_image=cv2.imread(temp_path)
		if temp_image is None:
			raise AssertionError(temp_path, ' not found')
		batch_group, shape=preprocess(temp_image, patch_size, stride, temp_path)
		mask_list=sess_interference(model, batch_group)
		c_mask=patch2image(mask_list, patch_size, stride, shape)
		c_mask=cv2.medianBlur((255*c_mask).astype(np.uint8), 3)
		c_mask=c_mask.astype(np.float)/255
		thr=0.5
		c_mask[c_mask<thr]=0
		c_mask[c_mask>=thr]=1
		center_edge_mask, gray_map, center_points, edge_points=center_edge(c_mask, temp_image)

		# feature extraction
		image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
		feature = feature_extract(center_points, edge_points, image)
		feature.append(index)
		features.append(feature)

		name = re.sub('.png', '', temp_name)
		mask_name = name + '_mask.png'
		label_name = name + '_label.png'
		cv2.imwrite(os.path.join(file_path, mask_name), gray_map)
		cv2.imwrite(os.path.join(file_path, label_name), center_edge_mask)
		te=time()
		print('Time cost: ', str(te-ts))
		# fig, ax=plt.subplots(1,2)
		# ax[0].imshow(cv2.cvtColor(center_edge_mask, cv2.COLOR_BGR2RGB))
		# ax[0].set_title('label')
		# ax[1].imshow(gray_map)
		# ax[1].set_title('Center and contour')

	# 'radius', 'perimeter', 'area', 'compactness', 'smoothness', 'concavity', 'concavity_points',
	# 'symmetry', 'fractal_dimension', 'texture'

	df = pd.DataFrame(features,
		columns=['radius_m','perimeter_m','area_m','compactness_m','smoothness_m','concavity_m','concavity_points_m',
		'symmetry_m','fractal_dimension_m','texture_m','radius_se', 'perimeter_se', 'area_se', 'compactness_se',
				 'smoothness_se', 'concavity_se','concavity_points_se', 'symmetry_se', 'fractal_dimension_se', 'texture_se',
				 'radius_w', 'perimeter_w', 'area_w', 'compactness_w','smoothness_w', 'concavity_w', 'concavity_points_w',
				 'symmetry_w', 'fractal_dimension_w', 'texture_w', 'id'])
	df.to_csv('extractedFeatures.csv', index=0)
	model.close_sess()
	print('mask generation done')
	print_ctime()
	# plt.show()

def main():
	data_folder='data'
	model_name='nucles_model_v3.meta'
	format='.png'
	process(data_folder, model_name, format)

if __name__ == '__main__':
	main()
