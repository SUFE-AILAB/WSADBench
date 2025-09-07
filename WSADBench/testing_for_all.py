import numpy as np

file = np.load("/data/coding/wsad/new_wzb/WSADBench/WSADBench/datasets/CV_by_I3D/UCF_Crime/segmentation/all_rgbs/32/Abuse/Abuse001_x264.mp4.npy", allow_pickle=True)
file1 = np.load("/data/coding/wsad/new_wzb/WSADBench/WSADBench/datasets/classical_bags_inexact/1_ALOI_bags.npz", allow_pickle=True)
file2 = np.load("/data/coding/wsad/new_wzb/WSADBench/WSADBench/datasets/Classical/1_ALOI.npz", allow_pickle=True)
print(file.shape) 
print(file1['X'].shape)
print(file2['X'].shape)