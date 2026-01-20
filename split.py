# import torch
# import numpy as np
# import pickle
# from tqdm import tqdm
#
# dataset = "ntu60"
# split = "split_12"
# #
# # split_1_5 = [4,19,31,47,51]## train:40091 test:16487 seen_train:36745 seen_test:15113 unseen:1374
# # split_1_10 = [4, 19, 31, 47, 51, 8, 21, 25, 46,
# #               54]  # train:40091 test:16487 seen_train:33437 seen_test:13741 unseen:2746
# # split_1_15 = [4, 19, 31, 47, 51, 2, 6, 7, 16, 24, 26, 29, 40, 57,
# #               58]  # train:40091 test:16487 seen_train:30062 seen_test:12363 unseen:4124
# # split_1_20 = [4, 19, 31, 47, 51, 8, 11, 12, 15, 17, 22, 25, 27, 30, 32, 36, 37, 45, 50,
# #               57]  # train:40091 test:16487 seen_train:27423 seen_test:11272 unseen:5215
# # split_1_25 = [4, 19, 31, 47, 51, 0, 2, 3, 5, 6, 7, 12, 14, 20, 24, 29, 30, 33, 35, 39, 45, 49, 50, 53,
# #               56]  # train:40091 test:16487 seen_train:23366 seen_test:9621 unseen:6866
# # split_1_30 = [4, 19, 31, 47, 51,
# #               0, 1, 3, 7, 8, 10, 12, 14, 16, 20, 22, 23, 24, 25, 28, 39, 41, 42, 44, 45, 48, 49, 53, 54,
# #               58]  # train:40091 test:16487 seen_train:20709 seen_test:8511 unseen:7976
# # split_1_35 = [4, 19, 31, 47, 51,
# #               1, 2, 5, 6, 7, 8, 10, 11, 12, 14, 15, 17, 18, 22, 24, 27, 28, 29, 30, 35, 38, 40, 41, 42, 44, 45, 48,
# #               53, 54, 58]  # train:40091 test:16487 seen_train:17348 seen_test:7147 unseen:9340
# # split_1_40 = [4, 19, 31, 47, 51,
# #               0, 3, 5, 7, 8, 11, 12, 13, 14, 15, 16, 21, 22, 25, 26, 27, 29, 30, 32, 34, 35, 36, 38, 39, 40, 42, 43,
# #               45,
# #               46, 52, 53, 54, 55, 56, 57]  # train:40091 test:16487 seen_train:14005 seen_test:5773 unseen:10714
# # split_1_45 = [4, 19, 31, 47, 51,
# #               0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 33, 35, 36,
# #               38, 40, 41, 42, 43, 45, 48, 50, 53, 55, 56,
# #               58]  # train:40091 test:16487 seen_train:10665 seen_test:4395 unseen:12092
# # split_1_50 = [4, 19, 31, 47, 51,
# #               0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 32, 33,
# #               34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 49, 52, 53, 54, 55, 56,
# #               58]  # train:40091 test:16487 seen_train:7349 seen_test:3019 unseen:13468
# # split_1_55 = [4, 19, 31, 47, 51,
# #               0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 22, 23, 24, 25, 26, 27, 29, 30, 32,
# #               33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 55,
# #               57]  # train:40091 test:16487 seen_train:4007 seen_test:1647 unseen:14840
# #1,2,3,5,6,7,8,9,10,11
# # ,12,13,14,15,16,17,18,20,21
# # ,22,23,24,25,26,27,28,29,30,
# # 32,33,34,35,36,37,38,39,40,
# # 41,42,43,44,45,46,48,49,50,
# # 52,53,54,55,56,57,58,59
# split_2 = [12,29,32,44,59]
# split_3 = [7,20,28,39,58]
#
# split_4 = [3, 18, 26, 38, 41, 60, 87, 99, 102, 110]
# split_5 = [5, 12, 14, 15, 17, 42, 67, 82, 100, 119]
# split_6 = [6, 20, 27, 33, 42, 55, 71, 97, 104, 118]
#
# split_7 = [1, 9, 20, 34, 50]
# split_8 = [3, 14, 29, 31, 49]
# split_9 = [2, 15, 39, 41, 43]
# split_12=[3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59]
# root="/media/zzf/ljn/wsx/Data/split/"
# root2="/media/zzf/ljn/wsx/zero-shot_data/sourcedata/"
# train_path = root2+dataset+'_frame50/xsub/train_position.npy'
# test_path = root2+dataset+'_frame50/xsub/val_position.npy'
# train_label_path = root2+dataset+'_frame50/xsub/train_label.pkl'
# test_label_path = root2+dataset+'_frame50/xsub/val_label.pkl'
#
# seen_train_data_path = root+dataset+"/"+split+"/seen_train_data.npy"
# seen_train_label_path = root+dataset+"/"+split+"/seen_train_label.npy"
# seen_test_data_path = root+dataset+"/"+split+"/seen_test_data.npy"
# seen_test_label_path = root+dataset+"/"+split+"/seen_test_label.npy"
# unseen_data_path = root+dataset+"/"+split+"/unseen_data.npy"
# unseen_label_path = root+dataset+"/"+split+"/unseen_label.npy"
#
# with open(train_label_path, 'rb') as f:
#     _, train_label = pickle.load(f)
#
# with open(test_label_path, 'rb') as f:
#     _, test_label = pickle.load(f)
#
# train_data = np.load(train_path)
# test_data = np.load(test_path)
#
# print("train size:",train_data.shape)
# print("test size:",test_data.shape)
#
# seen_train_data = []
# seen_train_label = []
# seen_test_data = []
# seen_test_label = []
# unseen_data = []
# unseen_label = []
#
# for i in range(len(train_label)):
#     if train_label[i] not in eval(split):
#         seen_train_label.append(train_label[i])
#         seen_train_data.append(train_data[i])
#
# for i in range(len(test_label)):
#     if test_label[i] in eval(split):
#         unseen_label.append(test_label[i])
#         unseen_data.append(test_data[i])
#     else:
#         seen_test_label.append(test_label[i])
#         seen_test_data.append(test_data[i])
#
# seen_train_data = np.array(seen_train_data)
# seen_test_data = np.array(seen_test_data)
# unseen_data = np.array(unseen_data)
#
# print(seen_train_data.shape)
# print(len(seen_train_label))
# print(seen_test_data.shape)
# print(len(seen_test_label))
# print(unseen_data.shape)
# print(len(unseen_label))
# np.save(seen_train_data_path, seen_train_data)
# np.save(seen_train_label_path, seen_train_label)
# np.save(seen_test_data_path, seen_test_data)
# np.save(seen_test_label_path, seen_test_label)
# np.save(unseen_data_path, unseen_data)
# np.save(unseen_label_path, unseen_label)
import torch
import numpy as np
import pickle
from tqdm import tqdm
import os

dataset = "ntu60"
root ="/media/zzf/ljn/wsx/Data/split/"
root2 = "/media/zzf/ljn/wsx/zero-shot_data/sourcedata/"

# 基础分割
# split_1 = [4, 19, 31, 47, 51]
# split_12=[3,5,9,12,15,40,42,47,51,56,58,59]
split_12=[3, 7, 13, 19, 22, 28, 34, 41, 47, 52, 56, 59]
# root="/media/zzf/ljn/wsx/Data/split/"
# remaining_numbers = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21,
#                      22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40,
#                      41, 42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59]
remaining_numbers = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26,
                     27, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50, 51, 53, 54, 55, 57, 58]
# 加载数据
train_path = root2 + dataset + '_frame50/xsub/train_position.npy'
test_path = root2 + dataset + '_frame50/xsub/val_position.npy'
train_label_path = root2 + dataset + '_frame50/xsub/train_label.pkl'
test_label_path = root2 + dataset + '_frame50/xsub/val_label.pkl'

with open(train_label_path, 'rb') as f:
    _, train_label = pickle.load(f)

with open(test_label_path, 'rb') as f:
    _, test_label = pickle.load(f)

train_data = np.load(train_path)
test_data = np.load(test_path)

print("开始处理48个分割方案...")

# 处理55个分割方案
for num in tqdm(remaining_numbers, desc="处理分割方案"):
    split_name = f"split_12_{num}"
    split_classes = split_12 + [num]

    # 创建输出目录
    split_dir = os.path.join(root, dataset, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # 定义输出路径
    seen_train_data_path = os.path.join(split_dir, "seen_train_data.npy")
    seen_train_label_path = os.path.join(split_dir, "seen_train_label.npy")
    seen_test_data_path = os.path.join(split_dir, "seen_test_data.npy")
    seen_test_label_path = os.path.join(split_dir, "seen_test_label.npy")
    unseen_data_path = os.path.join(split_dir, "unseen_data.npy")
    unseen_label_path = os.path.join(split_dir, "unseen_label.npy")

    # 分割数据
    seen_train_data = []
    seen_train_label = []
    seen_test_data = []
    seen_test_label = []
    unseen_data = []
    unseen_label = []

    # 处理训练数据
    for i in range(len(train_label)):
        if train_label[i] not in split_classes:
            seen_train_label.append(train_label[i])
            seen_train_data.append(train_data[i])

    # 处理测试数据
    for i in range(len(test_label)):
        if test_label[i] in split_classes:
            unseen_label.append(test_label[i])
            unseen_data.append(test_data[i])
        else:
            seen_test_label.append(test_label[i])
            seen_test_data.append(test_data[i])

    # 转换为numpy数组并保存
    np.save(seen_train_data_path, np.array(seen_train_data))
    np.save(seen_train_label_path, np.array(seen_train_label))
    np.save(seen_test_data_path, np.array(seen_test_data))
    np.save(seen_test_label_path, np.array(seen_test_label))
    np.save(unseen_data_path, np.array(unseen_data))
    np.save(unseen_label_path, np.array(unseen_label))

print("48个分割方案处理完成！")