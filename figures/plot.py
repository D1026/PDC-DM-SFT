import matplotlib.pyplot as plt
import re
import numpy as np

# --- training loss ---
# def extract_loss(log_file):
#     with open(log_file, 'r') as f:
#         log_data = f.readlines()
#
#     # 正则表达式来匹配日志中的loss和epoch
#     pattern = r"'loss': ([\d.]+).*'epoch': ([\d.]+)"
#
#     # 准备数据列表
#     losses = []
#     epochs = []
#
#     # 逐行处理日志数据
#     for line in log_data:
#         match = re.search(pattern, line)
#         print("line:", line)
#         print("match:", match)
#         if match:
#             loss, epoch = match.groups()
#             losses.append(float(loss))
#             epochs.append(float(epoch))
#             print(len(losses))
#             print(len(epochs))
#
#     return losses, epochs
#
#
# def plot_loss(log_file1, log_file2):
#     loss_values_1, step_values_1 = extract_loss(log_file1)
#     loss_values_2, step_values_2 = extract_loss(log_file2)
#
#     print("loss_values1:", len(loss_values_1))
#     print("loss_values2:", len(loss_values_2))
#
#     epoch_values1 = np.array(list(range(1, 2485))) / 800
#     epoch_values2 = np.array(list(range(1, 2408))) / 800
#     # epoch_values3 = np.array(list(range(1, 1173))) / 570
#
#     plt.plot(epoch_values1, loss_values_1, label='full paras', color='r')
#     plt.plot(epoch_values2, loss_values_2, label='emb freeze', color='g')
#     # plt.plot(epoch_values3, loss_values3, label='full+bugsql')
#
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Continue Pre-train Loss Curve')
#     plt.legend()
#     plt.show()
#
# # plot_loss('./pre-train.log', 'pre-train_freezeEmbedd-2.log')
#


# # -------------- loss decreasing with mask factors ----------
# def extract_loss(log_file):
#     with open(log_file, 'r') as file:
#         log_lines = file.readlines()
#
#     loss_values = []
#     step_values = []
#     for line in log_lines:
#         pattern = r"step is (\d+), loss is ([\d.]+)"  # 匹配模式
#         match = re.search(pattern, line)
#         if match:
#             step = int(match.group(1))
#             loss = float(match.group(2))
#             step_values.append(step)
#             loss_values.append(loss)
#
#     return step_values, loss_values
#
#
# def plot_loss(log_file1, log_file2, log_file3):
#     epoch_values1, loss_values1 = extract_loss(log_file1)
#     epoch_values2, loss_values2 = extract_loss(log_file2)
#     epoch_values3, loss_values3 = extract_loss(log_file3)
#
#     print("loss_values1:", len(loss_values1))
#     print("epoch_values2:", len(epoch_values2))
#     print("epoch_values3:", len(epoch_values3))
#     # epoch_values1 = np.array(list(range(1, 2485))) / 800
#     # epoch_values2 = np.array(list(range(1, 2408))) / 800
#     # epoch_values3 = np.array(list(range(1, 1173))) / 570
#
#     plt.plot(epoch_values2, loss_values2, label='p=0.0', color='g')
#     plt.plot(epoch_values1, loss_values1, label='p=0.5', color='r')
#     plt.plot(epoch_values3, loss_values3, label='p=0.8', color='y')
#
#     plt.axvline(x=425, color='g', linestyle='--')
#     plt.axvline(x=175, color='r', linestyle='--')
#     plt.axvline(x=125, color='y', linestyle='--')
#
#     plt.text(425, 0.1, 'best-ckpt:step 425', ha='center', va='bottom', color='g')
#     plt.text(175, 0.1, 'best-ckpt:step 175', ha='center', va='bottom', color='r')
#     plt.text(125, 0.15, 'best-ckpt:step 125', ha='center', va='bottom', color='y')
#     plt.xlabel('Steps')
#     plt.ylabel('Loss')
#     plt.title('Loss Curve')
#     plt.legend()
#     plt.show()
#
# plot_loss('./DS-DMSFT-0.5.log', 'DS-DMSFT-0.0.log', 'DS-DMSFT-0.8.log')




import matplotlib.pyplot as plt
import re
import numpy as np

# # ------------------ eval dataset loss ---------------
# def extract_loss(log_file):
#     with open(log_file, 'r') as file:
#         log_lines = file.readlines()
#
#     loss_values = []
#     epoch_values = []
#     for line in log_lines:
#         match_loss = re.search(r"'eval_loss': (\d+\.\d+)", line)
#         match_epoch = re.search(r"'epoch': (\d+\.\d+)", line)
#         if match_loss and match_epoch:
#             loss_values.append(float(match_loss.group(1)))
#             epoch_values.append(float(match_epoch.group(1)))
#     return epoch_values, loss_values
#
#
# def plot_loss(log_file1):
#     epoch_values1, loss_values1 = extract_loss(log_file1)
#
#     print("loss_values1:", len(loss_values1))
#     epoch_values1 = np.array(list(range(5, 161, 5))) / 35
#
#     plt.plot(epoch_values1, loss_values1, label='full-paras')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Loss Curve')
#     plt.legend()
#     plt.show()
#
# plot_loss('./eval.log')

# --- eval dataset Acc with mask ratio ---
# xs = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
# ys = [45.8, 46.6, 47.0, 50.0, 50.0, 50.0, 50.0, 48.1, 39.9]

import json
with open('./p_acc.json', 'r', encoding='utf-8') as f:
    xs, ys = json.load(f)

plt.plot(xs, ys, marker='o', label='deepseek', color='b')
plt.xlabel('mask ratio p')
# plt.ylabel()
plt.title('Bugfix Acc(%)')
for i, v in enumerate(ys):
    plt.annotate(str(v), (xs[i], v), textcoords="offset points", xytext=(0,5), ha='center')
plt.ylim(35, 55)
plt.legend()
plt.show()


# ---- CPT ----
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams.update({'font.size': 24})
# # 数据
# labels = ['cq7+SFT', 'cq7+DM-SFT', 'ds6.7+SFT', 'ds6.7+DM-SFT', 'dsV2+SFT', 'dsV2+DM-SFT']
# SFT = [42.6, 49.3, 43.8, 49.8, 43.9, 49.7]
# CPT_SFT = [43.9, 51.0, 45.1, 52.1, 45.7, 51.6]
#
# x = np.arange(len(labels))  # 标签位置
# width = 0.24  # 柱子宽度
#
# fig, ax = plt.subplots()
#
# # 画柱状图
# rects1 = ax.bar(x - width/2, SFT, width, label='no CPT')
# rects2 = ax.bar(x + width/2, CPT_SFT, width, label='CPT')
#
# # 添加一些文本标签，例如x轴和y轴的标签，标题等
# ax.set_ylabel('Accuracy(%)')
# ax.set_title('no CPT vs. CPT')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
#
# plt.ylim(0, 70)
# # 添加每个柱子的数值标签
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
# autolabel(rects1)
# autolabel(rects2)
#
# fig.tight_layout()
#
# plt.show()


# ---------- diff count distribution -------------
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import Counter
#
# # 示例数组
# data = np.array([1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 1, 16, 1, 1, 2, 1, 1, 6, 2, 1, 2, 2, 2, 6, 10, 1, 1, 2, 5, 26, 5, 1, 2, 1, 3, 1, 4, 1, 2, 3, 1, 1, 1, 1, 1, 30, 1, 1, 1, 2, 2, 1, 1, 1, 1, 3, 2, 1, 1, 2, 1, 1, 4, 4, 1, 2, 2, 1, 2, 3, 2, 1, 1, 2, 1, 1, 1, 1, 1, 44, 1, 1, 1, 1, 4, 7, 6, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 5, 1, 2, 1, 7, 2, 1, 1, 2, 4, 2, 1, 2, 1, 5, 1, 2, 1, 1, 25, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 4, 5, 2, 2, 2, 1, 1, 3, 1, 4, 2, 8, 1, 2, 2, 2, 8, 2, 9, 1, 1, 4, 1, 2, 2, 2, 1, 4, 10, 6, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 4, 2, 2, 3, 1, 1, 1, 2, 7, 1, 1, 2, 1, 9, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 43, 6, 2, 1, 1, 1, 1, 6, 7, 2, 2, 2, 2, 1, 6, 1, 1, 1, 2, 1, 1, 3, 1, 4, 4, 4, 1, 1, 1, 1, 1, 5, 1, 1, 6, 2, 3, 1, 14, 3, 1, 1, 3, 1, 4, 1, 1, 1, 1, 4, 1, 2, 1, 2, 4, 2, 3, 1, 2, 2, 1, 1, 3, 1, 1, 22, 1, 1, 2, 8, 1, 40, 2, 1, 12, 2, 2, 1, 1, 1, 3, 1, 1, 1, 2, 1, 2, 4, 2, 4, 1, 1, 1, 1, 4, 1, 2, 1, 1, 1, 11, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 3, 1, 1, 9, 1, 2, 1, 2, 1, 1, 1, 1, 4, 1, 1, 2, 2, 1, 1, 2, 3, 3, 2, 2, 2, 1, 2, 4, 1, 1, 1, 3, 1, 1, 1, 1, 1, 4, 1, 1, 1, 2, 24, 1, 1, 1, 2, 1, 1, 2, 3, 1, 1, 1, 33, 1, 2, 2, 1, 1, 3, 2, 2, 2, 1, 3, 1, 1, 1, 3, 2, 1, 1, 4, 2, 1, 1, 1, 2, 4, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 1, 1, 7, 2, 6, 12, 1, 1, 4, 13, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 5, 2, 1, 4, 1, 1, 2, 8, 1, 1, 1, 2, 2, 2, 1, 1, 7, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 4, 2, 1, 8, 1, 6, 1, 1, 1, 6, 3, 1, 1, 2, 2, 1, 1, 1, 1, 10, 10, 2, 1, 2, 1, 1, 2, 5, 2, 2, 1, 1, 1, 1, 3, 2, 1, 2, 1, 1, 4, 1, 1, 4, 12, 1, 1, 2, 8, 1, 1, 6, 1, 7, 4, 2, 1, 4, 1, 1, 1, 4, 1, 1, 1, 1, 4, 2, 1, 2, 2, 1, 1, 1, 12, 1, 3, 1, 3, 20, 4, 2, 2, 2, 1, 1, 4, 1, 1, 3, 1, 1, 5, 1, 1, 1, 1, 14, 3, 1, 1, 2, 1, 2, 1, 4, 7, 3, 19, 1, 1, 13, 1, 1, 1, 1, 2, 3, 1, 1, 3, 5, 5, 1, 2, 11, 2, 2, 1, 1, 1, 1, 6, 10, 1, 3, 3, 9, 1, 1, 3, 1, 2, 1, 6, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 4, 5, 1, 7, 1, 9, 9, 11, 1, 2, 2, 3, 1, 2, 6, 5, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 3, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 3, 2, 1, 3, 1, 1, 2, 1, 7, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 9, 1, 2, 2, 4, 1, 3, 2, 35, 12, 1, 32, 1, 2, 2, 1, 1, 1, 4, 1, 1, 1, 1, 3, 2, 2, 1, 3, 1, 1, 2, 4, 1, 4, 1, 7, 1, 1, 1, 2, 1, 3, 3, 1, 1, 2, 1, 1, 1, 1, 6, 3, 1, 1, 2, 1, 1, 1, 1, 1, 3, 1, 5, 3, 3, 1, 1, 1, 4, 1, 2, 2, 1, 1, 2, 2, 6, 1, 4, 1, 1, 2, 2, 2, 1, 6, 1, 1, 1, 1, 59, 2, 1, 2, 1, 8, 1, 1, 1, 1, 1, 2, 1, 2, 3, 12, 1, 1, 1, 1, 3, 1, 1, 1, 4, 4, 2, 1, 1, 1, 1, 2, 1, 4, 14, 4, 3, 3, 2, 1, 7, 3, 1, 4, 2, 2, 1, 1, 1, 1, 1, 1, 5, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 3, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 1, 1, 1, 5, 3, 1, 2, 1, 1, 4, 5, 3, 3, 1, 2, 4, 1, 1, 2, 10, 1, 1, 1, 1, 2, 1, 27, 29, 2, 2, 2, 11, 1, 1, 1, 4, 32, 1, 3, 2, 1, 1, 1, 3, 2, 6, 1, 1, 1, 5, 1, 6, 2, 1, 6, 1, 1, 1, 2, 1, 1, 1, 3, 1, 2, 3, 2, 1, 3, 2, 2, 5, 1, 1, 4, 1, 7, 12, 1, 6, 2, 1, 1, 1, 3, 1, 2, 1, 1, 1, 1, 1, 2, 2, 1, 10, 1, 1, 1, 4, 3, 2, 4, 3, 2, 2, 1, 2, 2, 1, 4, 1, 7, 1, 1, 4, 2, 1, 7, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 4, 2, 1, 2, 1, 3, 2, 3, 2, 1, 2, 2, 7, 1, 1, 1, 2, 2, 1, 5, 1, 1, 1, 3, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 2, 3, 3, 4, 1, 1, 2, 4, 1, 4, 1, 1, 1, 2, 2, 1, 1, 1, 6, 1, 1, 1, 1, 15, 1, 1, 2, 1, 1, 2, 2, 24, 1, 5, 1, 2, 4, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 15, 1, 1, 2, 1, 1, 7, 1, 1, 1, 4, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 22, 26, 2, 1, 3, 4, 3, 1, 2, 1, 1, 3, 1, 1, 1, 3, 1, 4, 1, 3, 1, 1, 1, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1, 2, 1])
#
# # 统计每个数值的频次
# counter = Counter(data)
#
# # 计算每个数值的占比
# total_count = len(data)
# percentages = {k: v / total_count * 100 for k, v in counter.items()}
#
# # 准备绘图数据
# values = list(percentages.keys())
# percentages = list(percentages.values())
#
# # 绘制直方图
# plt.bar(values, percentages, tick_label=values, color='#FFA500')
#
# font_properties = {'family': 'Times New Roman', 'weight': 'bold', 'size': 14}
#
# plt.xlabel('Number of Diff Lines', fontdict=font_properties)
# plt.ylabel('Proportion (%)', fontdict=font_properties)
# plt.title('Distribution of Diff Lines in Dataset', fontdict=font_properties)
# plt.show()
