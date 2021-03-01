import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D

###
# 首先，从SPEID提供的all_sequence_data.h5中获取数据（正：负 = 1：20）
# 然后，对负样本进行降采样
# 接着，将增强子(3000,4)和启动子(2000,4)数据通过hilbert编码（64*64*4）
# 最后，将编码结果以.h5的形式保存
###

cell = 'GM12878'  # IMR90


# 从h5数据集中加载数据
def load_data(cell_line):
    data_path = './data_Hilbert/all_sequence_data.h5'
    print('Loading ' + cell_line + ' data from ' + data_path)
    enhancers = None
    promoters = None
    labels = None
    with h5py.File(data_path, 'r') as hf:
        enhancers = np.array(hf.get(cell_line + '_X_enhancers')).transpose((0, 2, 1))
        promoters = np.array(hf.get(cell_line + '_X_promoters')).transpose((0, 2, 1))
        labels = np.array(hf.get(cell_line + 'labels')).astype(np.int)

    print('X_enhancers_shape:', enhancers.shape, len(enhancers))
    print('X_promoters_shape:', promoters.shape, len(promoters))
    print('labels:', type(labels), type(labels[0]), labels.shape)

    return enhancers, promoters, labels


enhancers, promoters, labels = load_data(cell)
#
# # 保存正样本（pos）和负样本的数量（neg）
# pos = 0
# neg = 0
# for i in labels:
#     if i == 0:
#         neg += 1
#     else:
#         pos += 1
# print(pos, neg)
#
# # 存储正样本和负样本对应的增强子和启动子
# pos_enhancers = enhancers[0:pos]
# pos_promoters = promoters[0:pos]
#
# neg_enhancers = enhancers[-neg:]
# neg_promoters = promoters[-neg:]
#
# ## 对负样本进行降采样
# # 在100以内随机取10个数字，分别用于增强子和启动子，实现正样本扩展
# random_num = np.random.randint(1, neg, pos)
# neg_enhancers_small = neg_enhancers[random_num]
# neg_promoters_small = neg_promoters[random_num]
#
# print(neg_enhancers_small.shape)
# print(neg_promoters_small.shape)
#
#
# # 合并所有数据
# len_pos = pos
# len_neg_small = len(neg_enhancers_small)
# len_all = len_neg_small + len_pos
# print('原始正样本：', pos)
# print('原始负样本：', neg)
# print('降采样负样本：', len(neg_enhancers_small))
# print('总样本：', len_all)
#
# new_enhancers_all = np.empty([len_all, 3000, 4])
# new_promoters_all = np.empty([len_all, 2000, 4])
# new_labels = np.empty([len_all, 1])
#
# number = 0
# for i in range(len_pos):
#     new_enhancers_all[i] = pos_enhancers[i]
#     new_promoters_all[i] = pos_promoters[i]
#     new_labels[number] = 1
#     number += 1
# print(number)
#
# for i in range(len_neg_small):
#     new_enhancers_all[len_pos + i] = neg_enhancers_small[i]
#     new_promoters_all[len_pos + i] = neg_promoters_small[i]
#     new_labels[len_pos + i] = 0
#     number += 1
# print(number)
#
# print(new_enhancers_all.shape)
# print(new_promoters_all.shape)
# print('负样本的EP数据降采样（finish!）')


# hilbert编码
def last_2_bits(x):
    return x & 3


def hindex_to_xy(hindex, N):
    positions = [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ]

    tmp = positions[last_2_bits(hindex)]
    hindex = hindex >> 2

    # 2. iteratively compute coords
    x = tmp[0]
    y = tmp[1]

    n = 4
    while n <= N:

        n2 = int(n / 2)
        pos_in_small_square = last_2_bits(hindex)

        if pos_in_small_square == 0:  # lower left
            tmp = x
            x = y
            y = tmp
        elif pos_in_small_square == 1:  # upper left
            x = x
            y = y + n2
        elif pos_in_small_square == 2:  # upper right
            x = x + n2
            y = y + n2
        elif pos_in_small_square == 3:  # lower right
            tmp = y
            y = (n2 - 1) - x
            x = (n2 - 1) - tmp
            x = x + n2

        hindex = hindex >> 2
        n *= 2

    return x, y


def draw_hilbert(order, fig_width, fig_height):
    fig, ax = plt.subplots()

    # Make graph square
    fig.set_size_inches(fig_width, fig_height)

    # Move graph window a little left and down
    # scatter([-0.1],[-0.1],s=0.01)

    N = 2 ** order;
    prev = (0, 0)

    print("drawing...")

    for i in range(N * N):
        curr = hindex_to_xy(i, N)
        # print(prev, curr)

        # line from prev to curr
        h_line = [prev, curr]
        (h_line_x, h_line_y) = zip(*h_line)
        ax.add_line(Line2D(h_line_x, h_line_y, linewidth=1, color='blue'))

        prev = curr

        if i % 1000 == 0:
            print(i, " done")

    plt.plot()
    plt.show()


def write_pixel_list_hilbert(order, file_name):
    point_list = []
    N = 2 ** order
    prev = (0, 0)

    print("-- writing --")

    pixel_count = 0
    with open(file_name, "w") as pixel_file:
        for i in range(N * N):
            curr = hindex_to_xy(i, N)
            point_list.append(curr)
            pixel_file.write(str(curr) + '\n')
            pixel_count += 1

    print("PixelCount: ", pixel_count)
    return point_list


file_name_hilbert = './data_Hilbert/' + cell + 'fantom_point_list.txt'
point_list = write_pixel_list_hilbert(6, file_name_hilbert)

dataset_enhancer = np.zeros((len(enhancers), 64, 64, 4))
dataset_promoter = np.zeros((len(promoters), 64, 64, 4))


def make_image(sequence):
    image_array = np.ones((64, 64, 4))
    for k in range(len(sequence)):
        x, y = point_list[k]  # 用(x,y)保存hilbert曲线的绘制顺序
        image_array[x][y] = sequence[k]  # 将坐标映射为图像
    for k in range(len(sequence), len(point_list)):
        x, y = point_list[k]
        image_array[x][y] = np.array([0, 0, 0, 0])
    return image_array


print(dataset_enhancer.shape)
print(dataset_promoter.shape)

# 将正负样本转化为三维图像
sample_count = 0
for k in range(len(enhancers)):
    if 'N' not in enhancers[k] and 'N' not in promoters[k]:
        dataset_enhancer[k] = make_image(enhancers[k])
        dataset_promoter[k] = make_image(promoters[k])
        sample_count += 1
labels = np.array(labels)
print(dataset_enhancer.shape)
print(dataset_promoter.shape)
print(labels.shape)


# 将正样本拓展之后的，增强子-启动子-label数据存储到h5文件中
file_name_hilbert_all = './data_Hilbert/' + cell + '_all_data_origin.h5'
if not os.path.exists(file_name_hilbert_all):
    with h5py.File(file_name_hilbert_all) as f:
        f['enhancers'] = dataset_enhancer
        f['promoters'] = dataset_promoter
        f['labels'] = labels
print('finish!')

