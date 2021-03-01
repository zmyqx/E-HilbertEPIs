import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from tqdm import tqdm
import pandas as pd
import random
# 从h5数据集中加载数据
def load_data(cell_line):
    data_path = r'G:\迅雷下载\all_sequence_data\all_sequence_data.h5'
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
    print(file_name)
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


def make_image(sequence):
    image_array = np.ones((64, 64, 4))
    for k in range(len(sequence)):
        x, y = point_list[k]  # 用(x,y)保存hilbert曲线的绘制顺序
        image_array[x][y] = sequence[k]  # 将坐标映射为图像
    for k in range(len(sequence), len(point_list)):
        x, y = point_list[k]
        image_array[x][y] = np.array([0, 0, 0, 0])
    return image_array

if __name__ == '__main__':
    a=[1,2,3,4]
    cell_name = ['NHEK', 'IMR90', 'HUVEC', 'K562', 'GM12878', 'HeLa-S3']
    data_path=r'K:/data_Hilbert/cell_npy/'
    csv_path=r'K:/data_Hilbert/cell_csv/'
    for cell in cell_name:
        file_name_hilbert = 'G:/data_Hilbert/' + cell + 'fantom_point_list.txt'
        file_name_hilbert_all = 'G:/data_Hilbert/' + cell + '_all_data_hilbert_aug.h5'

        enhancers, promoters, labels = load_data(cell)
        # 保存正样本（p）和负样本的数量（n）
        p = 0
        n = 0
        for i in labels:
            if i == 0:
                n += 1
            else:
                p += 1

        # 在100以内随机取10个数字，分别用于增强子和启动子，实现正样本扩展
        np.random.seed(10)
        random_num = np.random.randint(1, 100, 10)
        print('random:', random_num)
        # 存储正样本和负样本对应的增强子和启动子
        pos_enhancers = enhancers[0:p]
        pos_promoters = promoters[0:p]

        neg_enhancers = enhancers[-n:]
        neg_promoters = promoters[-n:]

        # 扩展正样本(enhancer交换顺序，promoter不变)

        exmp = pos_enhancers[0]
        aug_enhancers = np.empty([p * 20, 3000, 4])
        aug_promoters = np.empty([p * 20, 2000, 4])
        # aug_enhancers = np.empty([p , 3000, 4])
        # aug_promoters = np.empty([p , 2000, 4])

        i = 0
        len_list=[]
        len_sample =len(pos_enhancers)
        for n_num in range(len_sample):
            len_list.append(n_num)
        random.seed(10)
        random.shuffle(len_list)
        print(len_list)
        for num_e in len_list:  # 0-1253
            exmp_e = pos_enhancers[num_e]  # 第num_e个正样本对应的enhancer
            exmp_p = pos_promoters[num_e]  # 第num_e个正样本对应的promoter
            aug_enhancers[i] = exmp_e
            aug_promoters[i] = exmp_p
            i += 1
            #
            for r in range(len(random_num)):
                new_sample = np.concatenate((exmp_e[random_num[r]:], exmp_e[0:random_num[r]]), axis=0)
                aug_enhancers[i] = new_sample
                aug_promoters[i] = exmp_p
                i += 1
            for s in range(1, len(random_num)):
                new_sample = np.concatenate((exmp_p[random_num[s]:], exmp_p[0:random_num[s]]), axis=0)
                aug_promoters[i] = new_sample
                aug_enhancers[i] = exmp_e
                i += 1
        print(len(random_num))
        print(aug_enhancers.shape)
        print(aug_promoters.shape)

        # 合并所有数据
        len_p = p
        len_aug_p = len(aug_enhancers)
        len_all = len_aug_p + n

        aug_enhancers_all = np.empty([len_all, 3000, 4])
        aug_promoters_all = np.empty([len_all, 2000, 4])
        aug_labels = np.empty([len_all, 1])

        number = 0
        # for i in range(len_aug_p):
        #     aug_enhancers_all[i] = aug_enhancers[i]
        #     aug_promoters_all[i] = aug_promoters[i]
        #     aug_labels[number] = 1
        #     number += 1
        #
        # for i in range(n):
        #     aug_enhancers_all[len_aug_p + i] = neg_enhancers[i]
        #     aug_promoters_all[len_aug_p + i] = neg_promoters[i]
        #     aug_labels[len_aug_p + i] = 0
        #     number += 1
        aug_enhancers_all=np.vstack((aug_enhancers, neg_enhancers))
        aug_promoters_all=np.vstack((aug_promoters, neg_promoters))
        aug_labels[:len_aug_p]=1
        aug_labels[len_aug_p:] = 0
        print(aug_enhancers_all.shape)
        print(aug_promoters_all.shape)
        print('正样本的EP数据拓展（finish!）')

        point_list = write_pixel_list_hilbert(6, file_name_hilbert)

        # dataset_promoter = np.zeros((len(aug_promoters_all), 64, 64, 4))
        # dataset_enhancer = np.zeros((len(aug_enhancers_all), 64, 64, 4))

        # 将正负样本转化为三维图像
        sample_count = 0
        names_list=[]
        label_list=[]
        labels = np.array(aug_labels)
        for k in tqdm(range(len(aug_enhancers_all))):
            if 'N' not in aug_enhancers_all[k] and 'N' not in aug_enhancers_all[k]:
                enhancer = make_image(aug_enhancers_all[k])
                promoter = make_image(aug_promoters_all[k])
                label = labels[k]
                name = cell + '_' + str(k)
                label_list.append(label[0])
                names_list.append(name)
                cell_enhancer = data_path + cell + '_e_' + str(k) + '.npy'
                cell_promoter = data_path + cell + '_p_' + str(k) + '.npy'
                np.save(cell_enhancer, enhancer)
                np.save(cell_promoter, promoter)
                sample_count += 1


        print(labels.shape)


        #for i in tqdm(range(len(dataset_enhancer))):
        df1=pd.DataFrame({'name':names_list,'label':label_list})
        df1_name=csv_path+cell+'normal'+'.csv'
        df1.to_csv(df1_name,index=False)




        # 将正样本拓展之后的，增强子-启动子-label数据存储到h5文件中
        # if not os.path.exists(file_name_hilbert_all):
        #     with h5py.File(file_name_hilbert_all) as f:
        #         f['enhancers'] = dataset_enhancer
        #         f['promoters'] = dataset_promoter
        #         f['labels'] = labels
        print(cell + ' finish!')
    print('over!')
