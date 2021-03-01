from tqdm import tqdm


# 提取目标细胞系中的数据集
def get_target_sequence(cell):
    fin_enh = open('../data/' + cell + '/fa/pos_enhancer.fa', 'r')
    fin_pro = open('../data/' + cell + '/fa/pos_promoter.fa', 'r')
    enhancer = []
    promoter = []

    for line in fin_enh:
        if line[0] == '>':
            continue
        else:
            line = line.strip().lower()  # 全部转化为小写
            enhancer.append(line)
    for line in fin_pro:
        if line[0] == '>':
            continue
        else:
            line = line.strip().lower()  # 全部转化为小写
            promoter.append(line)
    return enhancer, promoter


# 最长公共子串
def find_public_string(s1, s2):
    flag = min(len(s1), len(s2))
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return mmax / flag


# 读取其他五个细胞系
def count_similary_from_celllines(cell, w):  # cell为细胞系，w为权重
    count_sim = 0
    enhancer, promoter = get_target_sequence(cell)  # 获取到另外一个细胞系的数据
    for i in tqdm(range(len(enh))):  # 遍历目标细胞系中的数据 len(enh)
        for j in range(len(enhancer)):  # 遍历另一个细胞系中的增强子序列
            count_E = find_public_string(enh[i], enhancer[j])
            if count_E >= w:
                count_P = find_public_string(pro[i], promoter[j])
                if count_P >= w:
                    count_sim += 1
                    print("count,index,min:", count_sim, j, min(count_E, count_P))
    return count_sim


target_cell = 'IMR90'
enh, pro = get_target_sequence(target_cell)

# cell_line = ['GM12878','IMR90','HeLa-S3','HUVCE','K562']
cell = 'HUVCE'
w = 0.5

count = count_similary_from_celllines(cell, w)
print(count)
