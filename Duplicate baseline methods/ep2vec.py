#!/usr/bin/env python
# encoding: utf-8

# system modules
import os
import time
import sys
# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
# random shuffle
from random import shuffle
# numpy
import numpy
# classifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

enhancers_num = 0
promoters_num = 0
positive_num = 0
negative_num = 0

# 分别读取控制台输入的kmer、步长、特征向量的维度、细胞系名称     python ep2vec.py 6 1 100 K562
kmer = 6
swin = 1
vlen = 300
cl = 'GM12878'
# kmer = int(sys.argv[1])  # the length of k-mer 6
# swin = int(sys.argv[2])  # the length of stride 1
# vlen = int(sys.argv[3])  # the dimension of embedding vector 100 每条序列都用100维的特征向量表示
# cl = sys.argv[4]  # the interested cell line

print("k:", kmer, "stride:", swin, "embedding vector size:", vlen, "cell line", cl)


# bed2sent: convert the enhancers.bed and promoters.bed to kmer sentense
# bed format: chr start end name
def bed2sent(filename, k, win):
    print('第一步：结合enhancer.fa和promoter.fa，并将enhancers.bed和promoters.bed转化成kmer形式，存储到.sent文件中')
    print('filename:', filename)
    if not os.path.isfile('./data/' + cl + '/' + filename + '.fa'):
        os.system(  # bedtools getfasta 根据bed文件的信息，从fasta中提取指定位置的DNA序列, -fi指定具体位置
            "bedtools getfasta -fi ./data/hg19/hg19.fa -bed " + './data/' + cl + "/" + filename + ".bed "
            + "-fo " + './data/' + cl + "/" + filename + ".fa")
        time.sleep(30)
    fin = open('./data/' + cl + '/' + filename + '.fa', 'r')  # 存储hg19的基因序列
    fout = open('./data/' + cl + '/' + filename + '_' + str(k) + '_' +
                str(swin) + '.sent', 'w')  # 存储kmer处理后的基因序列
    for line in fin:
        if line[0] == '>':
            continue
        else:
            line = line.strip().lower()  # 全部转化为小写
            length = len(line)
            i = 0
            while i <= length - k:
                fout.write(line[i:i + k] + ' ')  # 步长为k处理序列
                i = i + win
            fout.write('\n')


# generateTraining: extract the training set from pairs.csv and output the training pair with sentence
def generate_training():
    print('第二步：从XXXtrain.csv中提取训练集，结合.bed信息，并输出enhance_index,promoter_index,相互作用否，存到training.txt')
    global enhancers_num, promoters_num, positive_num, negative_num
    fin1 = open('./data/' + cl + '/' + 'enhancers.bed', 'r')
    fin2 = open('./data/' + cl + '/' + 'promoters.bed', 'r')
    enhancers = []  # 存储所有的enhancer信息
    promoters = []  # 存储所有的promoter信息
    for line in fin1:
        data = line.strip().split()
        enhancers.append(data[3])  # eg: K562|chr1:118400-118670
        enhancers_num = enhancers_num + 1  # 存储增强子的数量
    for line in fin2:
        data = line.strip().split()
        promoters.append(data[3])  # eg: K562|chr1:713800-714772
        promoters_num = promoters_num + 1  # 存储启动子的数量
    fin3 = open('./data/' + cl + 'train.csv', 'r')  # csv文件会自动增加一列index
    fout = open('./data/' + cl + '/' + 'training.txt', 'w')
    for line in fin3:
        if line[0] == 'b':
            continue
        else:
            data = line.strip().split(',')
            # 找到enhancer_name，存储enhancer对应的index（所在行）
            enhancer_index = enhancers.index(data[5])
            # 对应promoter_name，存储promoter对应的index（所在行）
            promoter_index = promoters.index(data[10])
            fout.write(str(enhancer_index) + '\t' +
                       str(promoter_index) + '\t' + data[7] + '\n')
            # 存储对应的增强子index,启动子index，以及它们是否发生相互作用
            if data[7] == '1':
                positive_num = positive_num + 1  # 存储label为1的数据数量
            else:
                negative_num = negative_num + 1  # 存储label为0的数据数量


# convert the sentence to doc2vec's tagged sentence 将句子转化成doc2vec这种带标签的句子
class TaggedLineSentence(object):
    def __init__(self, sources):  # 初始化构造方法
        # eg: {'GM12878/promoters_6_1.sent': 'PROMOTERS'}
        self.sources = sources

        flipped = {}

        # make sure that keys are unique 保证keys是唯一的的
        for key, value in sources.items():  # {words=[u'ggtgat',....,u'ctcata'],tags:['PROMOTERS_n']} (n为编号)
            if value not in flipped:
                flipped[value] = [key]  # 存储为字典
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):  # 迭代器
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(
                        utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


# train the embedding vector from the file  从文件中训练嵌入向量
def doc2vec(name, k, vlen):  # name：'enhancers'，'promoters'，k=6，vlen：100
    print('第三步：训练文件中的嵌入向量')
    print('name,k,vlen:' + name, k, vlen)
    filename = './data/' + cl + '/' + name + '_' + \
               str(k) + '_' + str(swin) + '.sent'  # kmers处理后的序列
    indexname = name.upper()  # 'promoters'/'enhancers'转换为大写
    sources = {filename: indexname}
    print('sources', sources)

    # 返回带标签的文本
    sentences = TaggedLineSentence(sources)  # 输出：TaggedDocument(words=[.sent里面的kmer行序列],tags=['PROMOTTERS_编号'])
    # print('sentences:', type(sentences.to_array()), len(sentences.to_array()[0][0]), sentences.to_array()[0])  # 即：[句子，句子序号] 93
    # s = (sentences.to_array())[:3]

    # 【初始化模型并创建词汇表】构建Doc2vec模型：获得文档的一个固定长度的向量表达（数据：多个文档，以及他们的标签）
    model = Doc2Vec(min_count=1,  # min_count：对字典做截断，低于该次数的单词会被丢弃
                    window=10,  # window：窗口大小,表示当前词与预测词在一个句子中的最大距离
                    vector_size=vlen,  # vector_size：词向量长度,样例中设置为100维
                    sample=1e-4,  # 高频词汇的随机降采样的配置阈值，范围(0,1e-5)，默认1e-3
                    negative=5,  # negative：如果>0，则会采用negative sampling，设置有多少个噪声单词(5-20)
                    workers=8)  # 控制训练的并行数

    # model.build_vocab(documents=s)
    model.build_vocab(documents=sentences.to_array())  # 从一系列文档中构建词汇表,
    print('语料库句子数量：', model.corpus_count)  # 语料库（句子）的数量，同tags编号数量
    print('语料库词汇数量：', model.corpus_total_words)  # 语料的总数量，例如：拆分为'******'（6位数编码）的数量
    # print(model.docvecs[2], len(model.docvecs))
    # print('ok!')

    for epoch in range(10):  # 【训练Doc2vec模型】，迭代次数为10次
        print('epoch:', epoch)
        model.train(sentences, total_examples=model.corpus_count,  # models.corpus_count说明和models.build_vocab用同种语料库
                    epochs=model.epochs)  # epochs迭代次数，默认5
        # model.train(sentences.sentences_perm())
    # print('语料库句子数量：', model.corpus_count)  # 语料库（句子）的数量，同tags编号数量
    # print('语料库词汇数量：', model.corpus_total_words)  # 语料的总数量，例如：拆分为'******'（6位数编码）的数量
    # print(model.docvecs[2], len(model.docvecs))
    model.save('./data/' + cl + '/' + name + '_' + str(k) + '_' +
               str(swin) + '_' + str(vlen) + '.d2v')


# train the model and print the result 样例：6 100
def train(k, vlen):
    print('第四步：训练模型，打印结果')
    # global enhancers_num, promoters_num, positive_num, negative_num
    enhancers_num, promoters_num, positive_num, negative_num = 100036, 8453, 2113, 2110
    print(enhancers_num, promoters_num, positive_num, negative_num)
    enhancer_model = Doc2Vec.load(  # d2v存储的是所有enhancers的向量化（100维）表示，数量为enhancer数量
        './data/' + cl + '/' + 'enhancers_' + str(k) + '_' + str(swin) + '_' + str(vlen) + '.d2v')
    promoter_model = Doc2Vec.load(
        './data/' + cl + '/' + 'promoters_' + str(k) + '_' + str(swin) + '_' + str(vlen) + '.d2v')

    num = positive_num + negative_num  # EPIs样本的总条数
    print('EPIs样本总条数:', num)
    arrays = numpy.zeros((num, vlen * 2))  # num行，200列
    labels = numpy.zeros(num)
    print('shape(arrays,labels):', type(arrays), type(labels))

    fin = open('./data/' + cl + '/' + 'training.txt', 'r')
    print('读取训练文件：' + './data/' + cl + '/' + 'training.txt')
    i = 0
    for line in fin:
        data = line.strip().split()
        prefix_enhancer = 'ENHANCERS_' + data[0]
        prefix_promoter = 'PROMOTERS_' + data[1]

        enhancer_vec = enhancer_model.docvecs[prefix_enhancer]
        promoter_vec = promoter_model.docvecs[prefix_promoter]

        enhancer_vec = enhancer_vec.reshape((1, vlen))
        promoter_vec = promoter_vec.reshape((1, vlen))
        # XXXX_vec的维数从(100,)变成了(1,100)

        arrays[i] = numpy.column_stack((enhancer_vec, promoter_vec))  # (4223,200)

        labels[i] = int(data[2])  # (4223,)
        i = i + 1
    labels = labels.astype('int')  # 将label从float类型转化为int型
    print('shape(arrays,labels)：', arrays.shape, labels.shape)
    print('shape(arrays[0],labels[0])：', arrays[0].shape, labels[0].shape)

    estimator = GradientBoostingClassifier(n_estimators=4000, learning_rate=0.001, max_depth=25, max_features='log2',
                                           random_state=0)
    # cv = StratifiedKFold(y=labels, n_folds=10, shuffle=True, random_state=0)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    scores = cross_val_score(estimator=estimator, X=arrays, y=labels,  # 为数据集进行指定次数的交叉验证，并为每次效果测评
                             scoring='f1', cv=cv, n_jobs=-1)  # 输出值有10个（10折验证），cv指定int型时，默认Kfold打乱
    print('f1:')
    print('{:2f} {:2f}'.format(scores.mean(), scores.std()))  # 平均分、标准偏差数
    scores = cross_val_score(estimator=estimator, X=arrays, y=labels,
                             scoring='roc_auc', cv=cv, n_jobs=-1)
    print('rod_auc:')
    print('{:2f} {:2f}'.format(scores.mean(), scores.std()))
    scores = cross_val_score(estimator=estimator, X=arrays, y=labels,
                             scoring='average_precision', cv=cv, n_jobs=-1)
    print('average_precision:')
    print('{:2f} {:2f}'.format(scores.mean(), scores.std()))


# bed2sent("promoters", kmer, swin)  # 样例：6
# bed2sent("enhancers", kmer, swin)  # 样例：1
# print('pre process done!')
# generate_training()
# print('generate training set done!')
# doc2vec("promoters", kmer, vlen)  # 样例：100
# doc2vec("enhancers", kmer, vlen)  # 样例：100
# print('doc2vec done!')
train(kmer, vlen)
print('finish!')
