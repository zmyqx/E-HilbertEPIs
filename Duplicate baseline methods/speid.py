#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from keras.preprocessing.text import Tokenizer  # 文本预处理模块
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input
from keras.layers import Conv1D, MaxPooling1D, Reshape, Permute
from keras.layers import Activation, BatchNormalization, concatenate
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
from keras.models import Model

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

np.random.seed(47)

import sys


# 将结果可视化绘制
def result_visualization(history):
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.show()


# i = int(sys.argv[1])
# names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK']
names = ['GM12878']
# name = names[i]
for name in names:
    print('Experiment on %s dataset' % name)

    print('Loading seq data...')
    pos_enhancers = open('./data/%s/fa/pos_enhancer.fa' % name, 'r').readlines()[1::2]
    pos_promoters = open('./data/%s/fa/pos_promoter.fa' % name, 'r').readlines()[1::2]
    neg_enhancers = open('./data/%s/fa/neg_enhancer.fa' % name, 'r').readlines()[1::2]
    neg_promoters = open('./data/%s/fa/neg_promoter.fa' % name, 'r').readlines()[1::2]

    MAX_LEN_en = 3000  # 不足的进行长度填充，长度超过的进行切分
    MAX_LEN_pr = 2000
    print('max_length(enhancer,promoter):', MAX_LEN_en, MAX_LEN_pr)

    pos_enhancers = [' '.join(enhancer[:-1].lower()) for enhancer in pos_enhancers]  # 词编码的时候需要这个空格
    neg_enhancers = [' '.join(enhancer[:-1].lower()) for enhancer in neg_enhancers]
    pos_promoters = [' '.join(promoter[:-1].lower()) for promoter in pos_promoters]
    neg_promoters = [' '.join(promoter[:-1].lower()) for promoter in neg_promoters]
    enhancers = pos_enhancers + neg_enhancers  # 增强子序列拼接，前面是positive，后面是negative
    promoters = pos_promoters + neg_promoters
    print('seq_num(enhancer,promoter):', len(enhancers))
    print('(pos,neg):', len(pos_enhancers), len(neg_enhancers))
    assert len(pos_enhancers) == len(pos_promoters)
    assert len(neg_enhancers) == len(neg_promoters)
    y = np.array([1] * len(pos_enhancers) + [0] * len(neg_enhancers))

    print('Tokenizing seqs...(文本及序列预处理)')  # 统一序列长度，不足长度的用0尾部填充
    NB_WORDS = 5
    # Tokenizer类：对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示
    # 两种方法向量化一个语料库：将每个文本转化为整数序列（每个整数都是词典中标记的索引）；将其转化为一个向量
    tokenizer = Tokenizer(num_words=NB_WORDS)  # num_words 处理的最大单词数量：acgtn(5)
    tokenizer.fit_on_texts(enhancers)  # 使用一系列文档来生成token词典，test为list类，每个元素为一个文档

    # 将多个文档转化为word下标的向量形式,shape为[len(texts)，len(text)](文档数，每条文档的长度)
    sequences = tokenizer.texts_to_sequences(enhancers)
    X_en = pad_sequences(sequences, maxlen=MAX_LEN_en, padding='post')
    print('X_en[0]:', X_en[0])

    sequences = tokenizer.texts_to_sequences(promoters)
    X_pr = pad_sequences(sequences, maxlen=MAX_LEN_pr, padding='post')
    print('所有word对应的编号id，从1开始：', tokenizer.word_index)

    acgt_index = tokenizer.word_index  # a,t,c,g四种
    print('Found %s unique tokens.' % len(acgt_index))

    print('Spliting train, valid, test parts...(拆分训练集，验证集，测试集,打乱数据的顺序)')
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    X_en = X_en[indices]
    X_pr = X_pr[indices]
    y = y[indices]

    f1 = []
    auc = []
    aupr = []
    acc = []

    print('进入循环')
    for fold in range(10):  # 10
        print('fold %d' % fold)
        kf = StratifiedKFold(n_splits=10)  # 十折
        x = 0
        for train_index, test_index in kf.split(X_en, y):
            if x == fold:
                break
            x += 1
        train_l = len(train_index)
        train_l = int(train_l * 85 / 90.)  # 训练集train，验证集val，测试集test(衡量最终模型性能)：85，5，10

        X_en_train = X_en[train_index[:train_l]]
        X_pr_train = X_pr[train_index[:train_l]]
        y_train = y[train_index[:train_l]]

        X_en_valid = X_en[train_index[train_l:]]
        X_pr_valid = X_pr[train_index[train_l:]]
        y_valid = y[train_index[train_l:]]

        X_en_test = X_en[test_index]
        X_pr_test = X_pr[test_index]
        y_test = y[test_index]

        embedding_vector_length = 4
        nb_words = min(NB_WORDS, len(acgt_index))  # kmer_index starting from 1（NB_WORDS-5, acgt_index-4）

        print('Building model...')
        print('fix embedding layer with one-hot vectors')  # one-hot编码
        acgt2vec = {'a': np.array([1, 0, 0, 0], dtype='float32'),
                    'c': np.array([0, 1, 0, 0], dtype='float32'),
                    'g': np.array([0, 0, 1, 0], dtype='float32'),
                    't': np.array([0, 0, 0, 1], dtype='float32'),
                    'n': np.array([0, 0, 0, 0], dtype='float32')}

        embedding_matrix = np.zeros((nb_words + 1, embedding_vector_length))
        print('embedding_matrix_shape:', embedding_matrix.shape)

        for acgt, i in acgt_index.items():
            # print("(acgt,i):", acgt, i)
            if i > NB_WORDS:
                continue
            vector = acgt2vec.get(acgt)
            # print('vector:', vector)
            if vector is not None:
                embedding_matrix[i] = vector
        print('embedding_matrix:', embedding_matrix)

        print('(en_train_shape,pr_train_shape):', X_en_train.shape, X_pr_train.shape)  # (3588, 3000) (3588, 2000)

        enhancer_input = Input(shape=(MAX_LEN_en,))  # 希望输入的是(MAX_LEN_en)维向量
        # Embedding 将正整数(索引值)转化为固定尺寸的稠密向量，只能作为模型第一层
        # 输入尺寸：(batch_size,sequence_length)的2D张量
        # 输出尺寸：(batch_size,sequence_length,output_dim)的3D张量
        # eg：enhancer (m,3000) ->(m,30000,4)
        enhancer_out = Embedding(input_dim=nb_words + 1,  # 词汇表的大小，即最大整数index+1。若编码0-9，词汇大小为10
                                 output_dim=embedding_vector_length,  # 词向量的维度
                                 input_length=MAX_LEN_en,  # 输入序列的长度，一次输入带有词汇的个数（固定值）
                                 trainable=False,
                                 weights=[embedding_matrix])(enhancer_input)
        # enhancer_out = Reshape((1, MAX_LEN_en, embedding_vector_length))(enhancer_out)
        # enhancer_out = Conv2D(filters=1024, kernel_size=40, activation='relu')(enhancer_out)

        enhancer_out = Conv1D(filters=1024,  # 输出空间的维度（卷积中滤波器的输出数量）
                              kernel_size=40,  # 卷积窗口长度
                              strides=1,  # 卷积步长
                              activation='relu'  # 要使用的激活函数（未指定则使用线性激活）
                              )(enhancer_out)
        enhancer_out = MaxPooling1D(pool_size=20,  # 最大池化的窗口大小
                                    strides=20  # 缩小比例的因数，若不指定，同pool_size
                                    )(enhancer_out)
        enhancer_out = Reshape((1024, -1))(enhancer_out)

        promoter_input = Input(shape=(MAX_LEN_pr,))
        promoter_out = Embedding(input_dim=nb_words + 1,
                                 output_dim=embedding_vector_length,
                                 input_length=MAX_LEN_pr,
                                 trainable=False,
                                 weights=[embedding_matrix]
                                 )(promoter_input)
        # promoter_out = Reshape((1, MAX_LEN_pr, embedding_vector_length))(promoter_out)
        promoter_out = Conv1D(filters=1024,
                              kernel_size=40,
                              activation='relu'
                              )(promoter_out)
        # promoter_out = Conv2D(filters=1024, kernel_size=40, activation='relu')(promoter_out)
        promoter_out = MaxPooling1D(pool_size=20,
                                    strides=20
                                    )(promoter_out)
        promoter_out = Reshape((1024, -1))(promoter_out)  # 输出尺寸：(batch_size,) + target_shape

        concat_out = concatenate(inputs=[enhancer_out, promoter_out], axis=-1)  # 返回所有输入张量通过axis轴串联起来的输出张量

        concat_out = Permute(dims=(2, 1))(concat_out)  # 根据给定的模式置换输入的维度。置换模式，索引从1开始。(2,1)置换维度1和2
        # 批量标准化层：在每一个批次的数据中标准化前一层的激活项，即，应用一个维持激活项平均值接近0，标准差接近1的转换
        concat_out = BatchNormalization()(concat_out)
        concat_out = Dropout(0.5)(concat_out)  # 在训练中每次更新时，将输入单元的按比率随机设置为0,这有助于防止过拟合
        concat_out = Bidirectional(  # RNN的双向封装器，对序列进行前向和后向计算
            LSTM(units=100,  # units输出空间的维度
                 return_sequences=False),  # return_sequences: 布尔值。是返回输出序列中的最后一个输出，还是全部序列
            merge_mode='concat'  # 前向和后向RNN的结合模式，{'sum','mul','concat','ave',None} ，None输出不被结合，作为列表返回
        )(concat_out)
        concat_out = BatchNormalization()(concat_out)
        concat_out = Dropout(rate=0.5)(concat_out)

        out = Dense(units=925)(concat_out)  # 全连接层。units：输出空间的维度
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(1, activation='sigmoid')(out)
        model = Model(inputs=[enhancer_input, promoter_input], outputs=[out])
        print(model.summary())
        model.compile(optimizer=optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

        test = False
        if not test:
            checkpointer = ModelCheckpoint(filepath="./SPEID/model/%s_bestmodel_fold%d.h5"
                                                    % (name, fold), verbose=1, save_best_only=True)
            earlystopper = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

            print('Training model...')
            m = model.fit([X_en_train, X_pr_train], y_train, epochs=32, batch_size=128,  # epochs = 60
                          shuffle=True,
                          validation_data=([X_en_valid, X_pr_valid], y_valid),
                          callbacks=[checkpointer, earlystopper],
                          verbose=1)
            result_visualization(m)

        print('Testing model...')
        model.load_weights('./SPEID/model/%s_bestmodel_fold%d.h5' % (name, fold))
        tresults = model.evaluate([X_en_test, X_pr_test], y_test)
        print('test_results:', tresults)
        y_pred = model.predict([X_en_test, X_pr_test], 100, verbose=1)  # verbose日志显示模式
        print('Calculating AUC...')
        f1.append(metrics.f1_score(y_test, y_pred > 0.5))
        auc.append(metrics.roc_auc_score(y_test, y_pred))
        aupr.append(metrics.average_precision_score(y_test, y_pred))
        acc.append(metrics.accuracy_score(y_test, y_pred > 0.5))

    print(name, 'result:')
    print('---------------------------------')
    print(f1, auc, aupr, acc)
    print(np.mean(f1), np.std(f1))
    print(np.mean(auc), np.std(auc))
    print(np.mean(aupr), np.std(aupr))
    print(np.mean(acc), np.std(acc))
    print('---------------------------------')

print('ok')
