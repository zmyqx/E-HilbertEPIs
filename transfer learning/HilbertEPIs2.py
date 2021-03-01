import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py
import math

cell = "IMR90"  # IMR90 NHEK GM12878
file_name_cell = './data_Hilbert/' + cell + '_all_data_hilbert_aug.h5'
with h5py.File(file_name_cell, 'r') as hf:
    dataset_enhancer = np.array(hf.get('enhancers'))
    dataset_promoter = np.array(hf.get('promoters'))
    labels = np.array(hf.get('labels'))

# print(dataset_enhancer.shape)
# print(dataset_promoter.shape)
# print(labels.shape)

# 打乱数据：dataset_promoter，dataset_enhancer，labels的顺序
n = len(dataset_enhancer)
indices = np.arange(n)
np.random.shuffle(indices)
print(indices)
dataset_enhancer = dataset_enhancer[indices]
dataset_promoter = dataset_promoter[indices]
labels = labels[indices]
# print(labels[:10])


def step_decay(epoch):
  initial_lrate = 3e-4
  drop = 0.5
  epochs_drop = 2
  lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
  return lrate

# 存储交叉验证的结果
f1 = []
auc = []
aupr = []
acc = []


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


# 十折交叉验证
for fold in range(1):  # 10
    print('fold %d' % fold)
    kf = StratifiedKFold(n_splits=10)  # 十折
    x = 0
    for train_index, test_index in kf.split(dataset_enhancer, labels):
        if x == fold:
            break
        x += 1
    train_l = len(train_index)
    train_l = int(train_l * 70 / 90.)  # 训练集train，验证集val，测试集test(衡量最终模型性能)：70，20，10

    X_en_train = dataset_enhancer[train_index[:train_l]]  # 1956
    X_pr_train = dataset_promoter[train_index[:train_l]]
    y_train = labels[train_index[:train_l]]

    X_en_valid = dataset_enhancer[train_index[train_l:]]  # 245
    X_pr_valid = dataset_promoter[train_index[train_l:]]
    y_valid = labels[train_index[train_l:]]

    X_en_test = dataset_enhancer[test_index]  # 246
    X_pr_test = dataset_promoter[test_index]
    y_test = labels[test_index]

    enhancer_input = Input(shape=(64, 64, 4))
    enhancer_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(enhancer_input)  # 256
    enhancer_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(enhancer_first_conv1)
    enhancer_first_MaxPool1 = Flatten()(enhancer_first_MaxPool1)

    promoter_input = Input(shape=(64, 64, 4))
    promoter_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(promoter_input)  # 128
    promoter_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(promoter_first_conv1)
    promoter_first_MaxPool1 = Flatten()(promoter_first_MaxPool1)

    branch_output = layers.concatenate([enhancer_first_MaxPool1, promoter_first_MaxPool1], axis=1)
    branch_output1 = layers.Dropout(0.5)(branch_output)
    branch_output2 = Dense(128, activation='relu')(branch_output1)  # 128
    output = Dense(1, activation='sigmoid')(branch_output2)

    model = Model(inputs=[enhancer_input, promoter_input], outputs=[output])
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-5), metrics=['acc'])

    test = False
    if not test:
        checkpointer = ModelCheckpoint(filepath="./model/%s_bestmodel_fold.h5"
                                                % cell, verbose=1, save_best_only=True) # 1
        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)  # 10

        print('Training model...')
        m = model.fit([X_en_train, X_pr_train], y_train, epochs=80, batch_size=256,  # 128
                      shuffle=True,
                      validation_data=([X_en_valid, X_pr_valid], y_valid),
                      callbacks=[checkpointer, earlystopper], verbose=1)  # [checkpointer, earlystopper]
        result_visualization(m)

    print('Testing model...')
    model.load_weights('./model/%s_bestmodel.h5' % cell)
    tresults = model.evaluate([X_en_test, X_pr_test], y_test)
    print('test_results:', tresults)
    y_pred = model.predict([X_en_test, X_pr_test], 256, verbose=1)  # 128
    print('Calculating AUC...')
    f1.append(metrics.f1_score(y_test, y_pred > 0.5))
    auc.append(metrics.roc_auc_score(y_test, y_pred))
    aupr.append(metrics.average_precision_score(y_test, y_pred))
    acc.append(metrics.accuracy_score(y_test, y_pred > 0.5))
print(cell, 'result:')
print('---------------------------------')
print(f1, auc, aupr, acc)
print('f1:', f1)
print('auc:', auc)
print('aupr:', aupr)
print('acc:', acc)
print('---------------------------------')

# 学习率 Adam（1e-5） epochs=60 batch=256

# loss: 0.0456 - acc: 0.9942 - val_loss: 0.1288 - val_acc: 0.9678
# test_results: [0.11276571213389738, 0.9708466453674122]
# IMR90 result:
# [0.9712824547600316] [0.9873223285486443] [0.9788717109107354] [0.9708466453674122]


# loss: 0.0721 - acc: 0.9924 - val_loss: 0.1348 - val_acc: 0.9707
# test_results: [0.13911136576796448, 0.9716063787713676]
# NHEK result:
# [0.9719877206446661] [0.9897685599825716] [0.9825461683489476] [0.9716063788409179]

# loss: 0.0065 - acc: 1.0000 - val_loss: 0.1092 - val_acc: 0.9721
# test_results: [0.10808955731760007, 0.9701633909542979]
# GM12878 result:
# [0.9706840390879478] [0.9884536709110605] [0.9812977933738757] [0.9701633909542979]

# epoch = 80
# test_results: [0.15595150122436852, 0.9612619808306709]
# IMR90 result:
# ---------------------------------
# [0.9612465041949659] [0.9879237639553429] [0.98028214264714] [0.9612619808306709]