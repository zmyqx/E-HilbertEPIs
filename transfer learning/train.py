import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras import layers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping,LearningRateScheduler
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import math
from keras.utils import to_categorical
import random
from sklearn.utils import shuffle
cell_name = ['NHEK', 'IMR90', 'HUVEC', 'K562', 'GM12878', 'HeLa-S3']

#cell = "IMR90"
def load_enhance(name,root_path=r'K:/data_Hilbert/cell_npy/'):
    name_list=name.split('_')
    path=root_path+name_list[0]+'_e_'+name_list[1]+'.npy'
    loadData = np.load(path)
    return loadData

def load_promoter(name,root_path=r'K:/data_Hilbert/cell_npy/'):
    name_list=name.split('_')
    path=root_path+name_list[0]+'_p_'+name_list[1]+'.npy'
    loadData = np.load(path)
    return loadData

def load_enhance1(name,root_path=r'D:/data_Hilbert_1/cell_npy/'):
    name_list=name.split('_')
    path=root_path+name_list[0]+'_e_'+name_list[1]+'.npy'
    loadData = np.load(path)
    return loadData

def load_promoter1(name,root_path=r'D:/data_Hilbert_1/cell_npy/'):
    name_list=name.split('_')
    path=root_path+name_list[0]+'_p_'+name_list[1]+'.npy'
    loadData = np.load(path)
    return loadData

def npydata_load(X_samples,y_samples,batch_size=64):
    out_num = len(X_samples) //batch_size
    X_samples=X_samples[:batch_size*out_num]
    y_samples = y_samples[:batch_size*out_num]
    batch_size=len(X_samples)/batch_size
    X_batches = np.split(X_samples, batch_size)
    y_batches = np.split(y_samples, batch_size)

    while True:
        for b in range(len(X_batches)):
            x_e = np.array(list(map(load_enhance, X_batches[b])))
            x_p = np.array(list(map(load_promoter, X_batches[b])))
            y = np.array(y_batches[b])
            yield [x_e,x_p], y

def npydata_load1(X_samples,y_samples,batch_size=64):
    out_num = len(X_samples) //batch_size
    X_samples=X_samples[:batch_size*out_num]
    y_samples = y_samples[:batch_size*out_num]
    batch_size=len(X_samples)/batch_size
    X_batches = np.split(X_samples, batch_size)
    y_batches = np.split(y_samples, batch_size)
    while True:
        for b in range(len(X_batches)):
            x_e = np.array(list(map(load_enhance1, X_batches[b])))
            x_p = np.array(list(map(load_promoter1, X_batches[b])))
            y = np.array(y_batches[b])
            yield [x_e,x_p], y

# 打乱数据：dataset_promoter，dataset_enhancer，labels的顺序
def step_decay(epoch,lr):
  initial_lrate = lr
  drop = 0.5
  epochs_drop = 10
  lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
  lrate=np.clip(lrate,9e-6,3e-4)
  print(lrate)
  return lrate

def acc_self(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred > 0.5)

def shufff_list(a,b):
    c = list(zip(a, b))
    random.shuffle(c)
    a[:], b[:] = zip(*c)
    return a,b



# # 存储交叉验证的结果
# f1 = []
# auc = []
# aupr = []
# acc = []


def binary_weight_loss(y_true, y_pred):
    return -tf.reduce_mean(20*y_true * tf.log(tf.clip_by_value(y_pred, 1e-10, 1.0)) + (1 - y_true) * tf.log(tf.clip_by_value(1 - y_pred, 1e-10, 1.0)))


acc=[]
loss=[]
def result_visualization(history,modename,cell,root_path='E:\额外\model2'):

    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)

    plt.savefig("%s\\%s_%s_val_acc.png"%(root_path,modename,cell))
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)
    plt.savefig("%s\\%s_%s_val_loss.png"%(root_path,modename,cell))
    plt.show()


def train_cell(cell_name,root_path=r'G:/data_Hilbert/cell_csv/',npydata_load=npydata_load,batch_size=128,epochs=100,test=False,lr=3e-4):
    npydata_load = npydata_load
    cell=cell_name
    #path_name=root_path+cell_name+'.csv'
    path_name = root_path + cell_name+'normal' + '.csv'
    df1=pd.read_csv(path_name)

    label_1=df1[df1['label']==1]
    # label_1 = shuffle(label_1).reset_index(drop=True)
    len_1=int(len(label_1)//20*0.9)*20
    len_2 = int(len(label_1) // 20 * 0.8) * 20
    label_1_train=label_1.loc[:len_2]
    label_1_val=label_1.loc[len_2+1:len_1]
    label_1_test=label_1.loc[len_1+1::20]
    print(len(label_1_test))
    label_0=df1[df1['label']==0]
    label_0 = shuffle(label_0).reset_index(drop=True)

    len_0_1 = int(len(label_0) // 20 * 0.9) * 20
    len_0_2 = int(len(label_0) // 20 * 0.8) * 20
    label_0=label_0.reset_index(drop=True)
    label_0_train=label_0.loc[:len_0_2]
    label_0_val=label_0.loc[len_0_2+1:len_0_1]
    label_0_test=label_0.loc[len_0_1+1::20]
    train=pd.concat([label_1_train,label_0_train],ignore_index=False).reset_index(drop=True)
    val=pd.concat([label_1_val,label_0_val],ignore_index=False).reset_index(drop=True)
    test_1=pd.concat([label_1_test,label_0_test],ignore_index=False).reset_index(drop=True)
    print(train)
    # print(val)
    print(len(test_1))
    # name_list=df1['name'].values
    # label_list=df1['label'].values
    x_train=train['name'].values
    x_test = test_1['name'].values
    y_train=train['label'].values
    y_test=test_1['label'].values
    x_vaild=val['name'].values
    y_vaild=val['label'].values
    # label_list = to_categorical(label_list)
    x_train,y_train=shufff_list(x_train,y_train)
    #x_vaild, y_vaild = shufff_list(x_vaild, y_vaild)
    # name_list=name_list[:10000]
    # label_list=label_list[:10000]
    # x_train1, x_test, y_train1, y_test = train_test_split(
    #     name_list, label_list,
    #     test_size=0.1 ,random_state=2 # test_size默认是0.25
    # )
    # x_train, x_vaild, y_train, y_vaild = train_test_split(
    #     x_train1, y_train1,
    #     test_size=0.2 ,random_state=2 # test_size默认是0.2
    # )
    #
    # for X_train ,y_train in npydata_load(x_train, y_train):
    #     print(X_train[0][0])
    #     break
    # return

    for fold in range(1):  # 10
        print('fold %d' % fold)
        #
        enhancer_input = Input(shape=(64, 64, 4))
        enhancer_first_conv1 = Conv2D(64, (5, 5), activation='relu', padding='same', strides=4)(enhancer_input)  # 256
        enhancer_first_MaxPool1 = MaxPooling2D((3, 3), strides=2)(enhancer_first_conv1)
        enhancer_first_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same',strides=2)(enhancer_first_MaxPool1)  # 256
        enhancer_first_MaxPool3= layers.GlobalMaxPooling2D()(enhancer_first_conv2)
        #enhancer_first_MaxPool2 = MaxPooling2D((3, 3), strides=2)(enhancer_first_conv2)
        #enhancer_first_MaxPool3 = Flatten()(enhancer_first_MaxPool2)


        promoter_input = Input(shape=(64, 64, 4))
        promoter_first_conv1 = Conv2D(64, (5, 5), activation='relu',  padding='same',strides=4)(promoter_input)  # 128
        promoter_first_MaxPool1 = MaxPooling2D((3, 3), strides=2)(promoter_first_conv1)
        promoter_first_conv2 = Conv2D(128, (3, 3), activation='relu',  padding='same',strides=2)(promoter_first_MaxPool1)  # 128

        promoter_first_MaxPool3 = layers.GlobalMaxPooling2D()(promoter_first_conv2)
        #promoter_first_MaxPool2 = MaxPooling2D((3, 3), strides=2)(promoter_first_conv2)
        #promoter_first_MaxPool3 = Flatten()(promoter_first_MaxPool2)

        branch_output = layers.concatenate([enhancer_first_MaxPool3, promoter_first_MaxPool3], axis=1)
        branch_output1 = layers.Dropout(0.5)(branch_output)
        # branch_output2 = Dense(1024, activation='relu')(branch_output1)
        branch_output3 = Dense(128, activation='relu')(branch_output1)
        output = Dense(1, activation='sigmoid')(branch_output3)

        model = Model(inputs=[enhancer_input, promoter_input], outputs=[output])
        # enhancer_input = Input(shape=(64, 64, 4))
        # enhancer_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(enhancer_input)  # 256
        # enhancer_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(enhancer_first_conv1)
        # enhancer_first_MaxPool1 = Flatten()(enhancer_first_MaxPool1)
        #
        # promoter_input = Input(shape=(64, 64, 4))
        # promoter_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(promoter_input)  # 128
        # promoter_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(promoter_first_conv1)
        # promoter_first_MaxPool1 = Flatten()(promoter_first_MaxPool1)
        #
        # branch_output = layers.concatenate([enhancer_first_MaxPool1, promoter_first_MaxPool1], axis=1)
        # branch_output1 = layers.Dropout(0.5)(branch_output)
        # branch_output2 = Dense(128, activation='relu')(branch_output1)  # 128
        # output = Dense(1, activation='sigmoid')(branch_output2)
        #
        # model = Model(inputs=[enhancer_input, promoter_input], outputs=[output])



        print(model.summary())

        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['acc'])
        # model.compile(loss=binary_weight_loss, optimizer=optimizers.Adam(lr=lr), metrics=['acc'])  # 1e-5 RMSprop

        if not test:
            checkpointer = ModelCheckpoint(filepath="./model3/%s_bestmodel_fold%d.h5"
                                               % (cell, fold), verbose=1, save_best_only=True)
            lrate = LearningRateScheduler(step_decay)
            earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

            print('Training model...')
            m = model.fit_generator(npydata_load(x_train, y_train,batch_size), epochs=epochs,
                                    steps_per_epoch=max(1, len(x_train)//batch_size),# 128
                                    shuffle=True,
                                    validation_data=npydata_load(x_vaild, y_vaild,batch_size),
                                    validation_steps=max(1, len(x_vaild)//batch_size),
                                    callbacks=[lrate,checkpointer, earlystopper], verbose=1)
            result_visualization(m,'model3',cell)

        print('Testing model...')
        model.load_weights('./model3/%s_bestmodel_fold%d.h5' % (cell, fold))
        tresults = model.evaluate_generator(npydata_load(x_test, y_test,batch_size=1),steps=len(x_test)//1)
        print('test_results:', tresults)
        y_pred = model.predict_generator(npydata_load(x_test, y_test,batch_size=1),steps=len(x_test)//1)
        print(len(y_test))
        print(y_pred.shape)
        print('Calculating AUC...')
        y_pred=y_pred
        f1=metrics.f1_score(y_test, y_pred > 0.5)
        auc=metrics.roc_auc_score(y_test, y_pred)
        aupr=metrics.average_precision_score(y_test, y_pred)
        acc=metrics.accuracy_score(y_test, y_pred > 0.5)
    print(cell, 'result:')
    print(f1, auc, aupr, acc)
    print('---------------------------------')
    return f1, auc, aupr, acc,m

def train_cell_all(cell_name,modecell,root_path=r'D:/data_Hilbert/cell_csv/',npydata_load=npydata_load,batch_size=128,epochs=100,test=False,lr=3e-4):
    npydata_load = npydata_load
    cell = cell_name
    modecell=modecell
    f1 = 0
    auc = 0
    aupr = 0
    acc = 0
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    x_vaild = []
    y_vaild = []
    cell_all=cell_name
    for cell in cell_all:
        #path_name=root_path+cell_name+'.csv'
        path_name = root_path + cell+'normal' + '.csv'
        df1=pd.read_csv(path_name)

        label_1=df1[df1['label']==1]
        # label_1 = shuffle(label_1).reset_index(drop=True)
        len_1=int(len(label_1)//20*0.9)*20
        len_2 = int(len(label_1) // 20 * 0.8) * 20
        label_1_train=label_1.loc[:len_2]
        label_1_val=label_1.loc[len_2+1:len_1]
        label_1_test=label_1.loc[len_1+1::20]
        print(len(label_1_test))
        label_0=df1[df1['label']==0]
        label_0 = shuffle(label_0).reset_index(drop=True)

        len_0_1 = int(len(label_0) // 20 * 0.9) * 20
        len_0_2 = int(len(label_0) // 20 * 0.8) * 20
        label_0=label_0.reset_index(drop=True)
        label_0_train=label_0.loc[:len_0_2]
        label_0_val=label_0.loc[len_0_2+1:len_0_1]
        label_0_test=label_0.loc[len_0_1+1::20]
        train=pd.concat([label_1_train,label_0_train],ignore_index=False).reset_index(drop=True)
        val=pd.concat([label_1_val,label_0_val],ignore_index=False).reset_index(drop=True)
        test_1=pd.concat([label_1_test,label_0_test],ignore_index=False).reset_index(drop=True)
        print(train)
        # print(val)
        print(len(test_1))
        # name_list=df1['name'].values
        # label_list=df1['label'].values
        x_train.extend(train['name'].values)
        x_test.extend(test_1['name'].values)
        y_train.extend(train['label'].values)
        y_test.extend(test_1['label'].values)
        x_vaild.extend(val['name'].values)
        y_vaild.extend(val['label'].values)
    # label_list = to_categorical(label_list)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_vaild = np.array(x_vaild)
    y_vaild = np.array(y_vaild)
    x_train,y_train=shufff_list(x_train,y_train)
    # x_vaild, y_vaild = shufff_list(x_vaild, y_vaild)
    # name_list=name_list[:10000]
    # label_list=label_list[:10000]
    # x_train1, x_test, y_train1, y_test = train_test_split(
    #     name_list, label_list,
    #     test_size=0.1 ,random_state=2 # test_size默认是0.25
    # )
    # x_train, x_vaild, y_train, y_vaild = train_test_split(
    #     x_train1, y_train1,
    #     test_size=0.2 ,random_state=2 # test_size默认是0.2
    # )
    #
    # for X_train ,y_train in npydata_load(x_train, y_train):
    #     print(X_train[0][0])
    #     break
    # return

    for fold in range(1):  # 10
        print('fold %d' % fold)
        #
        enhancer_input = Input(shape=(64, 64, 4))
        enhancer_first_conv1 = Conv2D(64, (5, 5), activation='relu', padding='same', strides=4)(enhancer_input)  # 256
        enhancer_first_MaxPool1 = MaxPooling2D((3, 3), strides=2)(enhancer_first_conv1)
        enhancer_first_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(
            enhancer_first_MaxPool1)  # 256
        enhancer_first_MaxPool3 = layers.GlobalMaxPooling2D()(enhancer_first_conv2)
        # enhancer_first_MaxPool2 = MaxPooling2D((3, 3), strides=2)(enhancer_first_conv2)
        # enhancer_first_MaxPool3 = Flatten()(enhancer_first_MaxPool2)

        promoter_input = Input(shape=(64, 64, 4))
        promoter_first_conv1 = Conv2D(64, (5, 5), activation='relu', padding='same', strides=4)(promoter_input)  # 128
        promoter_first_MaxPool1 = MaxPooling2D((3, 3), strides=2)(promoter_first_conv1)
        promoter_first_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(
            promoter_first_MaxPool1)  # 128

        promoter_first_MaxPool3 = layers.GlobalMaxPooling2D()(promoter_first_conv2)
        # promoter_first_MaxPool2 = MaxPooling2D((3, 3), strides=2)(promoter_first_conv2)
        # promoter_first_MaxPool3 = Flatten()(promoter_first_MaxPool2)

        branch_output = layers.concatenate([enhancer_first_MaxPool3, promoter_first_MaxPool3], axis=1)
        branch_output1 = layers.Dropout(0.5)(branch_output)
        # branch_output2 = Dense(1024, activation='relu')(branch_output1)
        branch_output3 = Dense(128, activation='relu')(branch_output1)
        output = Dense(1, activation='sigmoid')(branch_output3)

        model = Model(inputs=[enhancer_input, promoter_input], outputs=[output])
        # enhancer_input = Input(shape=(64, 64, 4))
        # enhancer_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(enhancer_input)  # 256
        # enhancer_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(enhancer_first_conv1)
        # enhancer_first_MaxPool1 = Flatten()(enhancer_first_MaxPool1)
        #
        # promoter_input = Input(shape=(64, 64, 4))
        # promoter_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(promoter_input)  # 128
        # promoter_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(promoter_first_conv1)
        # promoter_first_MaxPool1 = Flatten()(promoter_first_MaxPool1)
        #
        # branch_output = layers.concatenate([enhancer_first_MaxPool1, promoter_first_MaxPool1], axis=1)
        # branch_output1 = layers.Dropout(0.5)(branch_output)
        # branch_output2 = Dense(128, activation='relu')(branch_output1)  # 128
        # output = Dense(1, activation='sigmoid')(branch_output2)
        #
        # model = Model(inputs=[enhancer_input, promoter_input], outputs=[output])

        print(model.summary())

        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['acc'])
        # model.compile(loss=binary_weight_loss, optimizer=optimizers.Adam(lr=lr), metrics=['acc'])  # 1e-5 RMSprop

        if not test:
            checkpointer = ModelCheckpoint(filepath="./model3/%s_bestmodel_fold%d.h5"
                                                    % (modecell, fold), verbose=1, save_best_only=True)
            lrate = LearningRateScheduler(step_decay)
            earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

            print('Training model...')
            #model.load_weights('./model3/%s_bestmodel_fold%d.h5' % (modecell, fold))
            m = model.fit_generator(npydata_load(x_train, y_train, batch_size), epochs=epochs,
                                    steps_per_epoch=max(1, len(x_train) // batch_size),  # 128
                                    shuffle=True,
                                    validation_data=npydata_load(x_vaild, y_vaild, batch_size),
                                    validation_steps=max(1, len(x_vaild) // batch_size),
                                    callbacks=[lrate, checkpointer, earlystopper], verbose=1)
            result_visualization(m, 'model3', cell)

        print('Testing model...')
        model.load_weights('./model3/%s_bestmodel_fold%d.h5' % (modecell, fold))
        tresults = model.evaluate_generator(npydata_load(x_test, y_test, batch_size=1), steps=len(x_test) // 1)
        print('test_results:', tresults)
        y_pred = model.predict_generator(npydata_load(x_test, y_test, batch_size=1), steps=len(x_test) // 1)
        print(len(y_test))
        print(y_pred.shape)
        print('Calculating AUC...')
        y_pred = y_pred
        f1 = metrics.f1_score(y_test, y_pred > 0.5)
        auc = metrics.roc_auc_score(y_test, y_pred)
        aupr = metrics.average_precision_score(y_test, y_pred)
        acc = metrics.accuracy_score(y_test, y_pred > 0.5)
    print(cell, 'result:')
    print(f1, auc, aupr, acc)
    print('---------------------------------')
    return f1, auc, aupr, acc


def  train_cell_1(cell_name,modecell=cell_name,root_path=r'D:/data_Hilbert/cell_csv/',npydata_load=npydata_load,batch_size=128,epochs=100,test=False,lr=3e-4):
    npydata_load = npydata_load
    cell = cell_name
    modecell=modecell
    f1 = 0
    auc = 0
    aupr = 0
    acc = 0
    # path_name=root_path+cell_name+'.csv'
    path_name = root_path + cell_name + 'normal' + '.csv'
    df1 = pd.read_csv(path_name)

    label_1 = df1[df1['label'] == 1]
    # label_1 = shuffle(label_1).reset_index(drop=True)
    len_1 = int(len(label_1) // 20 * 0.9) * 20
    len_2 = int(len(label_1) // 20 * 0.8) * 20
    label_1_train = label_1.loc[:len_2:]
    label_1_val = label_1.loc[len_2 + 1:len_1:20]
    label_1_test = label_1.loc[len_1 + 1::20]

    label_0 = df1[df1['label'] == 0]
    label_0 = shuffle(label_0).reset_index(drop=True)

    len_0_1 = int(len(label_0) // 20 * 0.9) * 20
    len_0_2 = int(len(label_0) // 20 * 0.8) * 20
    label_0 = label_0.reset_index(drop=True)
    label_0_train = label_0.loc[:len_0_2:]
    label_0_val = label_0.loc[len_0_2 + 1:len_0_1:20]
    label_0_test = label_0.loc[len_0_1 + 1::20]
    train = pd.concat([label_1_train, label_0_train], ignore_index=False).reset_index(drop=True)
    val = pd.concat([label_1_val, label_0_val], ignore_index=False).reset_index(drop=True)
    test_1 = pd.concat([label_1_test, label_0_test], ignore_index=False).reset_index(drop=True)
    print(train)
    # print(val)
    print(test_1)
    # name_list=df1['name'].values
    # label_list=df1['label'].values
    x_train = train['name'].values
    x_test = test_1['name'].values
    y_train = train['label'].values
    y_test = test_1['label'].values
    x_vaild = val['name'].values
    y_vaild = val['label'].values
    # label_list = to_categorical(label_list)
    x_train, y_train = shufff_list(x_train, y_train)
    # x_vaild, y_vaild = shufff_list(x_vaild, y_vaild)
    # name_list=name_list[:10000]
    # label_list=label_list[:10000]
    # x_train1, x_test, y_train1, y_test = train_test_split(
    #     name_list, label_list,
    #     test_size=0.1 ,random_state=2 # test_size默认是0.25
    # )
    # x_train, x_vaild, y_train, y_vaild = train_test_split(
    #     x_train1, y_train1,
    #     test_size=0.2 ,random_state=2 # test_size默认是0.2
    # )
    #
    # for X_train ,y_train in npydata_load(x_train, y_train):
    #     print(X_train[0][0])
    #     break
    # return

    for fold in range(1):  # 10
        print('fold %d' % fold)
        #
        enhancer_input = Input(shape=(64, 64, 4))
        enhancer_first_conv1 = Conv2D(64, (5, 5), activation='relu', padding='same', strides=4)(enhancer_input)  # 256
        enhancer_first_MaxPool1 = MaxPooling2D((3, 3), strides=2)(enhancer_first_conv1)
        enhancer_first_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(
            enhancer_first_MaxPool1)  # 256
        enhancer_first_MaxPool3 = layers.GlobalMaxPooling2D()(enhancer_first_conv2)
        # enhancer_first_MaxPool2 = MaxPooling2D((3, 3), strides=2)(enhancer_first_conv2)
        # enhancer_first_MaxPool3 = Flatten()(enhancer_first_MaxPool2)

        promoter_input = Input(shape=(64, 64, 4))
        promoter_first_conv1 = Conv2D(64, (5, 5), activation='relu', padding='same', strides=4)(promoter_input)  # 128
        promoter_first_MaxPool1 = MaxPooling2D((3, 3), strides=2)(promoter_first_conv1)
        promoter_first_conv2 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(
            promoter_first_MaxPool1)  # 128

        promoter_first_MaxPool3 = layers.GlobalMaxPooling2D()(promoter_first_conv2)
        # promoter_first_MaxPool2 = MaxPooling2D((3, 3), strides=2)(promoter_first_conv2)
        # promoter_first_MaxPool3 = Flatten()(promoter_first_MaxPool2)

        branch_output = layers.concatenate([enhancer_first_MaxPool3, promoter_first_MaxPool3], axis=1)
        branch_output1 = layers.Dropout(0.5)(branch_output)
        # branch_output2 = Dense(1024, activation='relu')(branch_output1)
        branch_output3 = Dense(128, activation='relu')(branch_output1)
        output = Dense(1, activation='sigmoid')(branch_output3)

        model = Model(inputs=[enhancer_input, promoter_input], outputs=[output])
        # enhancer_input = Input(shape=(64, 64, 4))
        # enhancer_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(enhancer_input)  # 256
        # enhancer_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(enhancer_first_conv1)
        # enhancer_first_MaxPool1 = Flatten()(enhancer_first_MaxPool1)
        #
        # promoter_input = Input(shape=(64, 64, 4))
        # promoter_first_conv1 = Conv2D(128, (3, 3), activation='relu', strides=1)(promoter_input)  # 128
        # promoter_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(promoter_first_conv1)
        # promoter_first_MaxPool1 = Flatten()(promoter_first_MaxPool1)
        #
        # branch_output = layers.concatenate([enhancer_first_MaxPool1, promoter_first_MaxPool1], axis=1)
        # branch_output1 = layers.Dropout(0.5)(branch_output)
        # branch_output2 = Dense(128, activation='relu')(branch_output1)  # 128
        # output = Dense(1, activation='sigmoid')(branch_output2)
        #
        # model = Model(inputs=[enhancer_input, promoter_input], outputs=[output])

        #print(model.summary())

        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['acc'])
        # model.compile(loss=binary_weight_loss, optimizer=optimizers.Adam(lr=lr), metrics=['acc'])  # 1e-5 RMSprop

        if not test:
            checkpointer = ModelCheckpoint(filepath="./model3/%s_bestmodel_fold%d.h5"
                                                    % (cell, fold), verbose=1, save_best_only=True)
            lrate = LearningRateScheduler(step_decay)
            earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

            print('Training model...')
            model.load_weights('./model3/%s_bestmodel_fold%d.h5' % (modecell, fold))
            # for layer in model.layers:
            #     layer.trainable = False
            # model.layers[-1].trainable=True
            # model.layers[-2].trainable = True
            # print(model.summary())
            m = model.fit_generator(npydata_load(x_train, y_train, batch_size), epochs=epochs,
                                    steps_per_epoch=max(1, len(x_train) // batch_size),  # 128
                                    shuffle=True,
                                    validation_data=npydata_load(x_vaild, y_vaild, batch_size),
                                    validation_steps=max(1, len(x_vaild) // batch_size),
                                    callbacks=[lrate, checkpointer, earlystopper], verbose=1)
            result_visualization(m, 'model3', cell)

        print('Testing model...')
        model.load_weights('./model3/%s_bestmodel_fold%d.h5' % (modecell, fold))
        tresults = model.evaluate_generator(npydata_load(x_test, y_test, batch_size=1), steps=len(x_test) // 1)
        print('test_results:', tresults)
        y_pred = model.predict_generator(npydata_load(x_test, y_test, batch_size=1), steps=len(x_test) // 1)
        print(len(y_test))
        print(y_pred.shape)
        print('Calculating AUC...')
        y_pred = y_pred
        f1 = metrics.f1_score(y_test, y_pred > 0.5)
        auc = metrics.roc_auc_score(y_test, y_pred)
        aupr = metrics.average_precision_score(y_test, y_pred)
        acc = metrics.accuracy_score(y_test, y_pred > 0.5)
    print(cell, 'result:')
    print(f1, auc, aupr, acc)
    print('---------------------------------')
    return f1, auc, aupr, acc

if __name__ == '__main__':
    cell_name = ['NHEK', 'IMR90', 'HUVEC', 'K562', 'GM12878', 'HeLa-S3']
    results=[]
    cell='HeLa-S3'
    # f1, auc, aupr, acc=train_cell_all(cell_name= ['NHEK', 'IMR90', 'HUVEC', 'K562', 'GM12878'],modecell='all',root_path=r'K:/data_Hilbert/cell_csv/',batch_size=128,epochs=8,test=False,lr=3e-4)
    # f1, auc, aupr, acc = train_cell_1(cell_name= 'HeLa-S3',  modecell='all', root_path=r'K:/data_Hilbert/cell_csv/',
    #                                   batch_size=128, epochs=20, test=False, lr=3e-4)
    for cell_mode in cell_name:
        for cell in cell_name:
            # f1, auc, aupr, acc,_=train_cell(cell,root_path=r'K:/data_Hilbert/cell_csv/',batch_size=128,epochs=5,test=True,lr=3e-4)
            # results.append([f1, auc, aupr, acc])
            # K.clear_session()
            f1, auc, aupr, acc=train_cell_1(cell_name=cell,modecell=cell_mode,root_path=r'K:/data_Hilbert/cell_csv/',batch_size=128,epochs=30,test=True,lr=3e-4)
            results.append([f1, auc, aupr, acc])
            K.clear_session()
    for i,cell_mode in enumerate(cell_name):
        print(cell_mode)
        print(results[i * 6:i * 6+6])
#['NHEK', 'IMR90', 'HUVEC', 'K562', 'GM12878', 'HeLa-S3']
#[0.9838584208479191, 0.9702476038338658, 0.9671484888304862, 0.9688765182186235, 0.9645986265687899, 0.9738505747126437]