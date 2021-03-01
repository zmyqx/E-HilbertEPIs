import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping,LearningRateScheduler
import pandas as pd
import math
from sklearn.model_selection import train_test_split



cell_name = ['NHEK', 'IMR90', 'HUVEC', 'K562', 'GM12878', 'HeLa-S3']

#cell = "IMR90"
def load_enhance(name,root_path=r'G:/data_Hilbert/cell_npy/'):
    name_list=name.split('_')
    path=root_path+name_list[0]+'_e_'+name_list[1]+'.npy'
    loadData = np.load(path)
    return loadData

def load_promoter(name,root_path=r'G:/data_Hilbert/cell_npy/'):
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

def step_decay(epoch):
  initial_lrate = 3e-5
  drop = 0.5
  epochs_drop = 10
  lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
  lrate=np.clip(lrate,3e-6,3e-4)
  print(lrate)
  return lrate


# 打乱数据：dataset_promoter，dataset_enhancer，labels的顺序


# 存储交叉验证的结果



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


def train_cell(cell_name,root_path=r'G:/data_Hilbert/cell_csv/',batch_size=128,epochs=100,test=False):
    f1 = []
    auc = []
    aupr = []
    acc = []
    cell=cell_name
    path_name=root_path+cell_name+'.csv'
    df1=pd.read_csv(path_name)
    name_list=df1['name'].values
    label_list=df1['label'].values
    # name_list=name_list[:10000]
    # label_list=label_list[:10000]
    x_train1, x_test, y_train1, y_test = train_test_split(
        name_list, label_list,
        test_size=0.1 ,random_state=2 # test_size默认是0.25
    )
    x_train, x_vaild, y_train, y_vaild = train_test_split(
        x_train1, y_train1,
        test_size=0.2 ,random_state=2 # test_size默认是0.2
    )

    # for X_train ,y_train in npydata_load(x_train, y_train):
    #     pass
    # seed = 7
    # np.random.seed(seed)

    for fold in range(1):  # 10
        print('fold %d' % fold)

        enhancer_input = Input(shape=(64, 64, 4))
        enhancer_first_conv1 = Conv2D(16, (3, 3), activation='relu', strides=1)(enhancer_input)  # 256
        enhancer_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(enhancer_first_conv1)
        enhancer_first_MaxPool1 = Flatten()(enhancer_first_MaxPool1)

        promoter_input = Input(shape=(64, 64, 4))
        promoter_first_conv1 = Conv2D(16, (3, 3), activation='relu', strides=1)(promoter_input)  # 128
        promoter_first_MaxPool1 = MaxPooling2D((2, 2), strides=(1, 1))(promoter_first_conv1)
        promoter_first_MaxPool1 = Flatten()(promoter_first_MaxPool1)

        branch_output = layers.concatenate([enhancer_first_MaxPool1, promoter_first_MaxPool1], axis=1)
        branch_output1 = layers.Dropout(0.5)(branch_output)
        branch_output2 = Dense(2048, activation='relu')(branch_output1)
        branch_output2 = layers.Dropout(0.5)(branch_output2)
        branch_output3 = Dense(128, activation='relu')(branch_output2) # 128
        output = Dense(1, activation='sigmoid')(branch_output3)

        model = Model(inputs=[enhancer_input, promoter_input], outputs=[output])
        print(model.summary())
        model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=3e-5), metrics=['acc'])



        if not test:
            checkpointer = ModelCheckpoint(filepath="./model2/%s_bestmodel_fold%d.h5"
                                                    % (cell, fold), verbose=1, save_best_only=True)
            lrate = LearningRateScheduler(step_decay)

            earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
            print('Training model...')
            m = model.fit_generator(npydata_load(x_train, y_train,batch_size), epochs=epochs,
                                    steps_per_epoch=max(1, len(x_train)//batch_size),# 128
                                    shuffle=True,
                                    validation_data=npydata_load(x_vaild, y_vaild,batch_size),
                                    validation_steps=max(1, len(x_vaild)//batch_size),
                                    callbacks=[checkpointer, earlystopper], verbose=1)
            result_visualization(m)

        print('Testing model...')
        model.load_weights('./model2/%s_bestmodel_fold%d.h5' % (cell, fold))
        tresults = model.evaluate_generator(npydata_load(x_test, y_test,batch_size=1),steps=len(x_test)//1)
        print('test_results:', tresults)
        y_pred = model.predict_generator(npydata_load(x_test, y_test,batch_size=1),steps=len(x_test)//1)
        print(len(y_test))
        print(y_pred.shape)
        print('Calculating AUC...')
        f1.append(metrics.f1_score(y_test, y_pred > 0.5))
        auc.append(metrics.roc_auc_score(y_test, y_pred))
        aupr.append(metrics.average_precision_score(y_test, y_pred))
        acc.append(metrics.accuracy_score(y_test, y_pred > 0.5))
    print(cell, 'result:')
    print(f1, auc, aupr, acc)
    print('---------------------------------')

if __name__ == '__main__':
    train_cell('NHEK',root_path=r'G:/data_Hilbert/cell_csv/',batch_size=128,epochs=150,test=False)