import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py

cell = "IMR90"


file_name_cell = './data_Hilbert/' + cell + '_all_data_hilbert_without_aug.h5'
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
# print(indices)
dataset_enhancer = dataset_enhancer[indices]
dataset_promoter = dataset_promoter[indices]
labels = labels[indices]
# print(labels[:10])

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
        #     for train_index, test_index in kf.split(dataset_enh, lab):
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
    branch_output2 = Dense(128, activation='relu')(branch_output)
    output = Dense(1, activation='sigmoid')(branch_output2)

    model = Model(inputs=[enhancer_input, promoter_input], outputs=[output])
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-5), metrics=['acc'])  # 1e-5 RMSprop

    test = False
    if not test:
        checkpointer = ModelCheckpoint(filepath="./model/%s_bestmodel_fold%d.h5"
                                                % (cell, fold), verbose=1, save_best_only=True)
        #          earlystopper = EarlyStopping(monitor='val_loss', patience=50, verbose=1)

        print('Training model...')
        m = model.fit([X_en_train, X_pr_train], y_train, epochs=20, batch_size=128,  # 128
                      shuffle=True,
                      validation_data=([X_en_valid, X_pr_valid], y_valid),
                      callbacks=[checkpointer], verbose=1)  # [checkpointer, earlystopper]
        result_visualization(m)

    print('Testing model...')
    model.load_weights('./model/%s_bestmodel_fold%d.h5' % (cell, fold))
    tresults = model.evaluate([X_en_test, X_pr_test], y_test)
    print('test_results:', tresults)
    y_pred = model.predict([X_en_test, X_pr_test], 100, verbose=1)  # verbose日志显示模式  verbose=1
    # print(y_pred)
    print('Calculating AUC...')
    f1.append(metrics.f1_score(y_test, y_pred > 0.5))
    auc.append(metrics.roc_auc_score(y_test, y_pred))
    aupr.append(metrics.average_precision_score(y_test, y_pred))
    acc.append(metrics.accuracy_score(y_test, y_pred > 0.5))
print(cell, 'result:')
print('---------------------------------')
print(f1, auc, aupr, acc)
print(np.mean(f1), np.std(f1))
print(np.mean(auc), np.std(auc))
print(np.mean(aupr), np.std(aupr))
print(np.mean(acc), np.std(acc))
print('---------------------------------')
