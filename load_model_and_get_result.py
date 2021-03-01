import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
import h5py

cell = 'IMR90'

file_name_cell = './data_Hilbert/' + cell + \
                 '_all_data_hilbert_without_aug.h5'
with h5py.File(file_name_cell, 'r') as hf:
    dataset_enhancer = np.array(hf.get('enhancers'))
    dataset_promoter = np.array(hf.get('promoters'))
    labels = np.array(hf.get('labels'))

# 打乱数据：dataset_promoter，dataset_enhancer，labels的顺序
n = len(dataset_enhancer)
indices = np.arange(n)
np.random.shuffle(indices)
dataset_enhancer = dataset_enhancer[indices]
dataset_promoter = dataset_promoter[indices]
labels = labels[indices]

# 存储交叉验证的结果
f1 = []
auc = []
aupr = []
acc = []


# 混淆矩阵绘制（和为数据总量）
def cm_plot(original_label, predict_label, pic=None):
    cm = confusion_matrix(original_label, predict_label)  # 由原标签和预测标签生成混淆矩阵
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues)  # 画混淆矩阵，配色风格使用cm.Blues
    plt.colorbar()  # 颜色标签
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
            # annotate主要在图形中添加注释
            # 第一个参数添加注释
            # 第二个参数是注释的内容
            # xy设置箭头尖的坐标
            # horizontalalignment水平对齐
            # verticalalignment垂直对齐
            # 其余常用参数如下：
            # xytext设置注释内容显示的起始位置
            # arrowprops 用来设置箭头
            # facecolor 设置箭头的颜色
            # headlength 箭头的头的长度
            # headwidth 箭头的宽度
            # width 箭身的宽度
    plt.ylabel('True label')  # 坐标轴标签
    plt.xlabel('Predicted label')  # 坐标轴标签
    plt.title('confusion matrix')
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.show()


# 混淆矩阵绘制（和为1）
def cmpic(y_true, y_pred):
    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=0)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    cm = confusion_matrix(y_true, y_pred)
    labels = np.arange(len(cm))
    tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(9, 6), dpi=120)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        # c = cm_normalized[y_val][x_val]
        c = cm[y_val][x_val]
        c = round(c / len(y_true), 2)
        if (c > 0):
            plt.text(x_val, y_val, c, color='black', fontsize=20, va='center', ha='center')
            # plt.text(x_val, y_val, "%0.2f" %(c,), color='red', fontsize=7, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # show confusion matrix
    plt.show()


def acu_curve(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


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


from keras.models import load_model

for fold in range(1):  # 10
    print('fold %d' % fold)
    kf = StratifiedKFold(n_splits=10)  # 十折
    x = 0
    for train_index, test_index in kf.split(dataset_enhancer, labels):
        #     for train_index, test_index in kf.split(dataset_enh, lab):
        if x == fold:
            break
        x += 1

    X_en_test = dataset_enhancer[test_index]  # 246
    X_pr_test = dataset_promoter[test_index]
    y_test = labels[test_index]
    print(X_en_test.shape)

    model = load_model('./model/%s_bestmodel_fold%d.h5' % (cell, fold))
    tresults = model.evaluate([X_en_test, X_pr_test], y_test)
    print('test_results:', tresults)
    y_pred = model.predict([X_en_test, X_pr_test], 100,
                           verbose=1)  # verbose日志显示模式  verbose=1
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    print(cell, 'result:')
    print('---------------------------------')
    print('f1:', metrics.f1_score(y_test, y_pred > 0.5))
    print('auc:', metrics.roc_auc_score(y_test, y_pred))
    print('aupr:', metrics.average_precision_score(y_test, y_pred))
    print('acc:', metrics.accuracy_score(y_test, y_pred > 0.5))

    cmpic(y_test, y_pred > 0.5)
    print(type(y_test))
    print(y_pred)
    print(type(y_pred))

# cell = 'IMR90'
#
# file_name_cell = './data_Hilbert/' + cell + \
#                  '_all_data_hilbert_without_aug.h5'
#
# with h5py.File(file_name_cell, 'r') as hf:
#     dataset_enhancer = np.array(hf.get('enhancers'))
#     dataset_promoter = np.array(hf.get('promoters'))
#     labels = np.array(hf.get('labels'))
#
# X_en_test = dataset_enhancer[20].reshape(1,64,64,4)
# X_pr_test = dataset_promoter[20].reshape(1,64,64,4)
# y_test = labels[20]
#
# from keras.models import load_model
#
# model = load_model('./model/%s_bestmodel.h5' % cell)
# y_pred = model.predict([X_en_test, X_pr_test])
# print(y_pred)
# tresults = model.evaluate([X_en_test, X_pr_test], y_test)
# print('test_results:', tresults)
# print(cell, 'result:')
# print('---------------------------------')
# print('f1:', metrics.f1_score(y_test, y_pred > 0.5))
# print('auc:', metrics.roc_auc_score(y_test, y_pred))
# print('aupr:', metrics.average_precision_score(y_test, y_pred))
# print('acc:', metrics.accuracy_score(y_test, y_pred > 0.5))
