import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from operator import itemgetter
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import Sequential,  Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, cohen_kappa_score
from numpy import save,load

def init_variables():

    global df,x_test,y_test,best_model_path,df_origin, x_test_live, df_live_origin
    np.random.seed(2)
    tf.random.set_seed(2)
    num_features=196
    metatrader_dir="C:\\Users\\melgibson\\AppData\Roaming\\MetaQuotes\\Terminal\\6E837615CE50F086D7E2801AA8E2160A\\MQL5\\Files\\"
    f = open(metatrader_dir+"Parameters.txt","r")
    # print(f.readline().split(':')[1])
    # f.readline().split(':')[1]
    model_path = re.sub('[^A-Za-z0-9]+', '', f.readline().split(':')[1])

    best_model_path = os.path.join('.', 'best_models', model_path)

    df_origin = pd.read_csv(metatrader_dir+"Testing.csv")
    df=df_origin

    df_live_origin = pd.read_csv(metatrader_dir + "LiveTesting.csv")
    df_live=df_live_origin

    y_test = df['labels'].astype(np.int8).to_numpy()
    colums_needed = list(pd.read_csv(os.path.join(best_model_path, 'columns_needed.csv'), header=None).T.values[0])

    df = df[colums_needed]
    df_live=df_live[colums_needed]

    x_test = df.to_numpy()
    x_test_live = df_live.to_numpy()

def data_wrangling():
    global x_test,y_test, x_test_live
    my_imputer = SimpleImputer()
    x_test = my_imputer.fit_transform(x_test)
    x_test_live= my_imputer.fit_transform(x_test_live)
    mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?

    x_test = mm_scaler.fit_transform(x_test)
    x_test_live = mm_scaler.fit_transform(x_test_live)

    print("Shapes of train: {} {}".format( x_test.shape, y_test.shape))


def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        # print(type(x), type(x_temp), x.shape)
        x_temp[i] = np.reshape(x[i], (img_height, img_width))

    return x_temp

def reshape_arrays():
    global x_test, y_test , x_test_live

    one_hot_enc = OneHotEncoder(sparse=False, categories='auto')  # , categories='auto'
    y_test = one_hot_enc.fit_transform(y_test.reshape(-1, 1))

    dim = int(np.sqrt(196))
    x_test = reshape_as_image(x_test, dim, dim)
    x_test = np.stack((x_test,) * 3, axis=-1)

    x_test_live = reshape_as_image(x_test_live, dim, dim)
    x_test_live = np.stack((x_test_live,) * 3, axis=-1)



    print("Shapes of train:{} {}".format( x_test.shape, y_test.shape))

def f1_metric(y_true, y_pred):
    """
    this calculates precision & recall
    """

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # mistake: y_pred of 0.3 is also considered 1
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    # y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    # y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    # conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)
    # tf.Print(conf_mat, [conf_mat], "confusion_matrix")

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def check_baseline(pred, y_test):
    print("size of test set", len(y_test))
    e = np.equal(pred, y_test)
    print("TP class counts", np.unique(y_test[e], return_counts=True))
    print("True class counts", np.unique(y_test, return_counts=True))
    print("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
    print("baseline acc:", (holds/len(y_test)*100))

def evaluate_model():
    global model,x_test,y_test
    # save('x_test.npy', x_test)
    # save('y_test.npy', y_test)
    # x_test = load('x_test.npy')
    # y_test=load('y_test.npy')

    pred = model.predict(x_test)
    pred_classes = np.argmax(pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    check_baseline(pred_classes, y_test_classes)
    conf_mat = confusion_matrix(y_test_classes, pred_classes)
    print(conf_mat)
    labels = [0, 1, 2]
    # ax = sns.heatmap(conf_mat, xticklabels=labels, yticklabels=labels, annot=True)
    # ax.xaxis.set_ticks_position('top')
    f1_weighted = f1_score(y_test_classes, pred_classes, labels=None,
                           average='weighted', sample_weight=None)
    print("F1 score (weighted)", f1_weighted)
    print("F1 score (macro)", f1_score(y_test_classes, pred_classes, labels=None,
                                       average='macro', sample_weight=None))
    print("F1 score (micro)", f1_score(y_test_classes, pred_classes, labels=None,
                                       average='micro',
                                       sample_weight=None))  # weighted and micro preferred in case of imbalance
    # https://scikit-learn.org/stable/modules/model_evaluation.html#cohen-s-kappa --> supports multiclass; ref: https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
    print("cohen's Kappa", cohen_kappa_score(y_test_classes, pred_classes))

    prec = []
    for i, row in enumerate(conf_mat):
        prec.append(np.round(row[i] / np.sum(row), 2))
        print("precision of class {} = {}".format(i, prec[i]))
    print("precision avg", sum(prec) / len(prec))

init_variables()
data_wrangling()
reshape_arrays()

model = load_model(best_model_path,custom_objects={"f1_metric": f1_metric})
evaluate_model()

pred = model.predict(x_test_live)

prob=np.max(pred, axis=1)

pred_classes = np.argmax(pred, axis=1)

combo = np.stack((df_live_origin['date'].to_numpy(),prob, pred_classes), axis=1)
df = pd.DataFrame(combo)
pd.set_option('display.max_rows', 200)
df.to_csv('output_new.csv',header=False,index=False)
# newdf = df[(df.origin == "JFK") & (df.carrier == "B6")]


