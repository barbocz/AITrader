import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from numpy.random import seed
import os


best_model_path = os.path.join('', 'best_models','GBPUSD1')

np.random.seed(2)

df_orig = pd.read_csv( os.path.join(best_model_path,'Testing.csv'))
colums_needed=list(pd.read_csv(os.path.join(best_model_path,'columns_needed.csv'),header=None).T.values[0])
df=df_orig[colums_needed]
data = df.to_numpy()

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()


x_test = data.copy()
x_test = my_imputer.fit_transform(x_test)


mm_scaler = MinMaxScaler(feature_range=(0, 1)) # or StandardScaler?
data = mm_scaler.fit_transform(data)
x_test = mm_scaler.transform(x_test)

from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_custom_objects

def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        # print(type(x), type(x_temp), x.shape)
        x_temp[i] = np.reshape(x[i], (img_height, img_width))

    return x_temp

def f1_weighted(y_true, y_pred):
    y_true_class = tf.math.argmax(y_true, axis=1, output_type=tf.dtypes.int32)
    y_pred_class = tf.math.argmax(y_pred, axis=1, output_type=tf.dtypes.int32)
    conf_mat = tf.math.confusion_matrix(y_true_class, y_pred_class)  # can use conf_mat[0, :], tf.slice()
    # precision = TP/TP+FP, recall = TP/TP+FN
    rows, cols = conf_mat.get_shape()
    size = y_true_class.get_shape()[0]
    precision = tf.constant([0, 0, 0])  # change this to use rows/cols as size
    recall = tf.constant([0, 0, 0])
    class_counts = tf.constant([0, 0, 0])

    def get_precision(i, conf_mat):
        print("prec check", conf_mat, conf_mat[i, i], tf.reduce_sum(conf_mat[:, i]))
        precision[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[:, i]))
        recall[i].assign(conf_mat[i, i] / tf.reduce_sum(conf_mat[i, :]))
        tf.add(i, 1)
        return i, conf_mat, precision, recall

    def tf_count(i):
        elements_equal_to_value = tf.equal(y_true_class, i)
        as_ints = tf.cast(elements_equal_to_value, tf.int32)
        count = tf.reduce_sum(as_ints)
        class_counts[i].assign(count)
        tf.add(i, 1)
        return count

    def condition(i, conf_mat):
        return tf.less(i, 3)

    i = tf.constant(3)
    i, conf_mat = tf.while_loop(condition, get_precision, [i, conf_mat])

    i = tf.constant(3)
    c = lambda i: tf.less(i, 3)
    b = tf_count(i)
    tf.while_loop(c, b, [i])

    weights = tf.math.divide(class_counts, size)
    numerators = tf.math.multiply(tf.math.multiply(precision, recall), tf.constant(2))
    denominators = tf.math.add(precision, recall)
    f1s = tf.math.divide(numerators, denominators)
    weighted_f1 = tf.reduce_sum(tf.math.multiply(f1s, weights))
    return weighted_f1


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

dim = int(np.sqrt(196))
x_test = reshape_as_image(x_test, dim, dim)

x_test = np.stack((x_test,) * 3, axis=-1)
print(x_test.shape)


from tensorflow.keras.models import load_model

model = load_model(best_model_path,custom_objects={"f1_metric": f1_metric, "f1_weighted": f1_weighted})
# model = load_model('GBP_old.h5',custom_objects={"f1_metric": f1_metric, "f1_weighted": f1_weighted})

pred = model.predict(x_test)
# pred_classes = np.argmax(pred, axis=1)
prob=np.max(pred, axis=1)
pred_classes = np.argmax(pred, axis=1)

# # print(len(prob),' ',len(pred_classes))
# # print("Shapes of x, y train/cv/test {} {} {} {} {} {}".format(x_train.shape, y_train.shape, x_cv.shape, y_cv.shape, x_test.shape, y_test.shape))
combo = np.stack((df_orig['date'].to_numpy(),prob, pred_classes), axis=1)
df = pd.DataFrame(combo)
df.to_csv('output_new.csv',header=False,index=False)
print(df.head(25))
quit()



