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
    global df,num_features,model_path,best_model_path
    np.random.seed(2)
    tf.random.set_seed(2)
    num_features=196
    metatrader_dir="C:\\Users\\Barbocz Attila\\AppData\\Roaming\\MetaQuotes\\Terminal\\67381DD86A2959850232C0BA725E5966\\MQL5\\Files\\"
    f = open(metatrader_dir+"Parameters.txt","r")
    # print(f.readline().split(':')[1])
    # f.readline().split(':')[1]
    model_path = re.sub('[^A-Za-z0-9]+', '', f.readline().split(':')[1])

    best_model_path = os.path.join('.', 'best_models', model_path)
    try:
        os.mkdir(best_model_path)
    except OSError:
        pass
    else:
        print("Successfully created the directory %s " % best_model_path)

    df = pd.read_csv(metatrader_dir+"Training.csv")
    df = df.drop(columns=['date', 'open', 'high', 'low', 'close'])

    df['labels'] = df['labels'].astype(np.int8)
    print("Creating model for",model_path," with ", df.shape, " shape" )



def create_arrays_for_trainig_testing_validation():
    global df,x_train,x_validation,x_test,y_train,y_test,y_validation,list_features
    df = df.drop(columns=df.columns[df.nunique() <= 1])
    last_feature = df.columns.ravel()[df.columns.ravel().size - 2]

    list_features = list(df.loc[:, 'volume':last_feature].columns)
    print('Total number of features', len(list_features))
    x_train, x_test, y_train, y_test = train_test_split(df.loc[:, 'volume':last_feature].values, df['labels'].values,
                                                        train_size=0.8,
                                                        test_size=0.2, random_state=2, stratify=df['labels'].values)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, train_size=0.7, test_size=1 - 0.7,
                                                                    random_state=2, stratify=y_train)



def data_wrangling():
    global x_train,x_validation,x_test,y_train,y_test,y_validation,x_main
    my_imputer = SimpleImputer()
    x_train = my_imputer.fit_transform(x_train)
    x_validation = my_imputer.fit_transform(x_validation)

    # x_main = x_train.copy()
    # x_main = my_imputer.fit_transform(x_main)

    mm_scaler = MinMaxScaler(feature_range=(0, 1))  # or StandardScaler?
    x_train = mm_scaler.fit_transform(x_train)
    x_validation = mm_scaler.transform(x_validation)
    x_test = mm_scaler.transform(x_test)
    print("Shapes of train: {} {}, validation: {} {}, test: {} {}".format(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape,
                                                                          x_test.shape, y_test.shape))



def select_best_features():
    global x_train, x_validation, x_test, y_train, y_test, y_validation,x_main,list_features,num_features


    selection_method = 'all'
    topk = 200 if selection_method == 'all' else num_features


    if selection_method == 'anova' or selection_method == 'all':
        select_k_best = SelectKBest(f_classif, k=topk)
        if selection_method != 'all':
            x_train = select_k_best.fit_transform(x_main, y_train)
            x_validation = select_k_best.transform(x_validation)
            x_test = select_k_best.transform(x_test)
        else:
            select_k_best.fit(x_train, y_train)

        selected_features_anova = itemgetter(*select_k_best.get_support(indices=True))(list_features)
        # print(selected_features_anova)
        # print(select_k_best.get_support(indices=True))


    if selection_method == 'mutual_info' or selection_method == 'all':
        select_k_best = SelectKBest(mutual_info_classif, k=topk)
        if selection_method != 'all':
            x_train = select_k_best.fit_transform(x_main, y_train)
            x_validation = select_k_best.transform(x_validation)
            x_test = select_k_best.transform(x_test)
        else:
            select_k_best.fit(x_train, y_train)

        selected_features_mic = itemgetter(*select_k_best.get_support(indices=True))(list_features)
        # print(len(selected_features_mic), selected_features_mic)
        # print(select_k_best.get_support(indices=True))


    if selection_method == 'all':
        # common = list(set(selected_features_anova).intersection(selected_features_mic))
        common = list(set(selected_features_anova))
        # print("common selected featues", len(common), common)
        if len(common) < num_features:
            raise Exception(
                'number of common features found {} < {} required features. Increase "topk variable"'.format(
                    len(common), num_features))
        feat_idx = []
        for c in common:
            feat_idx.append(list_features.index(c))
        feat_idx = sorted(feat_idx[0:196])

    columns_needed = []
    for c in feat_idx:
        columns_needed.append(list_features[c])

    pd.DataFrame(columns_needed).to_csv(os.path.join('.', 'best_models', model_path, 'columns_needed.csv'),
                                        header=False, index=False)
    print("Featured columns: ",columns_needed)

    if selection_method == 'all':
        x_train = x_train[:, feat_idx]
        x_validation = x_validation[:, feat_idx]
        x_test = x_test[:, feat_idx]

    print("Shapes of train after best feature selection: {} {}, validation: {} {}, test: {} {}".format(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape,
                                                                                                       x_test.shape, y_test.shape))
    _labels, _counts = np.unique(y_train, return_counts=True)
    print("Percentage of class 0 = {}, class 1 = {}".format(_counts[0] / len(y_train) * 100,
                                                            _counts[1] / len(y_train) * 100))

def get_sample_weights(y):
    """
    calculate the sample weights based on class weights. Used for models with
    imbalanced data and one hot encoding prediction.

    params:
        y: class labels as integers
    """

    y = y.astype(int)  # compute_class_weight needs int labels
    class_weights = compute_class_weight('balanced', np.unique(y), y)

    print("real class weights are {}".format(class_weights), np.unique(y))
    print("value_counts", np.unique(y, return_counts=True))
    sample_weights = y.copy().astype(float)
    for i in np.unique(y):
        sample_weights[sample_weights == i] = class_weights[i]  # if i == 2 else 0.8 * class_weights[i]
        # sample_weights = np.where(sample_weights == i, class_weights[int(i)], y_)

    return sample_weights


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

def reshape_arrays():
    global x_train, x_validation, x_test, y_train, y_test, y_validation,num_features,sample_weights

    sample_weights = get_sample_weights(y_train)

    one_hot_enc = OneHotEncoder(sparse=False, categories='auto')  # , categories='auto'
    y_train = one_hot_enc.fit_transform(y_train.reshape(-1, 1))

    y_validation = one_hot_enc.fit_transform(y_validation.reshape(-1, 1))

    y_test = one_hot_enc.fit_transform(y_test.reshape(-1, 1))

    dim = int(np.sqrt(num_features))
    x_train = reshape_as_image(x_train, dim, dim)
    x_validation = reshape_as_image(x_validation, dim, dim)
    x_test = reshape_as_image(x_test, dim, dim)
    # adding a 1-dim for channels (3)
    x_train = np.stack((x_train,) * 3, axis=-1)
    x_test = np.stack((x_test,) * 3, axis=-1)
    x_validation = np.stack((x_validation,) * 3, axis=-1)

    print("Shapes of train: {} {}, validation: {} {}, test: {} {}".format(x_train.shape, y_train.shape, x_validation.shape, y_validation.shape,
                                                                          x_test.shape, y_test.shape))
def f1_custom(y_true, y_pred):
    y_t = np.argmax(y_true, axis=1)
    y_p = np.argmax(y_pred, axis=1)
    f1_score(y_t, y_p, labels=None, average='weighted', sample_weight=None, zero_division='warn')

def create_model_cnn():
    global params,model
    model = Sequential()

    params = {'batch_size': 80,
              'conv2d_layers': {'conv2d_do_1': 0.22, 'conv2d_filters_1': 20, 'conv2d_kernel_size_1': 2,
                                'conv2d_mp_1': 2,
                                'conv2d_strides_1': 1, 'kernel_regularizer_1': 0.0, 'conv2d_do_2': 0.05,
                                'conv2d_filters_2': 40, 'conv2d_kernel_size_2': 2, 'conv2d_mp_2': 2,
                                'conv2d_strides_2': 2,
                                'kernel_regularizer_2': 0.0, 'layers': 'two'},
              'dense_layers': {'dense_do_1': 0.22, 'dense_nodes_1': 100, 'kernel_regularizer_1': 0.0, 'layers': 'one'},
              'epochs': 3000, 'lr': 0.001, 'optimizer': 'adam'}

    print("Training with params {}".format(params))
    # (batch_size, timesteps, data_dim)
    # x_train, y_train = get_data_cnn(df, df.head(1).iloc[0]["timestamp"])[0:2]
    conv2d_layer1 = Conv2D(params["conv2d_layers"]["conv2d_filters_1"],
                           params["conv2d_layers"]["conv2d_kernel_size_1"],
                           strides=params["conv2d_layers"]["conv2d_strides_1"],
                           kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_1"]),
                           padding='valid',activation="relu", use_bias=True,
                           kernel_initializer='glorot_uniform',
                           input_shape=(x_train[0].shape[0],
                                        x_train[0].shape[1], x_train[0].shape[2]))
    model.add(conv2d_layer1)
    if params["conv2d_layers"]['conv2d_mp_1'] == 1:
        model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(params['conv2d_layers']['conv2d_do_1']))
    if params["conv2d_layers"]['layers'] == 'two':
        conv2d_layer2 = Conv2D(params["conv2d_layers"]["conv2d_filters_2"],
                               params["conv2d_layers"]["conv2d_kernel_size_2"],
                               strides=params["conv2d_layers"]["conv2d_strides_2"],
                               kernel_regularizer=regularizers.l2(params["conv2d_layers"]["kernel_regularizer_2"]),
                               padding='valid',activation="relu", use_bias=True,
                               kernel_initializer='glorot_uniform')
        model.add(conv2d_layer2)
        if params["conv2d_layers"]['conv2d_mp_2'] == 1:
            model.add(MaxPool2D(pool_size=2))
        model.add(Dropout(params['conv2d_layers']['conv2d_do_2']))

    model.add(Flatten())

    model.add(Dense(params['dense_layers']["dense_nodes_1"], activation='relu'))
    model.add(Dropout(params['dense_layers']['dense_do_1']))

    if params['dense_layers']["layers"] == 'two':
        model.add(Dense(params['dense_layers']["dense_nodes_2"], activation='relu',
                        kernel_regularizer=params['dense_layers']["kernel_regularizer_1"]))
        model.add(Dropout(params['dense_layers']['dense_do_2']))

    model.add(Dense(3, activation='softmax'))
    if params["optimizer"] == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=params["lr"])
    elif params["optimizer"] == 'sgd':
        optimizer = optimizers.SGD(lr=params["lr"], decay=1e-6, momentum=0.9, nesterov=True)
    elif params["optimizer"] == 'adam':
        optimizer = optimizers.Adam(learning_rate=params["lr"], beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_metric])
    # from keras.utils.vis_utils import plot_model use this too for diagram with plot
    # model.summary(print_fn=lambda x: print(x + '\n'))
    return model

def check_baseline(pred, y_test):
    print("size of test set", len(y_test))
    e = np.equal(pred, y_test)
    print("TP class counts", np.unique(y_test[e], return_counts=True))
    print("True class counts", np.unique(y_test, return_counts=True))
    print("Pred class counts", np.unique(pred, return_counts=True))
    holds = np.unique(y_test, return_counts=True)[1][2]  # number 'hold' predictions
    print("baseline acc:", (holds/len(y_test)*100))

def fit_model():
    global model,model_path,params,x_train,sample_weights,y_train,best_model_path


    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=100, min_delta=0.0001)
    # csv_logger = CSVLogger(os.path.join(OUTPUT_PATH, 'log_training_batch.log'), append=True)
    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=20, verbose=1, mode='min',
                            min_delta=0.001, cooldown=1, min_lr=0.0001)
    mcp = ModelCheckpoint(best_model_path, monitor='val_f1_metric', verbose=1,
                          save_best_only=True, save_weights_only=False, mode='max', period=1)  # val_f1_metric
    history = model.fit(x_train, y_train, epochs=params['epochs'], verbose=1,
                        batch_size=64,
                        # validation_split=0.3,
                        validation_data=(x_validation, y_validation),
                        callbacks=[mcp, rlp, es]
                        , sample_weight=sample_weights)

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
create_arrays_for_trainig_testing_validation()
data_wrangling()
select_best_features()
reshape_arrays()

get_custom_objects().update({"f1_metric": f1_metric})
model = create_model_cnn()
fit_model()

model = load_model(best_model_path,custom_objects={"f1_metric": f1_metric})
evaluate_model()



