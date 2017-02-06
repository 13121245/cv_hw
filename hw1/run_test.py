# -*-coding:utf-8 -*-

import data_operation
import knn
import time
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

k_list = [1, 5, 9]
fea_k_list = [200, 300, 500]
# fea_k_list = [300, 500]


def random_project_knn(train_data, test_data):
    x_train = train_data['data']        # 50000-by-3072 matrix
    y_train = train_data['labels']      # 1-by-50000 matrix
    x_test = test_data['data']          # 10000-by-3072 matrix
    y_test = test_data['labels']        # 1-by-10000 matrix
    (n_train, fea_n) = x_train.shape
    n_test = x_test.shape[0]
    for fea_k in fea_k_list:
        tf_matrix = get_rp_matrix(fea_n, fea_k)           # fea_n-by-fea_k matrix
        x_train_n = np.dot(x_train, tf_matrix)
        x_test_n = np.dot(x_test, tf_matrix)
        for k in k_list:
            # labels = knn.knn(np.asmatrix(x_test_n.T), np.asmatrix(x_train_n.T), np.asmatrix(y_train), k)
            clf = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=k)
            clf.fit(x_train_n, y_train)
            labels = clf.predict(x_test_n)
            accuracy = float(np.sum(labels == y_test)) / n_test
            print 'random project_knn: D is ', fea_k, ', k is', k, ', Accuracy is ', accuracy


def random_project_svm(train_data, test_data):
    x_train = train_data['data']
    y_train = train_data['labels']
    x_test = test_data['data']
    y_test = test_data['labels']
    (n_train, fea_n) = x_train.shape
    n_test = x_test.shape[0]
    # clf = svm.SVC(decision_function_shape='ovo')
    clf = svm.LinearSVC()
    for fea_k in fea_k_list:
        tf_matrix = get_rp_matrix(fea_n, fea_k)
        x_train_n = np.dot(x_train, tf_matrix)
        x_test_n = np.dot(x_test, tf_matrix)        # num_of_sample-by-num_of_feature
        clf.fit(x_train_n, y_train)
        labels = clf.predict(x_test_n)
        accuracy = float(np.sum(labels == y_test)) / n_test
        print 'random project_svm: D is ', fea_k, ', Accuracy is ', accuracy


def svd_svm(train_data, test_data):
    x_train = train_data['data']
    y_train = train_data['labels']
    x_test = test_data['data']
    y_test = test_data['labels']
    n_test = x_test.shape[0]
    for fea_k in fea_k_list:
        x_train_n = get_svd_matrix(x_train.T, fea_k)  # fea_k-by-num_of_x_train
        x_test_n = get_svd_matrix(x_test.T, fea_k)
        # clf = svm.SVC(decision_function_shape='ovo')
        clf = svm.LinearSVC()
        clf.fit(x_train_n.T, y_train)
        labels = clf.predict(x_test_n.T)
        accuracy = float(np.sum(labels == y_test)) / n_test
        print 'svd_svm: D is ', fea_k, ', Accuracy is ', accuracy


def svd_knn(train_data, test_data):
    x_train = train_data['data']
    y_train = train_data['labels']
    x_test = test_data['data']
    y_test = test_data['labels']
    n_test = x_test.shape[0]
    for fea_k in fea_k_list:
        x_train_n = get_svd_matrix(x_train.T, fea_k)  # fea_k-by-num_of_x_train
        x_test_n = get_svd_matrix(x_test.T, fea_k)
        for k in k_list:
            # labels = knn.knn(np.asmatrix(x_test_n), np.asmatrix(x_train_n),
            #                  np.asmatrix(y_train), k)
            clf = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=k)
            clf.fit(x_train_n.T, y_train)
            labels = clf.predict(x_test_n.T)
            accuracy = float(np.sum(labels == y_test)) / n_test
            print 'svd_knn: ', ', k is', k, ', Accuracy is ', accuracy


def random_project_lr(train_data, test_data):
    x_train = train_data['data']
    y_train = train_data['labels']
    x_test = test_data['data']
    y_test = test_data['labels']
    (n_train, fea_n) = x_train.shape
    n_test = x_test.shape[0]
    # clf = svm.SVC(decision_function_shape='ovo')
    clf = svm.LinearSVC()
    for fea_k in fea_k_list:
        tf_matrix = get_rp_matrix(fea_n, fea_k)
        x_train_n = np.dot(x_train, tf_matrix)
        x_test_n = np.dot(x_test, tf_matrix)  # num_of_sample-by-num_of_feature
        clf.fit(x_train_n, y_train)
        labels = clf.predict(x_test_n)
        accuracy = float(np.sum(labels == y_test)) / n_test
        print 'random project_svm: D is ', fea_k, ', Accuracy is ', accuracy


def color_data_run():
    train_data = data_operation.get_color_train_data()
    test_data = data_operation.get_color_test_data()
    random_project_knn(train_data, test_data)
    # svd_knn(train_data, test_data)
    # random_project_svm(train_data, test_data)
    # svd_svm(train_data, test_data)


def gray_data_run():
    train_data = data_operation.get_gray_train_data()
    test_data = data_operation.get_gray_test_data()
    # random_project_knn(train_data, test_data)
    # svd_knn(train_data, test_data)
    # random_project_svm(train_data, test_data)
    svd_svm(train_data, test_data)


def get_svd_matrix(data_old, fea_k):
    '''
    :param data_old:  num_of_feature-By-num_of_sample array
    :param fea_k: 降维后的feature个数
    :return: 降维后的 fea_k-by-num_of_sample array
    '''
    u, s, v = np.linalg.svd(data_old, full_matrices=False)
    data_new = np.dot(u[:, 0:fea_k].T, data_old)
    return data_new


def get_rp_matrix(fea_n, fea_k):
    '''
    :param fea_n: 原始的特征数目
    :param fea_k: 降维后的特征数目
    :return: tf_matrx: 随机投影矩阵 fea_n-by-fea_k matrix
    '''
    rn = np.random.standard_normal(fea_n*fea_k)
    rn = rn.reshape(fea_n, fea_k)
    tf_matrix = np.asmatrix(rn)
    return tf_matrix


if __name__ == '__main__':
    # gray_data_run()
    color_data_run()
    # a = np.random.randn(9, 6)
    # b = get_svd_matrix(a, 3)
    # print b.shape