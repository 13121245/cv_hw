# -*-coding:utf-8 -*-

import data_operation
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def random_project(fea_k):
    train_data = data_operation.get_color_train_data()
    x_train = train_data['data']  # 50000-by-3072 matrix
    y_train = train_data['labels']  # 1-by-50000 matrix
    test_data = data_operation.get_color_test_data()
    x_test = test_data['data']      # 10000-by-3072 matrix
    y_test = test_data['labels']    # 1-by-10000 matrix
    k_list = [1, 5, 9]
    (n_train, fea_n) = x_train.shape
    n_test = x_test.shape[0]
    tf_matrix = get_rp_matrix(fea_n, fea_k)  # fea_k-by-fea_n matrix
    x_ntrain = np.dot(x_train, tf_matrix)
    x_ntest = np.dot(x_test, tf_matrix)
    for k in k_list:
        clf = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=k)
        clf.fit(x_ntrain, y_train)
        labels = clf.predict(x_ntest)
        accuracy = float(np.sum(labels == y_test)) / n_test
        print 'random project: D is ', fea_k, ', k is', k, ', Accuracy is ', accuracy


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
    random_project(500)