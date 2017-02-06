# -*- coding:utf-8 -*-
import numpy as np
from numpy import matlib
from sklearn import neighbors


def knn(data, data_train, label_train, k):
    '''
    :param data: 要预测的数组     P-by-N_test matrix
    :param data_train: 训练数据  P-by-N_train matrix
    :param label_train: 训练标签 1-by-N_train matrix
    :param k:
    :return: labels: 预测的标签值 1-by-N_test matrix
    '''
    (fea_n, n_test) = data.shape
    n_train = data_train.shape[1]
    labels = list()
    if k > n_train:
        print 'k=', k, ' is larger than the number of samples', n_train
        k = n_train
    for i in range(n_test):
        data_i = data[:, i]
        tmp_data = matlib.repmat(data_i, 1, n_train)    # P-by-N_train matrix
        diff_mat = tmp_data - data_train
        diff_mat = np.multiply(diff_mat, diff_mat)
        dist_mat = diff_mat.sum(0)      # 1-by-N_train matrix
        sorted_index = np.argsort(dist_mat)
        vote_dict = {}
        # print diff_mat, sorted_index
        for j in range(k):
            index = sorted_index[0, j]
            label = label_train[0, index]
            vote_dict[label] = vote_dict.get(label, 0) + 1
        vote_list = sorted(vote_dict.iteritems(), key=lambda d: d[1], reverse=True)
        labels.append(vote_list[0][0])
    return np.array(labels)

if __name__ == '__main__':
    d_train = np.asmatrix([[1, 2, 3], [4, 5, 6]])
    d_label = np.asmatrix([0, 0, 1])
    d_test = np.asmatrix([[0, 3], [0, 6]])
    print knn(d_test, d_train, d_label, 2)