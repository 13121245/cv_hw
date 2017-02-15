# -*- coding:utf-8 -*-
# author: zjw

import cv2
import logging
import traceback
import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import sparse_encode
from sklearn.decomposition import SparseCoder
from sklearn import svm
import numpy as np

from hw1 import data_operation

IM_SIZE = 32
CLUSTER_K = 50
K_LIST = [1, 5, 9]


def get_sift_feature(img):
    """得到指定图像的dense sift特征
    :param img: 图像的np array
    :return: des
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT()
    d_sift = cv2.FeatureDetector_create("Dense")
    d_sift.setDouble('initFeatureScale', 4)
    d_sift.setInt('initXyStep', 2)
    key_points = d_sift.detect(gray)
    key_points, des = sift.compute(gray, key_points)
    # print len(key_points)
    # print des.shape, des.dtype
    # print des
    return des


def get_img_matrix(im_array):
    """
    :param img_data_list: 输入为一幅图像数据的数组 1-by-3072
    :return: 图像的矩阵 shape 为(32, 32, 3)
    """
    # img_matrix = np.empty((IM_SIZE, IM_SIZE, 3))
    # img_matrix[:, :, 0] = im_array[0:1024].reshape(IM_SIZE, IM_SIZE)
    # img_matrix[:, :, 1] = im_array[1024:2048].reshape(IM_SIZE, IM_SIZE)
    # img_matrix[:, :, 2] = im_array[2048:3072].reshape(IM_SIZE, IM_SIZE)
    img_matrix = im_array.reshape((32*32, 3), order='F').reshape(32, 32, 3)
    # cv2.imshow('img', img_matrix)
    # cv2.waitKey(0)
    return img_matrix


def save_sift_data(origin_data, data_file_path):
    """
    :param origin_data: 原始图像数据
    :param data_file_path: 图像的sift特征保存位置
    :return:
    """
    X = origin_data['data']    # num_of_sample-by-3072 matrx
    Y = origin_data['labels']  # 1-by-num_of_sample matrix
    num_x = X.shape[0]
    sift_data = list()
    for i in range(0, min(num_x, num_x)):
        img = get_img_matrix(X[i])
        sf = get_sift_feature(img)
        sift_data.append(sf)
    result_dict = dict()
    result_dict['data'] = np.array(sift_data, dtype=np.uint8)
    result_dict['labels'] = Y
    print result_dict['data'].shape
    data_operation.save_sift_data(result_dict, data_file_path)


def knn(sc_train, sc_test):
    """
    :param sc_train: 训练数据的稀疏编码和标签
    :param sc_test:  测试数据的稀疏编码和标签
    :return:
    """
    x_train = sc_train['data']    # 稀疏编码， 训练数据个数-by-字典中的个数
    y_train = sc_train['labels']  # 训练数据标签
    x_test = sc_test['data']      # 稀疏编码， 测试数据个数-by-字典中的个数
    y_test = sc_test['labels']    # 测试数据标签
    print type(x_test)
    n_test = len(x_test)
    for k in K_LIST:
        clf = KNeighborsClassifier(algorithm='kd_tree', n_neighbors=k)
        clf.fit(x_train, y_train)
        labels = clf.predict(x_test)
        accuracy = float(np.sum(labels == y_test)) / n_test
        logging.info('k is' + str(k) + ', Accuracy is ' + str(accuracy))


def svm_c(sc_train, sc_test):
    """
        :param sc_train: 训练数据的稀疏编码和标签
        :param sc_test:  测试数据的稀疏编码和标签
        :return:
    """
    x_train = sc_train['data']    # 稀疏编码， 训练数据个数-by-字典中的个数
    y_train = sc_train['labels']  # 训练数据标签
    x_test = sc_test['data']      # 稀疏编码， 测试数据个数-by-字典中的个数
    y_test = sc_test['labels']    # 测试数据标签
    n_test = len(x_test)
    clf = svm.LinearSVC()
    clf.fit(x_train, y_train)
    labels = clf.predict(x_test)
    accuracy = float(np.sum(labels == y_test)) / n_test
    logging.info('svm Accuracy is ' + str(accuracy) )


def init():
    """ step 1 初始化sift特征
    :return:
    """
    logging.info('step 1')
    # 得到训练数据和测试数据的sift特征，将其保存到本地
    train_data = data_operation.get_color_train_data()
    save_sift_data(train_data, data_operation.sift_train_file)
    test_data = data_operation.get_color_test_data()
    save_sift_data(test_data, data_operation.sift_test_file)


def get_dictionary():
    """step 2 cluster these features into bases(centroids) using k-means
    :return:
    """
    logging.info('step 2')
    sift_train = data_operation.get_sift_train_data()
    sift_data = sift_train['data']
    sift_data_shape = sift_data.shape
    print sift_data_shape
    # sift_features_all = np.array([], dtype=np.uint8)
    # sift_features_all.shape = (0, sift_data.shape[2])
    # for img_sift in sift_data:
    #     sift_features_all = np.concatenate((sift_features_all, img_sift))
    sift_features_all = np.reshape(sift_data, (sift_data_shape[0]*sift_data_shape[1], 128))
    # print sift_features_all[1]
    print sift_features_all.shape
    # kmeans = KMeans(n_clusters=CLUSTER_K, random_state=0).fit(sift_features_all)
    kmeans = MiniBatchKMeans(n_clusters=CLUSTER_K, batch_size=100).fit(sift_features_all)
    data_operation.save_dictionary_data(kmeans.cluster_centers_)


def get_sparse_codes():
    """
    step3  得到训练和测试图像的稀疏编码
    learn sparse codes with these bases, and perform average pooling on sparse codes per image
    :return:
    """
    logging.info('step 3')
    dictionary = data_operation.get_dictionary_data()
    print dictionary.shape
    # train
    sift_train = data_operation.get_sift_train_data()
    train_data = sift_train['data']
    train_label = sift_train['labels']
    sparse_train = list()
    print train_data.shape
    for i, x in enumerate(train_data):
        print i
        # codes = sparse_encode(x, dictionary)
        se = SparseCoder(dictionary=dictionary)
        codes = se.transform(x)
        sparse_train.append(np.average(codes, axis=0))
        # print np.average(codes, axis=0)
    result_dict = dict()
    result_dict['data'] = sparse_train
    result_dict['labels'] = train_label
    data_operation.save_sparse_data(result_dict, data_operation.sparse_train_file)
    # test
    sift_test = data_operation.get_sift_test_data()
    test_data = sift_test['data']
    test_label = sift_test['labels']
    sparse_test = list()
    for i, x in enumerate(test_data):
        # codes = sparse_encode(x, dictionary)
        print i
        se = SparseCoder(dictionary=dictionary)
        codes = se.transform(x)
        sparse_test.append(np.average(codes, axis=0))
    result_dict = dict()
    result_dict['data'] = sparse_test
    result_dict['labels'] = test_label
    data_operation.save_sparse_data(result_dict, data_operation.sparse_test_file)


def classify():
    """
    step 4 进行分类预测
    use k-NN (k = 1, 5, 9) and multi-class linear classifier (least squares) to classify testing data
    """
    logging.info('step 4')
    sc_train = data_operation.get_sparse_train_data()
    sc_test = data_operation.get_sparse_test_data()
    knn(sc_train, sc_test)
    svm_c(sc_train, sc_test)


def run():
    logging.basicConfig(filename='hw2.log', filemode="w", level=logging.DEBUG)
    logging.info(datetime.datetime.utcnow())
    try:
        # init()
        # get_dictionary()
        get_sparse_codes()
        classify()
    except Exception as e:
        logging.exception(e)

if __name__ == "__main__":
    run()
