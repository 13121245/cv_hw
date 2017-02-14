# -*- coding:utf-8 -*-
from pylab import *
from PIL import Image
import cPickle
import os
import numpy as np

dir_path = os.path.abspath('../../cifar')
color_test_file = 'test_batch'
gray_test_file = 'gray_test_data'
gray_train_file = 'gray_train_data'
sift_train_file = 'sift_train'
sift_test_file = 'sift_test'
dictionary_file = 'dictionary'
sparse_train_file = 'sparse_train'
sparse_test_file = 'sparse_test'
im_size = 32
im_chanel = 3


def get_color_train_data():
    data_file_name = 'data_batch'
    all_data = {}
    count = 0
    for file_name in os.listdir(dir_path):
        if not file_name.startswith(data_file_name):
            continue
        count += 1
        file_path = os.path.join(dir_path, file_name)
        with open(file_path, 'rb') as train_data:
            data_dict = cPickle.load(train_data)
            if count == 1:
                all_data['data'] = data_dict['data']
                all_data['labels'] = data_dict['labels']
                all_data['filenames'] = data_dict['filenames']
            else:
                all_data['data'] = np.concatenate((all_data['data'], data_dict['data']))
                all_data['labels'].extend(data_dict['labels'])
                all_data['filenames'].extend(data_dict['filenames'])
    return all_data


def get_color_test_data():
    file_path = os.path.join(dir_path, color_test_file)
    with open(file_path, 'rb') as test_data:
        data_dict = cPickle.load(test_data)
    return data_dict


def get_gray_train_data():
    file_path = os.path.join(dir_path, gray_train_file)
    with open(file_path, 'rb') as test_data:
        data_dict = cPickle.load(test_data)
        print data_dict['data'].shape
    return data_dict


def get_gray_test_data():
    file_path = os.path.join(dir_path, gray_test_file)
    with open(file_path, 'rb') as test_data:
        data_dict = cPickle.load(test_data)
    return data_dict


def show_color_image(im_array):
    # im = array([im_array[0:1024], im_array[1024:2048], im_array[2048:3072]])
    # im = im.T
    # imshow(im.reshape(32, 32, 3))
    im_n = im_array.reshape((im_size*im_size, 3), order='F')
    imshow(im_n.reshape(32, 32, 3))
    show()


def show_gray_image(im_array):
    im_n = im_array.reshape((im_size, im_size))
    imshow(im_n)
    show()


def save_gray_train_data():
    color_data = get_color_train_data()
    gray_data = color_to_gray(color_data)
    __save_data(gray_data, gray_train_file)


def save_gray_test_data():
    color_data = get_color_test_data()
    gray_data = color_to_gray(color_data)
    __save_data(gray_data, gray_test_file)


def get_sift_train_data():
    file_path = os.path.join(dir_path, sift_train_file)
    with open(file_path, 'rb') as sift_train:
        data_dict = cPickle.load(sift_train)
    return data_dict


def get_sift_test_data():
    file_path = os.path.join(dir_path, sift_test_file)
    with open(file_path, 'rb') as sift_test:
        data_dict = cPickle.load(sift_test)
    return data_dict


def save_sift_data(sift_data, file_name):
    print 'start to save sift data with filel_name ', file_name
    __save_data(sift_data, file_name)


def save_dictionary_data(d_data):
    print 'start to save dictionary data with filel_name ', dictionary_file
    __save_data(d_data, dictionary_file)


def get_dictionary_data():
    file_path = os.path.join(dir_path, dictionary_file)
    with open(file_path, 'rb') as cook_book:
        data_dict = cPickle.load(cook_book)
    return data_dict


def get_sparse_train_data():
    file_path = os.path.join(dir_path, sparse_train_file)
    with open(file_path, 'rb') as sc_train:
        data_dict = cPickle.load(sc_train)
    return data_dict


def get_sparse_test_data():
    file_path = os.path.join(dir_path, sparse_test_file)
    with open(file_path, 'rb') as sc_test:
        data_dict = cPickle.load(sc_test)
    return data_dict

def save_sparse_data(sparse_data, file_name):
    print 'start to save sparse_data data with file_name ', file_name
    __save_data(sparse_data, file_name)


def __save_data(gray_data, file_name):
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, 'wb') as df:
        cPickle.dump(gray_data, df)


def color_to_gray(color_data):
    gray_data = {
        'labels': color_data['labels'],
        'filenames': color_data['filenames'],
        'data': array([])
    }
    data_list = []
    data = color_data['data']
    im_num = color_data['data'].shape[0]   # 图片数
    pixel_num = im_size*im_size             # 图片像素个数
    print im_num, pixel_num
    # gray = (0.3008 * r + 0.5859 * g + 0.1133 * b)
    get_gray = lambda r, g, b: 0.3008 * r + 0.5859 * g + 0.1133 * b
    for i in range(im_num):
        p_list = []
        for j in range(pixel_num):
            gray_value = int(get_gray(data[i, j], data[i, j+pixel_num], data[i, j+pixel_num*2]))
            # print gray_value
            p_list.append(gray_value)
        data_list.append(p_list)
    gray_data['data'] = array(data_list, dtype=np.uint8)
    return gray_data


if __name__ == '__main__':
    # print train_data['data'].shape
    train = get_color_train_data()
    # print train['data'].shape, type(train['labels']), len(train['labels'])
    # __save_data(train, 'testing_all')
    # show_color_image(train['data'][99])
    # save_gray_train_data()
    # save_gray_test_data()
    # g_train = get_gray_train_data()
    # print g_train['data'].shape
    # show_gray_image(g_train['data'][99])
    # print g_train['data'][99]