# -*- coding:utf-8 -*-
# author: zjw
import os
from hw2 import main_p
from hw1 import data_operation
data_operation.dir_path = os.path.abspath('../cifar')
main_p.run()