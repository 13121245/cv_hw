import numpy as np
from sklearn import random_projection

import data_operation

X = data_operation.get_color_train_data()['data'][:50000]
transformer = random_projection.GaussianRandomProjection(eps=0.5)
X_new = transformer.fit_transform(X)
print X_new.shape
