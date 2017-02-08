# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_mxnet
http://www.lewisjin.coding.me
~~~~~~~~~~~~~~~
This script implement by Jin Fagang.
: copyright: (c) 2017 Didi-Chuxing.
: license: Apache2.0, see LICENSE for more details.
"""
import mxnet as mx
import cv2
import skimage
import numpy as np


img = cv2.imread('elephant.jpg')
img = cv2.resize(img, (150, 100), cv2.INTER_LINEAR)
cv2.imshow('image', img)
cv2.waitKey()

img = np.reshape(img, (1, 3, 100, 150))/255.0
print(img.shape)


model_prefix = 'models/lenet'
model_loaded = mx.model.FeedForward.load(model_prefix, 50)
prob = model_loaded.predict(img)
print(prob)
print(np.argmax(prob))