# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
read_rec
http://www.lewisjin.coding.me
~~~~~~~~~~~~~~~
This script implement by Jin Fagang.
: copyright: (c) 2017 Didi-Chuxing.
: license: Apache2.0, see LICENSE for more details.
"""
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np

batch_size = 4
# data_shape: channels, height, width
data_iter = mx.io.ImageRecordIter(
        path_imgrec="tiny5_val.rec",
        data_shape=(3, 200, 300),
        batch_size=batch_size,
        rand_mirror=True,
)
data_iter.reset()
batch = data_iter.next()
data = batch.data[0]
print(data)
for i in range(0, 4):
    plt.subplot(1, 4, i + 1)
    plt.imshow(data[i].asnumpy().astype(np.uint8).transpose((1, 2, 0)))
plt.show()