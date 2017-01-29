# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_mxnet
http://www.lewisjin.coding.me
~~~~~~~~~~~~~~~
This script implement by Jin Fagang.
: copyright: (c) 2017 Didi-Chuxing.
: license: Apache2.0, see LICENSE for more details.
"""
import mxnet as mx
import numpy as np
import os
import sys
import logging
logging.getLogger().setLevel(level=logging.DEBUG)


def lenet(n_classes):
    data = mx.symbol.Variable('data')

    conv1 = mx.symbol.Convolution(data=data, kernel=(5, 5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2, 2), stride=(2, 2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2, 2), stride=(2, 2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=n_classes)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet


def mlp_net(n_classes):
    data = mx.symbol.Variable('data')
    data = mx.sym.Flatten(data=data)
    fc1 = mx.symbol.FullyConnected(data=data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
    fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
    act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type="relu")
    fc3 = mx.symbol.FullyConnected(data=act2, name='fc3', num_hidden=n_classes)
    mlp = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return mlp


def find_checkpoint(path, prefix):
    prefix = prefix.split('/')[1]
    if os.path.isdir(path):
        logging.info('models path at {0}'.format(path))
        logging.info('searching params and symbols...')
        files = os.listdir(path)
        if not files:
            logging.warning('can not find params or json file under {0}'.format(path))
            logging.warning('start from beginning train..')
        else:
            params = []
            for f in files:
                if 'params' in f and prefix in f:
                    params.append(f)
                if 'json' in f and prefix in f:
                    json = f
            print(params)
            last_params = params[-1]
            last_params = last_params.split('-')[1].split('.')[0]
            print(int(last_params))
            return int(last_params)
    else:
        logging.warning('path is invalid. pass full path please..')
        return False


if __name__ == '__main__':
    # set train params
    num_classes = 5
    batch_size = 8
    lr = 0.001
    num_epoch = 1400
    # saving model to lenet-symbol.json and lenet-0001.params.
    model_prefix = 'models/lenet'

    # setting dirs
    base_dir = sys.path[0]

    train_iter = mx.io.ImageRecordIter(
        path_imgrec="tiny5_train.rec",
        data_shape=(3, 150, 200),
        batch_size=batch_size,
        rand_mirror=True
    )
    val_iter = mx.io.ImageRecordIter(
        path_imgrec="tiny5_val.rec",
        data_shape=(3, 150, 200),
        batch_size=batch_size,
        rand_mirror=True
    )

    # define net
    net = lenet(n_classes=num_classes)
    # add checkpoint move on training
    model_path = sys.path[0] + '/models'
    iteration = find_checkpoint(model_path, model_prefix)
    if iteration:
        logging.info('continue training at epoch {0}'.format(iteration))
        model_loaded = mx.model.FeedForward.load(
            model_prefix,
            iteration,
            ctx=mx.gpu(0),
            num_epoch=num_epoch,
            learning_rate=lr,
        )
        model_loaded.fit(
            X=train_iter,
            eval_data=val_iter,
            epoch_end_callback=mx.callback.do_checkpoint(model_prefix, period=50)
        )

    else:
        model = mx.model.FeedForward.create(
            ctx=mx.gpu(0),
            symbol=net,
            X=train_iter,
            eval_data=val_iter,
            num_epoch=num_epoch,
            learning_rate=lr,
            epoch_end_callback=mx.callback.do_checkpoint(model_prefix, period=50)
        )





