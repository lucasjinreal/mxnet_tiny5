# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
im2rec_gen_list
http://www.lewisjin.coding.me
~~~~~~~~~~~~~~~
This script implement by Jin Fagang.
: copyright: (c) 2017 Didi-Chuxing.
: license: Apache2.0, see LICENSE for more details.
"""
import os
import sys
import argparse
import numpy as np

current_path = os.path.abspath(os.path.dirname(__file__))

class_index_dict = {}


def generate_words(train_path):
    print(train_path)
    i = 0
    words_file = open('words.txt', 'w+')
    for fn in os.listdir(train_path):
        filepath = os.path.join(train_path, fn)
        if os.path.isdir(filepath):
            print('Find the {0} class: {1}'.format(i, fn))
            words_file.write(str(i) + ' ' + fn + '\n')
            class_index_dict[fn] = str(i)
            i = (i+1)
    words_file.close()
    print('All class index and names has been saved into words.txt!')
    print(class_index_dict)


def generate_train_images_path(train_path, is_shuffle, is_split):
    train_txt = open('train_list.txt', 'w+')
    print(train_path)
    all_train_lines = []
    i = 0
    for fn in os.listdir(train_path):
        filepath = os.path.join(train_path, fn)
        if os.path.isdir(filepath):
            for root, dirs, files in os.walk(filepath):
                for file in files:
                    imagefilepath = '/' + fn + '/' + file
                    print('{0} {1} {2}'.format(i, imagefilepath, class_index_dict[fn]))
                    all_train_lines.append(str(i) + ' ' + class_index_dict[fn] + ' ' +
                                           imagefilepath + '\n')
                    i = (i+1)

    if is_shuffle and is_split:
        np.random.shuffle(all_train_lines)
        l = len(all_train_lines)
        boundry = int(l*0.8)
        train_lines = all_train_lines[0: boundry]
        val_lines = all_train_lines[boundry:]
        print('split {0} train and {1} val'.format(boundry, l-boundry))

        val_txt = open('val_list.txt', 'w+')
        for line in val_lines:
            val_txt.write(line)
        for line in train_lines:
            train_txt.write(line)
        val_txt.close()
    elif not is_shuffle and not is_split:
        for line in all_train_lines:
            train_txt.write(line)
    elif is_shuffle and not is_split:
        np.random.shuffle(all_train_lines)
        for line in all_train_lines:
            train_txt.write(line)

    train_txt.close()
    print('All train images path and class index has been saved into train_list.txt!')


def generate_valid_images_path(val_path):
    valid_txt = open('val_list.txt', 'w+')

    all_val_lines = []
    i = 0
    for fn in os.listdir(val_path):
        filepath = os.path.join(val_path, fn)
        if os.path.isdir(filepath):
            for root, dirs, files in os.walk(filepath):
                for file in files:
                    imagefilepath = '/' + fn + '/' + file
                    print('{0} {1} {2}'.format(i, imagefilepath, class_index_dict[fn]))
                    all_val_lines.append(str(i) + ' ' + class_index_dict[fn] + ' ' +
                                         imagefilepath + '\n')
                    i = (i + 1)
    np.random.shuffle(all_val_lines)
    for line in all_val_lines:
        valid_txt.write(line)
    valid_txt.close()
    print('All val images path and class index has been saved into val_list.txt!')


def parse_args():
    parse = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='''This script using for mxnet generate train_list.txt
         and val_list.txt This will generate alongside this script directory.'''
    )
    parse.add_argument('-train', type=str, help='''your train images path,
    you can place all your classes images into a train folder,
    this script can solve it automatically.''')
    parse.add_argument('-val', type=str, help='''your valid images path,
    place all your valid images into a val folder.''')
    parse.add_argument('-shuffle', type=bool, default=False, help='if True, train images will shuffle.')
    parse.add_argument('-split', type=bool, default=False, help='''if True, train images will
    split into train and val as 80% for train, -val param you can leave none.''')

    args = parse.parse_args()

    return vars(args)


if __name__ == '__main__':
    args = parse_args()
    train_images_path = args['train']
    valid_images_path = args['val']
    is_shuffle = args['shuffle']
    is_split = args['split']
    if os.path.isdir(args['train']):
        generate_words(train_images_path)
        generate_train_images_path(train_images_path, is_shuffle, is_split)
    else:
        print('Your train images path seems invalid, please check it and try again.')

    if valid_images_path:
        if os.path.isdir(args['val']):
            generate_valid_images_path(valid_images_path)
        else:
            print('Your valid images path seems invalid.')



