# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
play
http://www.lewisjin.coding.me
~~~~~~~~~~~~~~~
This script implement by Jin Fagang.
: copyright: (c) 2017 Didi-Chuxing.
: license: Apache2.0, see LICENSE for more details.
"""
import os
import sys


base_dir = sys.path[0]
print(base_dir)

a = os.path.isabs('models')
a = os.path.isdir(base_dir + '/models')
print(a)