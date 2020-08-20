# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import log, log10, ceil
import os
import pickle

import numpy as np


class EasyDict(dict):

    def __init__(self, *args, **kwargs):
        super(EasyDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# ------------------------------------------------------------------------------
# File IO
# ------------------------------------------------------------------------------

def get_fn(k, config, directory):
    fn = []
    for key in sorted(config):
        if isinstance(config[key], bool):
            if config[key]:
                fn.append(key)
        else:
            fn.append(key + str(config[key]))
    return os.path.join(directory, 'k{}'.format(k), '_'.join(fn))


def write_to_file(data, k, config, directory):
    fn = get_fn(k, config, directory)
    if os.path.exists(fn):
        print('{} exists'.format(fn))
        return
    try:
        os.makedirs(os.path.dirname(fn))
    except:
        pass
    with open(fn, 'wb') as f:
        pickle.dump(data, f)


def read_from_file(k, config, directory):
    fn = get_fn(k, config, directory)
    f = open(fn, 'rb')
    data = pickle.load(f)
    return data


def check_file_exists(k, config, directory, msg=None):
    fn = get_fn(k, config, directory)
    exists = os.path.exists(fn)
    if exists and msg:
        print(msg)
    return exists


def get_default_delta(k, epsilon):
    if epsilon < 0:
        return -1.0
    return 1 / 10**ceil(log10(k * 100))


# ------------------------------------------------------------------------------
# Generating random variable
# ------------------------------------------------------------------------------

def generate_max_geom(k, gamma, eta, m, seed):
    """Generate rv ~ max{eta, max{Z_1,..., Z_k}} where Z_i~Geom(gamma/(1+gamma)).

    Returns:
        m such rv
    """
    if k == 0:
        Z = np.zeros(m)
    else:
        np.random.seed(seed)
        # generate Z = max{Z_1,..., Z_k}, P(Z <= i) = (1 - 1/(1+gamma)^i)^k from unif
        U = np.random.uniform(size=m)    # uniform rv
        Z = -np.log(1 - np.power(U, 1 / k)) / log(1 + gamma)
        Z = np.ceil(Z)
    return np.maximum(Z, eta)
