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

"""Simulation (not actual implementation) for private FM sketch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from math import sqrt, log, exp, ceil
import numpy as np
import scipy.integrate as integrate
import scipy.special

from privateFM.utils import generate_max_geom, EasyDict


# ------------------------------------------------------------------------------
# FM sketch
# ------------------------------------------------------------------------------

def FM(k, gamma, eta, m, seed):
    """Non private FM.

    Returns:
        m rv ~ max{eta, max{Z_1,..., Z_k}} where Z_i~Geom(gamma/(1+gamma)).
    """
    if k == 0:
        print('FM gets k=0')
        return -1
    return generate_max_geom(k, gamma, eta, m, seed)


def set_k_p_eta(config):
    """A helper function for computing k_p and eta."""
    epsilon, delta, m, gamma = config.epsilon, config.delta, config.m, config.gamma
    if not 0 < epsilon < float('inf') or not 0 < delta < 1:
        k_p = 0
        eta = 0
    else:
        eps1 = epsilon / 4 / sqrt(m * log(1 / delta))
        k_p = ceil(1 / (exp(eps1) - 1))
        eta = ceil(-log(1 - exp(-eps1)) / log(1 + gamma))
        if config.morePhantom:
            k_p = max((1 + gamma)**eta, k_p)
    return k_p, eta


def FMPrivate(k, config, seed, estimation_option='quantile'):
    """Private FM.

    Args:
        k: true # distinct
        config: contains epsilon, delta, m, gamma
        seed: random seed
        estimation_option: quantile, mean_harmo, mean_geom

    Returns:
        estimation, i_max
    """
    if config.epsilon > 0 and 0 < config.delta < 1:
        assert config.epsilon <= 2 * log(1 / config.delta)
    k_p, eta = set_k_p_eta(config)
    I = FM(k + k_p, config.gamma, eta, config.m, seed)
    param = EasyDict(config=config, k_p=k_p, factor=0)
    return make_estimate(I, estimation_option, param), I


# ------------------------------------------------------------------------------
# Estimation
# ------------------------------------------------------------------------------

def make_estimate(I, option, param):
    """Make the final cardinality estimation given I.

    Args:
        option: quantile, mean_harmo, mean_geom
        param: a dictionary containing k_p and config and factor (if use quantile)

    Returns:
        estimation
    """
    assert option in ['quantile', 'mean_harmo', 'mean_geom']

    gamma = param.config.gamma
    k_p = param.k_p
    m = param.config.m
    I = np.array(I)

    if option == 'quantile':
        factor = param.factor
        return (1 + gamma)**np.quantile(I, exp(-1) - gamma * factor) - k_p

    debias = get_debias(m, option, gamma)
    if option == 'mean_geom':    # Durand & Frajolet http://algo.inria.fr/flajolet/Publications/DuFl03.pdf
        return (1 + gamma)**np.mean(I) * debias - k_p
    if option == 'mean_harmo':    # HLL https://en.wikipedia.org/wiki/HyperLogLog
        return m / np.sum(np.power(1 + gamma, -I)) * debias - k_p

    raise ValueError('make_estimation gets wrong option.')


def get_debias(m, option, gamma):
    if option == 'mean_geom':
        return (scipy.special.gamma(-1 / m) *
                        ((1 + gamma)**(-1 / m) - 1) / log(1 + gamma))**(-m) / (1 + gamma)
    if option == 'mean_harmo':
        if gamma == 1.0:
            if m <= 16:
                debias = 0.673
            elif m <= 32:
                debias = 0.697
            elif m <= 64:
                debias = 0.709
            elif m >= 128:
                debias = 0.7213 / (1 + 1.079 / m)
            return debias
        else:
            debias = 1 / integrate.quad(
                    lambda u: (log((u + 1 + gamma) /
                                                 (u + 1)) / log(1 + gamma))**m * m, 0, float('inf'))[0]
            if debias > 2:
                m = 10000
                debias = 1 / integrate.quad(
                        lambda u:
                        (log((u + 1 + gamma) /
                                 (u + 1)) / log(1 + gamma))**m * m, 0, float('inf'))[0]
                # print('gamma is larger than 2, changed')
            return debias
