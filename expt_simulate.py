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

"""Simulation (not actual implementation) for experiments for private FM sketch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import numpy as np
from privateFM.FM_simulate import FMPrivate
from privateFM.utils import EasyDict, write_to_file, check_file_exists, get_fn
from privateFM.utils import get_default_delta


FLAGS = flags.FLAGS

flags.DEFINE_integer('k', 1024, 'True cardinality.')

flags.DEFINE_float('epsilon', -1.0, 'DP epsilon. -1 for non-private.')
flags.DEFINE_float('delta', -1.0, 'DP delta. -1 for using the default.')

flags.DEFINE_integer('m', 64, 'The number of repetition.')
flags.DEFINE_float('gamma', 1.0, 'Accuracy parameter.')
flags.DEFINE_bool('morePhantom', False, 'If set, #phantom=k_p+(1+gamma)**eta.')

flags.DEFINE_string('res_dir', './res', 'Directory to write results.')

N_RUNS = 100


def stats(k, config, n_runs, estimation_option):
    estimations, Is = [-1]*n_runs, {}
    for run in range(n_runs):
        estimations[run], Is[run] = FMPrivate(k, config, seed=run, estimation_option=estimation_option)
    error = np.abs(np.subtract(estimations, k))
    mre = np.mean(np.divide(error, k))
    return np.mean(estimations), np.std(estimations), mre, estimations, Is


def expt_with_flag(k, config, n_runs, res_dir):
    if config.delta == -1:
        config.delta = get_default_delta(k, config.epsilon)
    assert 0 < config.gamma <= 1
    print('k={}, m={}, gamma={}, DP=({}, {})'.format(k, config.m, config.gamma, config.epsilon, config.delta))

    if check_file_exists(k, config, res_dir, 'file exists. not running'):
        return
    mean, std, mre, estimations, Is = stats(k, config, n_runs, estimation_option='mean_harmo')
    print('Over {} runs, estimation is {:.2f}({:.2f}), mre={:.4f}'.format(n_runs, mean, std, mre))
    write_to_file({'estimation': estimations, 'I': Is}, k, config, res_dir)


def main(unused_argv):
    config = EasyDict(m=FLAGS.m,
                      epsilon=FLAGS.epsilon, delta=FLAGS.delta,
                      gamma=FLAGS.gamma,
                      morePhantom=FLAGS.morePhantom)
    expt_with_flag(FLAGS.k, config, N_RUNS, FLAGS.res_dir)


if __name__ == '__main__':
    app.run(main)
