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

"""Functions for plotting the results."""

from absl import app
from absl import flags

import numpy as np
import os
import matplotlib.pyplot as plt

from privateFM.utils import EasyDict, read_from_file
from privateFM.utils import get_default_delta
from privateFM.FM_simulate import make_estimate, set_k_p_eta, get_debias

FLAGS = flags.FLAGS

flags.DEFINE_float('epsilon', -1.0, 'DP epsilon. -1 for non-private.')
flags.DEFINE_float('gamma', 1.0, 'Accuracy parameter.')
flags.DEFINE_string('res_dir', './res', 'Directory to write results.')


def plot_figure(ax1, x, ys, xlabel=None, ylabel=None, title=None, xscale='log', yscale=None, ylim=None):
    """
    Args:
        x: x axis value
        ys: a dict of y values to plot. If y is 2d, plot error bar
    """
    params = {'legend.handlelength': 1,
              'grid.alpha': 1,
              'grid.color': "#cccccc",
              'grid.linestyle': '--',
              'legend.fontsize': 14,
              }
    plt.rcParams.update(params)
    markersize = 5

    x = np.array(x)
    plt.setp(ax1, xticks=x, xticklabels=['2^{}'.format(int(np.log2(xx))) for xx in x])

    shapes = 'o^+*x'
    line_styles = ['--', '-', '-.', ':']
    labels = list(ys.keys())

    for idx, label in enumerate(labels):
        y = np.array(ys[label])
        style = shapes[idx % len(shapes)] + line_styles[idx % len(line_styles)]
        if y.ndim == 1:
            ax1.plot(x, y, style, markersize=markersize, label=labels[idx])
        elif y.ndim == 2:
            ax1.plot(x, y[:,0], style, markersize=markersize, label=labels[idx])
            ax1.fill_between(x, y[:,0] - y[:,1], y[:,0] + y[:,1], alpha=0.2, linewidth=3, capstyle='round', linestyle='-')
        else:
            raise ValueError('y has {} dimensions'.format(y.ndim))

    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.grid(linestyle='--')

    if xscale:
        assert xscale in ['log', 'linear']
        ax1.set_xscale(xscale)
    if yscale:
        ax1.set_yscale(yscale)
    if ylim:
        ax1.set_ylim(ylim)
    return ax1


def process_estimation(estimation, k):
    """Get some statistics of the estimations."""
    estimation = np.array(estimation)
    est_mean = np.mean(estimation)
    est_std = np.std(estimation)

    error = np.abs(estimation - k)
    mre_mean = np.mean(error / k)
    mre_std = np.std(error / k)
    return ((mre_mean, mre_std), (est_mean, est_std))


def process_data(data, k, config, factor=0):
    """Read data, make estimation and return statistics of estimations."""
    Is = data['I'].values()
    assert len(Is) == 100
    param = EasyDict({'config': config, 'k_p': set_k_p_eta(config)[0], 'factor':factor})
    est_quantile = [make_estimate(I, 'quantile', param) for I in Is]
    est_geom = [make_estimate(I, 'mean_geom', param) for I in Is]
    est_harmo = [make_estimate(I, 'mean_harmo', param) for I in Is]
    return (process_estimation(est, k) for est in [est_quantile, est_geom, est_harmo])


def main(unused_argv):
    epsilon = FLAGS.epsilon
    gamma = FLAGS.gamma
    ks = [2**i for i in range(12, 21)]
    ms = [1024, 4096, 32768]
    quantile_factor = 1/12
    morePhantom = False

    est_qs, est_gs, est_hs = {}, {}, {}
    mre_qs, mre_gs, mre_hs = {}, {}, {}
    for m in ms:
        label = 'm={}'.format(m)
        est_qs[label], est_gs[label], est_hs[label] = [], [], []
        mre_qs[label], mre_gs[label], mre_hs[label] = [], [], []
        for k in ks:
            config = EasyDict(m=m, gamma=gamma, epsilon=epsilon, delta=get_default_delta(k, epsilon), morePhantom=morePhantom)
            data = read_from_file(k, config, FLAGS.res_dir)

            (mre_q, est_q), (mre_g, est_g), (mre_h, est_h) = process_data(data, k, config, quantile_factor)
            est_qs[label].append(est_q)
            est_gs[label].append(est_g)
            est_hs[label].append(est_h)
            mre_qs[label].append(mre_q)
            mre_gs[label].append(mre_g)
            mre_hs[label].append(mre_h)

    ms = np.array(ms)
    figsize=(12, 3)
    ylim = (0, 0.07)

    plt.style.use('seaborn-colorblind')
    fig, axs = plt.subplots(nrows=1, ncols=3, constrained_layout=True, figsize=figsize)
    for i, (ests, mres) in enumerate([(est_qs, mre_qs), (est_gs, mre_gs), (est_hs, mre_hs)]):
        ylabel = r'mean relative error' if i == 0 else None
        axs[i] = plot_figure(axs[i], ks, mres, xlabel=r'$F_0$', ylabel=ylabel, ylim=ylim)
        plt.setp(axs[i], xticks=ks, xticklabels=['$2^{'+str(int(np.log2(xx)))+'}$' for xx in ks])

    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=ms.size, framealpha=1)  # put legend in the top center

    fig_name = os.path.join(FLAGS.res_dir, 'mre_m_eps{}_gamma{}{}'.format(float(epsilon), gamma, ('_morePhantom' if morePhantom else ''))) + '.pdf'
    print(fig_name)
    fig.savefig(fig_name, bbox_inches='tight')


if __name__ == '__main__':
    app.run(main)