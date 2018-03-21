import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
import math

from data_analysis_tools import *
from analytical_plots import *
from sklearn import linear_model

def make_neff_comparison(neff_data, ax, label, axis_fontsize):
    fstr_comparison = np.load('data/fstr_neff_det_comparison.npy')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('population size, N', fontweight='bold', fontsize=axis_fontsize)
    ax.set_ylabel('rate of diversity loss, $\mathbf{\Lambda}$', fontweight='bold', fontsize=axis_fontsize)
    ax.text(4E2, 1E-3, label, fontweight='bold', fontsize=12)
    ax.text(2E5, 1E-5, '$\Lambda \sim N^{\\alpha_{\mathrm{H}}}$', fontsize=10, color='k')
    ax.set_yticks([1E-4, 1E-6, 1E-8])
    ax.set_ylim([8E-9, 5E-4])

    for i, elem in enumerate(neff_data):
        fstr = fstr_comparison[i]
        N_arr = elem[0]
        kappa_arr = elem[1]

        x_BSC = list(N_arr)
        x_BSC.append(0.2*min(N_arr))
        x_BSC.append(5*max(N_arr))
        x_BSC = np.array(sorted(x_BSC))

        #fit theory
        regr = linear_model.LinearRegression()
        X = np.array(np.log(N_arr)**(-3)).reshape(len(N_arr), 1)
        Y = kappa_arr.reshape(len(kappa_arr), 1)
        regr.fit(X, Y)
        const = regr.coef_[0][0]
        const = kappa_arr[np.where(N_arr==500000)]*np.log(N_arr[np.where(N_arr==500000)])**6
        kappa_BSC = const/(np.log(x_BSC)**6)


        if len(N_arr) != 0 and len(kappa_arr) != 0:
            coeffs, res = linear_reg(np.log(N_arr), np.log(kappa_arr))
            fit = np.poly1d(coeffs)
            N_fit = [0.2*min(N_arr), 5*max(N_arr)]
            est = np.exp(fit(np.log(N_fit)))

            if fstr > -0.5:
                clr='b'
                ax.plot(N_fit, est, c=clr, lw=1, ls=':')
                ax.scatter(N_arr, kappa_arr, s=30, edgecolor=clr, facecolor='none', lw=1)
            else:
                clr='r'
                ax.plot(x_BSC, kappa_BSC, lw=1, ls='-', c=clr, label='theory, $\ln^{-6}{N}$')
                ax.plot(N_fit, est, lw=1, ls=':', c=clr, label='power law fit')
                ax.scatter(N_arr, kappa_arr, s=30, edgecolor=clr, facecolor='none', lw=1, label='simulations')

    legend_properties={'weight':'normal', 'size':6}
    ax.legend(loc='lower left', prop=legend_properties, scatterpoints=1)


def make_Lambda_phase_plot(ax, gamma_list, label, axis_fontsize, title_fontsize, markings_fontsize):

    x_min = -1.33
    x_max = 0.45
    y_min = -1.05
    y_max = -0.25 
    y_arr = np.arange(-1.0, 0., 0.001)
    x_weak = -0.25*np.ones(len(y_arr))
    x_pulled_2 = -0.5*np.ones(len(y_arr))
    x_metastable = np.zeros(len(y_arr))

    #For plotting horizontal lines
    x_full = np.arange(x_min, x_max, 0.001)
    x_pushed = np.arange(-0.5, x_max, 0.001)
    x_pulled = np.arange(x_min, -0.5, 0.001)
    x_sublinear = np.arange(x_min, -0.25, 0.001)
    x_linear = np.arange(-0.25, x_max, 0.001)

    x_weak_pushed = np.arange(-0.5, 0.0, 0.001)
    x_strong_pushed = np.arange(0.0, 0.3, 0.001)
    ones_full = -np.ones(len(x_full))#horizontal line

    ax.set_xlabel('Allee threshold, $\mathbf{\\rho^*}$', fontsize=axis_fontsize, fontweight='bold')
    ax.set_ylabel('diversity exponent, $\mathbf{\\alpha_H}$', fontsize=axis_fontsize, fontweight='bold')
    ax.text(-1.5, -0.20, label, fontweight='bold', fontsize=12)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xticks([-1.25, -0.75, -0.25, 0.25])
    ax.set_yticks([-0.25, -0.5, -0.75, -1.0])
    ax.set_xticklabels(['-1.25', '-0.75', '-0.25', '0.25'])
    ax.set_yticklabels(['-0.25', '-0.5', '-0.75', '-1'])

    #Plot extras
    ax.fill_between(x_pulled, y_min, y_max, facecolor='lightsalmon', alpha=0.5)
    ax.fill_between(x_pushed, y_min, y_max, facecolor='lightskyblue', alpha=0.5)
    ax.plot(x_full, ones_full, '--', c='k', lw=2)

    ax.scatter(gamma_list.T[0], gamma_list.T[1], s=50, facecolor='none', edgecolor='k', marker='^', lw=2)

    ax.text(-1.15, -0.21, 'pulled', fontsize=title_fontsize, fontweight='bold', color='lightsalmon')
    ax.text(-0.25, -0.21, 'pushed', fontsize=title_fontsize, fontweight='bold', color='lightskyblue')
    ax.text(-0.47, -0.4, 'semi-\n'+'pushed', fontsize=markings_fontsize, fontweight='bold', color='k')
    ax.text(0.00, -0.4, 'fully-\n'+'pushed', fontsize=markings_fontsize, fontweight='bold', color='k')


def make_vel_comparison(ax, label, axis_fontsize):
    ax.set_xlabel('time, t', fontweight='bold', fontsize=axis_fontsize)
    ax.set_ylabel('average front position, $\langle X_{\mathrm{f}} \\rangle$', fontweight='bold', fontsize=axis_fontsize)
    ax.set_xticks([0, 4000, 8000, 12000])
    ax.set_yticks([0, 400, 800, 1200])
    ax.set_ylim([0, 1500])
    ax.set_xlim([0, 9800])
    ax.text(-1200, 1550, 'A', fontweight='bold', fontsize=12)

    pulled_arr = np.load('data/velocity_N10000_gf0.01_migr0.25_B0.0_avg_cooperative.npy')
    pushed_arr = np.load('data/velocity_N10000_gf0.01_migr0.25_B10.0_avg_cooperative.npy')

    number_of_points = 20
    reduced_het_indices = (1000/number_of_points)*np.arange(number_of_points)

    x_plot = [pulled_arr.T[0][i] for i in reduced_het_indices]
    vel_plot = [pulled_arr.T[1][i] for i in reduced_het_indices]
    coeffs, res = linear_reg(x_plot, vel_plot)
    fit = np.poly1d(coeffs)
    x_fit = [0.9*x_plot[0], 1.1*x_plot[-1]] #choose range for plotting fit
    est = fit(x_fit)

    clr = 'r'
    lbl = 'pulled'
    ax.scatter(x_plot, vel_plot, s=20, edgecolor=clr, facecolor='none', lw=1, label=lbl)
    ax.plot(x_fit, est, c=clr, lw=1)

    x_plot = [pushed_arr.T[0][i] for i in reduced_het_indices]
    vel_plot = [pushed_arr.T[1][i] for i in reduced_het_indices]
    coeffs, res = linear_reg(x_plot, vel_plot)
    fit = np.poly1d(coeffs)
    x_fit = [0.9*x_plot[0], 1.1*x_plot[-1]] #choose range for plotting fit
    est = fit(x_fit)

    #Plot results
    clr = 'b'
    lbl = 'pushed'
    ax.scatter(x_plot, vel_plot, s=20, edgecolor=clr, facecolor='none', lw=1, label=lbl)
    ax.plot(x_fit, est, c=clr, lw=1)

    legend_properties={'weight':'normal', 'size':6}
    ax.legend(loc='best', prop=legend_properties, scatterpoints=1)


def make_vscaling_comparison(ax, label, axis_fontsize):
    pulled_arr = np.load('data/v_scaling_B0.0.npy')
    pushed_arr = np.load('data/v_scaling_B10.0.npy')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('population size, N', fontweight='bold', fontsize=axis_fontsize)
    ax.set_ylabel('velocity correction, $v-v_d$', fontweight='bold', fontsize=axis_fontsize)
    ax.text(2E4, 4E-5, '$v - v_d \sim N^{\\alpha_{\mathrm{v}}}$', fontsize=10, color='k')
    ax.set_xticks([1E3, 1E5, 1E7])
    ax.set_yticks([1E-2, 1E-4, 1E-6, 1E-8])
    ax.set_xlim([1E2, 1E7])
    ax.set_ylim([5E-9, 1E-1])
    ax.text(3E1, 1.6E-1, 'B', fontweight='bold', fontsize=12)

    N_arr = pulled_arr.T[0] 
    vel_arr = pulled_arr.T[1] 
    coeffs, res = linear_reg(np.log(N_arr), np.log(vel_arr))
    fit = np.poly1d(coeffs)
    N_fit = [0.2*min(N_arr), 5*max(N_arr)]
    est = np.exp(fit(np.log(N_fit)))

    x_theory = np.append(N_arr, [0.2*min(N_arr), 5*max(N_arr)])
    const = vel_arr[np.where(N_arr==500000)]*np.log(N_arr[np.where(N_arr==500000)])**3
    vel_theory = const/(np.log(x_theory)**3)

    clr='r'
    ax.plot(N_fit, est, lw=1, ls=':', c=clr, label='power law fit')
    ax.plot(N_fit, est, lw=1, ls='-', c=clr, label='theory, $\ln^{-2}{N}$')
    ax.scatter(N_arr, vel_arr, s=30, edgecolor=clr, facecolor='none', lw=1, label='simulations')

    N_arr = pushed_arr.T[0] 
    vel_arr = pushed_arr.T[1] 
    coeffs, res = linear_reg(np.log(N_arr), np.log(vel_arr))
    fit = np.poly1d(coeffs)
    N_fit = [0.2*min(N_arr), 5*max(N_arr)]
    est = np.exp(fit(np.log(N_fit)))

    clr='b'
    ax.plot(N_fit, est, c=clr, lw=1, ls=':')
    ax.scatter(N_arr, vel_arr, s=30, edgecolor=clr, facecolor='none', lw=1)


    legend_properties={'weight':'normal', 'size':6}
    ax.legend(loc='lower left', prop=legend_properties, scatterpoints=1)


def make_vel_phase_plot(ax, label, axis_fontsize, title_fontsize, markings_fontsize):
    vel_exponent = np.load('data/velocity_corr_scaling.npy')

    x_min = -0.2
    x_max = 10.3
    y_min = -1.05
    y_max = 0.
    y_arr = np.arange(-1.0, 0., 0.001)
    x_weak = 4.0*np.ones(len(y_arr))
    x_pulled_2 = 2.0*np.ones(len(y_arr))
    x_metastable = np.zeros(len(y_arr))

    x_one = np.arange(x_min, x_max, 0.001)
    y_one = -np.ones(len(x_one))

    #For plotting horizontal lines
    x_full = np.arange(x_min, x_max, 0.001)
    x_pushed = np.arange(2.0, x_max, 0.001)
    x_pulled = np.arange(x_min, 2.0, 0.001)
    x_sublinear = np.arange(x_min, 4.0, 0.001)
    x_linear = np.arange(4.0, x_max, 0.001)

    x_weak_pushed = np.arange(2.0, x_max, 0.001)
    x_strong_pushed = np.arange(0.0, 0.3, 0.001)
    ones_full = np.ones(len(x_full))#horizontal line

    ax.set_xlabel('cooperativity, B', fontsize=axis_fontsize, fontweight='bold')
    ax.set_ylabel('velocity exponent, $\mathbf{\\alpha_{\mathrm{v}}}$', fontsize=axis_fontsize, fontweight='bold')
    ax.text(-2.5, 0.05, label, fontweight='bold', fontsize=12)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    #Plot extras
    ax.fill_between(x_pushed, y_min, y_max, facecolor='lightskyblue', alpha=0.5)
    ax.fill_between(x_pulled, y_min, y_max, facecolor='lightsalmon', alpha=0.5)
    ax.plot(x_one, y_one, ls='--', c='k')
    ax.plot(x_full, ones_full, '--', c='k', lw=2)

    ax.scatter(vel_exponent.T[0], vel_exponent.T[1], s=30, marker='s', lw=2, edgecolor='k', facecolor='none')

    ax.text(-0.2, 0.03, 'pulled', fontsize=title_fontsize, fontweight='bold', color='lightsalmon')
    ax.text(4.5, 0.03, 'pushed', fontsize=title_fontsize, fontweight='bold', color='lightskyblue')
    ax.text(2.5, -0.2, 'semi-\n'+'pushed', fontsize=markings_fontsize, fontweight='bold', color='k')
    ax.text(6.5, -0.2, 'fully-\n'+'pushed', fontsize=markings_fontsize, fontweight='bold', color='k')


def FigS1_other_models():
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 12}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(cm2inch(17.8), cm2inch(9)))
    semipushed_ratio = (3./(2.*np.sqrt(2.)))

    #----------------------------------------------------------------------------------#
    predation_arr = np.load('data/predation_model.npy') 
    predation_arr = predation_arr[predation_arr[:, 1].argsort()]
    vF = min(predation_arr.T[1])
    x_min = 0.
    x_max = 0.8
    y_min = 0.42
    y_max = 0.6

    ax = fig.add_subplot(121)
    ax.set_title('predator satiation', fontsize=14, fontweight='bold')
    ax.set_xlabel('predation threshold, $\mathbf{n^*/N}$', fontsize=14, fontweight='bold')
    ax.set_ylabel('expansion velocity, $\mathbf{v}$', fontsize=14, fontweight='bold')
    ax.text(-0.1, 0.61, 'A', fontsize=14, fontweight='bold')

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.45, 0.5, 0.55])

    ax.fill_between([x_min, 0.08], y_min, y_max, facecolor='lightskyblue', alpha=0.5)
    ax.fill_between([0.08, 0.35], y_min, y_max, facecolor='lightgreen', alpha=0.5)
    ax.fill_between([0.35, x_max], y_min, y_max, facecolor='lightsalmon', alpha=0.5)

    ax.scatter(predation_arr.T[0], predation_arr.T[1], facecolor='k', edgecolor='none', s=30)
    ax.plot([x_min, x_max], vF*np.ones(2), color='r', ls='--')
    ax.plot([x_min, x_max], vF*semipushed_ratio*np.ones(2), color='r', ls='--', lw=2)
    #----------------------------------------------------------------------------------#

    #----------------------------------------------------------------------------------#
    yeast_arr = np.load('data/velocity_yeast.npy')
    vF = min(yeast_arr.T[2])
    x_min = 5E-4
    x_max = 2.0
    y_min = 0.54
    y_max = 0.6

    ax = fig.add_subplot(122)
    ax.set_title('yeast model', fontsize=14, fontweight='bold')
    ax.set_xlabel('sucrose concentration, $\mathbf{s}$', fontsize=14, fontweight='bold')
    ax.set_ylabel('expansion velocity, $\mathbf{v}$', fontsize=14, fontweight='bold')
    ax.text(1.5E-4, 0.6025, 'B', fontsize=14, fontweight='bold')
    ax.text(0.05, 0.545, 'pulled', fontsize=12, fontweight='bold', rotation=90, color='k', transform=ax.transAxes)
    ax.text(0.0045, 0.59, 'semi-pushed', fontsize=12, fontweight='bold')
    ax.text(0.86, 0.67, 'fully-pushed', fontsize=12, fontweight='bold', rotation=90, color='k', transform=ax.transAxes)

    ax.set_xscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_xticks([0.001, 0.01, 0.1, 1.0])
    ax.set_yticks([0.54, 0.56, 0.58])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.fill_between([x_min, 0.004], y_min, y_max, facecolor='lightsalmon', alpha=0.5)
    ax.fill_between([0.004, 0.4], y_min, y_max, facecolor='lightgreen', alpha=0.5)
    ax.fill_between([0.4, x_max], y_min, y_max, facecolor='lightskyblue', alpha=0.5)

    ax.scatter(yeast_arr.T[1], yeast_arr.T[2], facecolor='k', edgecolor='none', s=30)
    ax.plot([x_min, x_max], vF*np.ones(2), color='r', ls='--')
    ax.plot([x_min, x_max], vF*semipushed_ratio*np.ones(2), color='r', ls='--', lw=2)
    #----------------------------------------------------------------------------------#

    plt.tight_layout()
    plt.savefig('plots/FigS1_other_models.pdf')


def FigS3_three_classes():
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 10}
    matplotlib.rc('font', **font)
    label = ''
    axis_fontsize = 12

    B_comparison = np.load('data/B_neff_comparison_si.npy')
    neff_data = np.load('data/neff_comparison_data_si.npy')

    fig = plt.figure(figsize=(cm2inch(17.8), cm2inch(8.7)), dpi=100)
    ax = fig.add_subplot(121)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('population size, N', fontweight='bold', fontsize=axis_fontsize)
    ax.set_ylabel('rate of diversity loss, $\Lambda$', fontweight='bold', fontsize=axis_fontsize)
    ax.text(2E2, 8E-3, label, fontweight='bold', fontsize=12)
    ax.text(4E5, 1.5E-4, '$\Lambda \sim \ln^{-3}{N}$', fontsize=10, color='r')
    ax.text(4E5, 2E-5, '$\Lambda \sim N^{-0.58}$', fontsize=10, color='g')
    ax.text(4E5, 2E-6, '$\Lambda \sim N^{-1}$', fontsize=10, color='b')
    ax.set_yticks([1E-3, 1E-5, 1E-7])
    ax.set_xlim([5E2, 2E7])
    ax.set_ylim([1E-7, 3E-3])
    ax.text(2E2, 3.8E-3, 'A', fontsize=12, fontweight='bold')

    for i, elem in enumerate(neff_data):
        B = B_comparison[i]
        N_arr = elem[0]
        kappa_arr = elem[1]

        x_BSC = list(N_arr)
        x_BSC.append(0.2*min(N_arr))
        x_BSC.append(5*max(N_arr))
        x_BSC = np.array(sorted(x_BSC))

        #fit theory
        regr = linear_model.LinearRegression()
        X = np.array(np.log(N_arr)**(-3)).reshape(len(N_arr), 1)
        Y = kappa_arr.reshape(len(kappa_arr), 1)
        regr.fit(X, Y)
        const = regr.coef_[0][0]
        kappa_BSC = const/(np.log(x_BSC)**3)


        if len(N_arr) != 0 and len(kappa_arr) != 0:
            coeffs, res = linear_reg(np.log(N_arr), np.log(kappa_arr))
            fit = np.poly1d(coeffs)
            N_fit = [0.2*min(N_arr), 5*max(N_arr)]
            est = np.exp(fit(np.log(N_fit)))

            if B > 4.0:
                clr='b'
                ax.plot(N_fit, est, c=clr, lw=1, ls=':')
                ax.scatter(N_arr, kappa_arr, s=30, edgecolor=clr, facecolor='none', lw=2, label='fully-pushed, B='+str(B))
            elif B > 2.0:
                clr='g'
                ax.plot(N_fit, est, c=clr, lw=1, ls=':')
                ax.scatter(N_arr, kappa_arr, s=30, edgecolor=clr, facecolor='none', lw=2, label='semi-pushed, B='+str(B))
            else:
                clr='r'
                ax.plot(N_fit, est, lw=1, ls=':', c=clr)
                ax.scatter(N_arr, kappa_arr, s=30, edgecolor=clr, facecolor='none', lw=2, label='pulled, B='+str(B))

    legend_properties={'weight':'normal', 'size':8}
    ax.legend(loc='lower left', prop=legend_properties, scatterpoints=1)


    diff_data = np.load('data/D_comparison_data_si.npy')
    ax = fig.add_subplot(122)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('population size, N', fontweight='bold', fontsize=axis_fontsize)
    ax.set_ylabel('diffusion constant, $D_{\mathrm{f}}$', fontweight='bold', fontsize=axis_fontsize)
    ax.text(1E2, 1.5E-1, label, fontweight='bold', fontsize=12)
    ax.text(2E5, 2E-3, '$D_{\mathrm{f}} \sim \ln^{-3}{N}$', fontsize=10, color='r')
    ax.text(2E5, 3E-4, '$D_{\mathrm{f}} \sim N^{-0.55}$', fontsize=10, color='g')
    ax.text(2E5, 1E-5, '$D_{\mathrm{f}} \sim N^{-1}$', fontsize=10, color='b')
    ax.set_yticks([1E-2, 1E-4, 1E-6])
    ax.set_xlim([2E2, 2E7])
    ax.set_ylim([1E-7, 1E-1])
    ax.text(8.0E1, 1.5E-1, 'B', fontsize=12, fontweight='bold')


    for i, elem in enumerate(diff_data):
        B = elem[0][0]
        N_arr = elem[1]
        D_arr = elem[2]

        x_theory = list(N_arr)
        x_theory.append(0.2*min(N_arr))
        x_theory.append(5*max(N_arr))
        x_theory = np.array(sorted(x_theory))

        #fit theory
        regr = linear_model.LinearRegression()
        X = np.array(np.log(N_arr)**(-3)).reshape(len(N_arr), 1)
        Y = D_arr.reshape(len(D_arr), 1)
        regr.fit(X, Y)
        const = regr.coef_[0][0]
        Df_theory = const/(np.log(x_theory)**3)

        coeffs, res = linear_reg(np.log(N_arr), np.log(D_arr))
        fit = np.poly1d(coeffs)
        N_fit = [0.1*min(N_arr), 5*max(N_arr)]
        est = np.exp(fit(np.log(N_fit)))

        if B > 4.0:
            clr='b'
            ax.plot(N_fit, est, c=clr, lw=1, ls=':')
            ax.scatter(N_arr, D_arr, s=30, edgecolor=clr, facecolor='none', lw=2, label='fully-pushed, B='+str(B))
        elif B > 2.0:
            clr='g'
            ax.plot(N_fit, est, lw=1, ls=':', c=clr)
            ax.scatter(N_arr, D_arr, s=30, edgecolor=clr, facecolor='none', lw=2, label='semi-pushed, B='+str(B))
        else:
            clr='r'
            ax.plot(N_fit, est, lw=1, ls=':', c=clr)
            ax.scatter(N_arr, D_arr, s=30, edgecolor=clr, facecolor='none', lw=2, label='pulled, B='+str(B))

    legend_properties={'weight':'normal', 'size':8}
    ax.legend(loc='lower left', prop=legend_properties, scatterpoints=1)

    plt.tight_layout(pad=2)
    plt.savefig('plots/FigS3_scalings.pdf')


def FigS4_metastable_theory():
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 10}
    matplotlib.rc('font', **font)

    title_size = 14
    axis_size = 12

    Df_B7 = np.load('data/Df_theory_comparison_B7.0.npy')
    Df_B10 = np.load('data/Df_theory_comparison_B10.0.npy')
    Lm_B7 = np.load('data/Lambda_theory_comparison_B7.0.npy')
    Lm_B10 = np.load('data/Lambda_theory_comparison_B10.0.npy')


    fig = plt.figure(figsize=(cm2inch(17.8), cm2inch(17.8)))
    ax = fig.add_subplot(221)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1E3, 1E5, 1E7])
    ax.set_yticks([1E-3, 1E-5, 1E-7])
    ax.set_xlim([1E3, 1E7])
    ax.set_ylim([1E-7, 4E-3])
    ax.set_ylabel('front diffusion, $\mathbf{D_{\mathrm{f}}}$', fontweight='bold', fontsize=axis_size)
    ax.set_title('fully-pushed, B = 7', fontweight='bold', fontsize=title_size)
    ax.text(6E2, 6E-3, 'A', fontweight='bold', fontsize=14)

    N_arr = Df_B7.T[0]
    Df_arr = Df_B7.T[1]
    N_fit = np.array([0.2*min(N_arr), 5*max(N_arr)])
    ax.scatter(N_arr, Df_arr, s=50, facecolor='none', edgecolor='k', lw=2, label='simulations')
    ax.plot(N_fit, Df_theory(0.01, 0.25, 7., N_fit), lw=2, c='b', label='exact prediction')
    legend_properties={'weight':'normal', 'size':10}
    ax.legend(loc='best', prop=legend_properties, scatterpoints=1)


    ax = fig.add_subplot(222)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1E3, 1E5, 1E7])
    ax.set_yticks([1E-3, 1E-5, 1E-7])
    ax.set_xlim([1E3, 1E7])
    ax.set_ylim([1E-7, 4E-3])
    ax.set_title('fully-pushed, B = 10', fontweight='bold', fontsize=title_size)
    ax.text(6E2, 6E-3, 'B', fontweight='bold', fontsize=14)

    N_arr = Df_B10.T[0]
    Df_arr = Df_B10.T[1]
    N_fit = np.array([0.2*min(N_arr), 5*max(N_arr)])
    ax.scatter(N_arr, Df_arr, s=50, facecolor='none', edgecolor='k', lw=2, label='simulations')
    ax.plot(N_fit, Df_theory(0.01, 0.25, 10, N_fit), lw=2, c='b', label='exact prediction')
    legend_properties={'weight':'normal', 'size':10}
    ax.legend(loc='best', prop=legend_properties, scatterpoints=1)


    ax = fig.add_subplot(223)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1E3, 1E5, 1E7])
    ax.set_yticks([1E-3, 1E-5, 1E-7])
    ax.set_xlim([6E2, 1E7])
    ax.set_ylim([1E-7, 1E-3])
    ax.set_xlabel('population size, N', fontweight='bold', fontsize=axis_size)
    ax.set_ylabel('rate diversity decay, $\Lambda$', fontweight='bold', fontsize=axis_size)
    ax.text(7E2, 1.5E-3, 'C', fontweight='bold', fontsize=14)

    N_arr = Lm_B7.T[0]
    Lm_arr = Lm_B7.T[1]
    N_fit = np.array([0.2*min(N_arr), 5*max(N_arr)])
    ax.scatter(N_arr, Lm_arr, s=50, facecolor='none', edgecolor='k', lw=2, label='simulations')
    ax.plot(N_fit, Lambda_theory_cooperative(7.0, N_fit), lw=2, ls='-', label='exact prediction')
    legend_properties={'weight':'normal', 'size':10}
    ax.legend(loc='best', prop=legend_properties, scatterpoints=1)

    ax = fig.add_subplot(224)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([1E3, 1E5, 1E7])
    ax.set_yticks([1E-3, 1E-5, 1E-7])
    ax.set_xlim([6E2, 1E7])
    ax.set_ylim([1E-7, 1E-3])
    ax.set_xlabel('population size, N', fontweight='bold', fontsize=axis_size)
    ax.text(7E2, 1.5E-3, 'D', fontweight='bold', fontsize=14)


    N_arr = Lm_B10.T[0]
    Lm_arr = Lm_B10.T[1]
    N_fit = np.array([0.2*min(N_arr), 5*max(N_arr)])
    ax.scatter(N_arr, Lm_arr, s=50, facecolor='none', edgecolor='k', lw=2, label='simulations')
    ax.plot(N_fit, Lambda_theory_cooperative(10.0, N_fit), lw=2, ls='-', label='exact prediction')
    legend_properties={'weight':'normal', 'size':10}
    ax.legend(loc='best', prop=legend_properties, scatterpoints=1)

    plt.tight_layout(pad=1.2, h_pad=2.0, w_pad=1.5)
    plt.savefig('plots/FigS4_theory_metastable.pdf')


def FigS5_metastable():
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 14}
    matplotlib.rc('font', **font)

    gamma_list_r = np.load('data/gamma_list_r_allee.npy')
    D_array = np.load('data/D_array_allee.npy')

    x_min = -1.33
    x_max = 0.35
    y_min = -1.05
    y_max = 0.
    y_arr = np.arange(-1.0, 0., 0.001)
    x_weak = -0.25*np.ones(len(y_arr))
    x_pulled_2 = -0.5*np.ones(len(y_arr))
    x_metastable = np.zeros(len(y_arr))

    #For plotting horizontal lines
    x_full = np.arange(x_min, x_max, 0.001)
    x_pushed = np.arange(-0.5, x_max, 0.001)
    x_pulled = np.arange(x_min, -0.5, 0.001)
    x_sublinear = np.arange(x_min, -0.25, 0.001)
    x_linear = np.arange(-0.25, x_max, 0.001)

    x_weak_pushed = np.arange(-0.5, 0.0, 0.001)
    x_strong_pushed = np.arange(0.0, 0.3, 0.001)
    ones_full = -np.ones(len(x_full))#horizontal line

    fig = plt.figure(figsize=(cm2inch(17.8), cm2inch(20.0)), dpi=100)
    ax = fig.add_subplot(211)
    ax.set_ylabel('diversity exponent, $\mathbf{\\alpha_H}$', fontsize=14, fontweight='bold')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xticks([-1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25])
    ax.set_yticks([0.0, -0.25, -0.5, -0.75, -1.0])
    ax.set_xticklabels(['-1.25', '-1.0', '-0.75', '-0.5', '-0.25', '0.0', '0.25'])
    ax.set_yticklabels(['0', '-0.25', '-0.5', '-0.75', '-1'])


    #Plot extras
    ax.fill_between(x_pulled, y_min, y_max, facecolor='lightsalmon', alpha=0.5)
    ax.fill_between(x_pushed, y_min, y_max, facecolor='lightskyblue', alpha=0.5)
    ax.plot(x_full, ones_full, '--', c='k', lw=2)
    ax.plot([0.0, 0.0], [y_min, y_max], '--', c='k', lw=2)
    ax.plot([-1.0, -1.0], [y_min, y_max], '--', c='k', lw=2)

    ax.scatter(gamma_list_r.T[0], gamma_list_r.T[1], s=100, facecolor='none', edgecolor='k', marker='^', lw=3)

    ax.text(-0.90, 0.04, 'pulled', fontsize='16', fontweight='bold', color='lightsalmon')
    ax.text(-0.25, 0.04, 'pushed', fontsize='16', fontweight='bold', color='lightskyblue')
    ax.text(-0.40, -0.2, 'semi-\n'+'pushed', fontsize='12', fontweight='bold', color='k')
    ax.text(0.02, -0.2, 'fully-\n'+'pushed', fontsize='12', fontweight='bold', color='k')
    ax.text(0.03, -0.55, 'metastable state\n'+'strong Allee effect', fontsize='10', fontweight='bold', color='w', rotation=90)
    ax.text(-0.10, -0.58, 'unstable state\n'+'weak Allee effect', fontsize='10', fontweight='bold', color='w', rotation=90)
    ax.text(-0.97, -0.57, 'unstable state\n'+'weak Allee effect', fontsize='10', fontweight='bold', color='w', rotation=90)
    ax.text(-1.12, -0.35, 'unstable state, Fisher-like\n'+'no Allee effect', fontsize='10', fontweight='bold', color='w', rotation=90)


    ax = fig.add_subplot(212)
    ax.set_xlabel('Allee threshold, $\mathbf{\\rho^*}$', fontsize=14, fontweight='bold')
    ax.set_ylabel('diffusion exponent, $\mathbf{\\alpha_D}$', fontsize=14, fontweight='bold')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xticks([-1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25])
    ax.set_yticks([0.0, -0.25, -0.5, -0.75, -1.0])
    ax.set_xticklabels(['-1.25', '-1.0', '-0.75', '-0.5', '-0.25', '0.0', '0.25'])
    ax.set_yticklabels(['0', '-0.25', '-0.5', '-0.75', '-1'])


    #Plot extras
    ax.fill_between(x_pulled, y_min, y_max, facecolor='lightsalmon', alpha=0.5)
    ax.fill_between(x_pushed, y_min, y_max, facecolor='lightskyblue', alpha=0.5)
    ax.plot(x_full, ones_full, '--', c='k', lw=2)
    ax.plot([0.0, 0.0], [y_min, y_max], '--', c='k', lw=2)
    ax.plot([-1.0, -1.0], [y_min, y_max], '--', c='k', lw=2)

    ax.scatter(D_array.T[0], D_array.T[1], s=100, facecolor='k', edgecolor='none', marker='x', lw=3)

    ax.text(-0.40, -0.2, 'semi-\n'+'pushed', fontsize='12', fontweight='bold', color='k')
    ax.text(0.02, -0.2, 'fully-\n'+'pushed', fontsize='12', fontweight='bold', color='k')
    ax.text(0.03, -0.55, 'metastable state\n'+'strong Allee effect', fontsize='10', fontweight='bold', color='w', rotation=90)
    ax.text(-0.10, -0.58, 'unstable state\n'+'weak Allee effect', fontsize='10', fontweight='bold', color='w', rotation=90)
    ax.text(-0.97, -0.57, 'unstable state\n'+'weak Allee effect', fontsize='10', fontweight='bold', color='w', rotation=90)
    ax.text(-1.12, -0.35, 'unstable state, Fisher-like\n'+'no Allee effect', fontsize='10', fontweight='bold', color='w', rotation=90)


    plt.tight_layout(pad=1.2)
    plt.savefig('plots/FigS5_phasediagram.pdf')


def FigS6_velocity_corrections():
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 8}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(cm2inch(17.8), cm2inch(5.9)), dpi=100)
    ax = fig.add_subplot(131)
    make_vel_comparison(ax, 'A', 9)
    ax = fig.add_subplot(132)
    make_vscaling_comparison(ax, 'B', 9)
    ax = fig.add_subplot(133)
    make_vel_phase_plot(ax, 'C', 9, 8, 9)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig('plots/FigS6_velocity_full.pdf')


def FigS7_no_demographic_noise():
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 6}
    matplotlib.rc('font', **font)

    het_det_data = np.load('data/hetero_plot_det_data.npy')
    lambda_comparison_det = np.load('data/neff_comparison_det_data.npy')
    gamma_list = np.load('data/gamma_list_r_allee_det.npy')

    fig = plt.figure(figsize=(cm2inch(17.8), cm2inch(5.9)), dpi=100)
    ax = fig.add_subplot(131)
    make_het_comparison(het_det_data, ax, 'A', 8)
    ax = fig.add_subplot(132)
    make_neff_comparison(lambda_comparison_det, ax, 'B', 8)
    ax = fig.add_subplot(133)
    make_Lambda_phase_plot(ax, gamma_list, 'C', 8, 8, 6)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.9)
    plt.savefig('plots/FigS7_det_diversity.pdf')


def FigS8_deterministic_comparison():
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 14}
    matplotlib.rc('font', **font)


    fig = plt.figure(figsize=(cm2inch(18.4), cm2inch(18.4)))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('population size, N', fontweight='bold')
    ax.set_ylabel('rate of diversity loss, $\mathbf{\Lambda}$', fontweight='bold')

    fstr = -0.3
    (N_det, Lambda_det) = np.load('data/deterministic_comparison_detdata.npy')
    N_fit = np.array([0.2*min(N_det), 5*max(N_det)])
    coeffs, res = linear_reg(np.log(N_det), np.log(Lambda_det))
    fit = np.poly1d(coeffs)
    alpha_f = fluctuations_exponent(fstr)[-1]
    det_est = (Lambda_det[0]/N_det[0]**alpha_f)*(N_fit**alpha_f)

    (N_stoch, Lambda_stoch) = np.load('data/deterministic_comparison_stochdata.npy')
    coeffs, res = linear_reg(np.log(N_stoch), np.log(Lambda_stoch))
    fit = np.poly1d(coeffs)
    alpha_mf = meanfield_exponent(fstr)[-1]
    stoch_est = (Lambda_stoch[0]/N_stoch[0]**alpha_mf)*(N_fit**alpha_mf)

    ax.scatter(N_det, Lambda_det, s=100, lw=2, marker='D', edgecolor='darkolivegreen', facecolor='none', label='deterministic-front model, $\mathbf{\gamma_\mathrm{n} = 0}$')
    ax.plot(N_fit, det_est, lw=2, c='darkolivegreen', label='deterministic-front theory, $\mathbf{\zeta_c = \\frac{1}{q}\ln{N}}$')
    ax.scatter(N_stoch, Lambda_stoch, s=100, lw=2, marker='o', edgecolor='lawngreen', facecolor='none', label='fluctuating-front model, $\mathbf{\gamma_\mathrm{n} = 0}$')
    ax.plot(N_fit, stoch_est, lw=2, c='lawngreen', label='fluctuating-front theory, $\mathbf{\zeta_c = \\frac{1}{k}\ln{N}}$')

    ax.legend(loc='upper right', scatterpoints=1, fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/FigS8_deterministic_comparison.pdf')


def FigS9_theory_comparison():
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 12}
    matplotlib.rc('font', **font)

    dx = 0.001

    fig = plt.figure(figsize=(12,5.5))

    ax = fig.add_subplot(121)
    ax.set_xlabel('Allee threshold, $\mathbf{\\rho^*}$', fontsize=16, fontweight='bold')
    ax.set_ylabel('velocity exponent, $\mathbf{\\alpha_v}$', fontsize=16, fontweight='bold')
    ax.set_xlim(-1.05, 0.55)
    ax.set_ylim(-2.1, 0.1)
    ax.set_xticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5])
    ax.text(-1.1, 0.2, 'A', fontsize=18, fontweight='bold')

    rhop_arr = np.arange(-1., -0.5, dx)
    detp_arr = np.zeros(len(rhop_arr))
    stochp_arr = np.zeros(len(rhop_arr))

    rhou_arr = np.arange(-0.5, 0.0, dx)
    detu_arr = - ((-1./rhou_arr) - 2.)/(-1./rhou_arr)
    rhom_arr = np.arange(0.0, 0.5, dx)
    detm_arr = - det_metastable(1. - 2.*rhom_arr)

    rhosemi_arr = np.arange(-0.5, -0.25, dx)
    stochsemi_arr = - ((-1./rhosemi_arr) - 2.)/2.
    rhofully_arr = np.arange(-0.25, 0.5, dx)
    stochfully_arr = - np.ones(len(rhofully_arr))

    ax.plot(rhop_arr, detp_arr, ls='-', lw=3, c='g', label='deterministic fronts w/ cutoff')
    ax.plot(rhou_arr, detu_arr, ls='-', lw=3, c='g')
    ax.plot(rhom_arr, detm_arr, ls='-', lw=3, c='g')
    ax.plot(rhop_arr, stochp_arr, ls='-', lw=3, c='r', label='stochastic theory')
    ax.plot(rhosemi_arr, stochsemi_arr, ls='-', lw=3, c='r')
    ax.plot(rhofully_arr, stochfully_arr, ls='-', lw=3, c='r')

    ax.legend(loc='lower left', fontsize=12)

    ax = fig.add_subplot(122)
    ax.set_xlabel('Allee threshold, $\mathbf{\\rho^*}$', fontsize=16, fontweight='bold')
    ax.set_ylabel('diversity exponent, $\mathbf{\\alpha_H}$', fontsize=16, fontweight='bold')
    ax.set_xlim(-1.05, 0.55)
    ax.set_ylim(-1.1, 0.1)
    ax.set_xticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5])
    ax.text(-1.1, 0.15, 'B', fontsize=18, fontweight='bold')

    rhop_arr = np.arange(-1., -0.5, dx)
    detp_arr = np.zeros(len(rhop_arr))
    stochp_arr = np.zeros(len(rhop_arr))

    rhosemi_arr = np.arange(-0.5, -0.25, dx)
    stochsemi_arr = - ((-1./rhosemi_arr) - 2.)/2.
    detsemi_arr = - (2 - 4./(-1./rhosemi_arr))
    rhofully_arr = np.arange(-0.25, 0.5, dx)
    stochfully_arr = - np.ones(len(rhofully_arr))

    ax.plot(rhop_arr, detp_arr, ls='-', lw=3, c='g', label='deterministic fronts')
    ax.plot(rhosemi_arr, detsemi_arr, ls='-', lw=3, c='g')
    ax.plot(rhofully_arr, stochfully_arr, ls='-', lw=3, c='g')
    ax.plot(rhop_arr, stochp_arr, ls='-', lw=3, c='r', label='fluctuating fronts')
    ax.plot(rhosemi_arr, stochsemi_arr, ls='-', lw=3, c='r')
    ax.plot(rhofully_arr, stochfully_arr, ls='-', lw=3, c='r')


    ax.legend(loc='upper right', fontsize=12)
    plt.tight_layout(pad=3.0)
    plt.savefig('plots/FigS9_comparison.pdf')


if __name__=='__main__':
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 12}
    matplotlib.rc('font', **font)

    FigS1_other_models()
    FigS2_ancestry()
    FigS3_three_classes()
    FigS4_metastable_theory()
    FigS5_metastable()
    FigS6_velocity_corrections()
    FigS7_no_demographic_noise()
    FigS8_deterministic_comparison()
    FigS9_theory_comparison()

    plt.show()
