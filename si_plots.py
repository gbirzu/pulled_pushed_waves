import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
import math

#from data_rate_analysis_tools import *
from data_analysis_tools import *
from analytical_plots import *
from sklearn import linear_model

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

def metastable_diversity_loss(label, axis_fontsize, output_path):
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 9}
    matplotlib.rc('font', **font)

    allee_data = np.load('data/metastable_diversity_allee.npy')
    quartic_data = np.load('data/metastable_diversity_quartic.npy')

    fig = plt.figure(figsize=(cm2inch(11.4), cm2inch(8.7)), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('population size, N', fontweight='bold', fontsize=axis_fontsize)
    ax.set_ylabel('rate of diversity loss, $\Lambda$', fontweight='bold', fontsize=axis_fontsize)
    ax.text(2E2, 8E-3, label, fontweight='bold', fontsize=12)
    #ax.set_xticks([1E-3, 1E-5, 1E-7])
    #ax.set_yticks([1E-3, 1E-5, 1E-7])
    #ax.set_ylim([1E-7, 1E-3])

    colors = iter(cm.rainbow(np.linspace(0, 1, len(allee_data)+len(quartic_data)))) 
    const = allee_data[0][1][0] #fix value of first point
    for i, elem in enumerate(allee_data):
        fstr = elem[2][0]
        Lambda_arr = const*elem[1]/elem[1][0]
        ax.scatter(elem[0], Lambda_arr, s=30, marker='^', edgecolor=next(colors), facecolor='none', lw=2, label='Allee effect model, $\\rho^*=$'+str(fstr))
    for i, elem in enumerate(quartic_data):
        fstr = elem[2][0]
        Lambda_arr = const*elem[1]/elem[1][0]
        ax.scatter(elem[0], Lambda_arr, s=30, marker='v', edgecolor=next(colors), facecolor='none', lw=2, label='Highly nonlinear model, $\\rho^*=$'+str(fstr))

    guide_x = np.array([0.5*min(allee_data[0][0]), 2*max(allee_data[0][0])])
    guide_y = 2.*const*allee_data[0][0][0]/guide_x
    ax.plot(guide_x, guide_y)

    legend_properties={'weight':'normal', 'size':6}
    ax.legend(loc='lower left', prop=legend_properties, scatterpoints=1)
    plt.tight_layout()
    plt.savefig(output_path+'metastable.pdf')


def FigS3_three_classes(neff_data, label, axis_fontsize):
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 10}
    matplotlib.rc('font', **font)

    B_comparison = np.load('data/B_neff_comparison_si.npy')

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
    #ax.set_xticks([1E-3, 1E-5, 1E-7])
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
        #kappa_BSC = ((np.pi**2)*2*0.01)/(np.log(x_BSC)**3)

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
            print 'B = ', B, ', alpha_H = ', coeffs[0]

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
    #ax.set_xticks([1E-3, 1E-5, 1E-7])
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
        #Df_theory = (2.*0.01*np.pi**4/3.)/(profile_decay(0.01, 0.25, B)**2*np.log(x_theory)**3)

        #fit theory
        regr = linear_model.LinearRegression()
        X = np.array(np.log(N_arr)**(-3)).reshape(len(N_arr), 1)
        Y = D_arr.reshape(len(D_arr), 1)
        regr.fit(X, Y)
        const = regr.coef_[0][0]
        Df_theory = const/(np.log(x_theory)**3)

        coeffs, res = linear_reg(np.log(N_arr), np.log(D_arr))
        print 'B = ', B, ', alpha_D = ', coeffs[0]
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


def theory_metastable(all_data, axis_size, output_path):
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 8}
    matplotlib.rc('font', **font)

    sort_params = [['g', 0.01], ['m', 0.25]]
    new_data = all_data.filter_data(all_data.experiment_list, sort_params)

    fig = plt.figure(figsize=(cm2inch(17.8), cm2inch(17.8)))
    ax_counter = 1
    label_counter = 0
    labels = ['A', 'B', 'C', 'D']

    for B in all_data.B_list:
        filter_params = [['B', B]]
        fit_data = all_data.filter_data(new_data, filter_params)
        N_arr = np.array([elem.N for elem in fit_data])
        kappa_arr = np.array([elem.kappa for elem in fit_data])
    
        if len(N_arr) != 0 and len(kappa_arr) != 0:
            if B==7. or B==10.:
                ax = fig.add_subplot(2,2,ax_counter)
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xticks([1E3, 1E5, 1E7])
                ax.set_yticks([1E-3, 1E-5, 1E-7])
                ax.set_xlim([6E2, 1E7])
                ax.set_ylim([1E-7, 1E-3])
                ax.set_xlabel('population size, N', fontweight='bold', fontsize=axis_size)
                if ax_counter==1:
                    ax.set_ylabel('rate diversity decay, $\Lambda$', fontweight='bold', fontsize=axis_size)
                ax.set_title('fully-pushed, B = ' + str(B), fontweight='bold', fontsize=9)
                ax.text(5E2, 3E-3, labels[label_counter], fontweight='bold', fontsize=12)


                ax.scatter(N_arr, kappa_arr, s=50, facecolor='none', edgecolor='k', lw=2, label='simulations')
                if B > 4.0:
                    ax.plot(N_fit, Lambda_theory(B)/N_fit, lw=2, ls='-', label='exact prediction')
                else:
                    ax.plot(N_fit, est)
                legend_properties={'weight':'normal', 'size':6}
                ax.legend(loc='best', prop=legend_properties, scatterpoints=1)

                ax_counter += 1
                label_counter += 1


            gamma_list_r.append([B, coeffs[0]])
            gamma_list_v.append([v_F(0.01, 0.125, B), v_F_cutoff(0.01, 0.125, B, N_arr[-1]), v_avg, coeffs[0], B])
                #gamma_list_v.append([v_Fquartic(0.01, 0.125, B), v_Fquartic_cutoff(0.01, 0.125, B, N_arr[-1]), v_avg, coeffs[0], B])
            print B, coeffs[0]
            
    plt.tight_layout(pad=1.2)
    plt.savefig(output_path+'/Lambda_theory_comparison.pdf')


def FigS5_metastable(gamma_list_r, D_array):
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 14}
    matplotlib.rc('font', **font)

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

def fit_heterozygosity(x_het, het, surv):
    X_INIT = min(min(range(len(het)), key=lambda i:abs(het[i] - 0.1)), 1000) #find initial fitting point
    X_FIN = 0
    while (surv[X_FIN] > 0.05 and X_FIN < len(het) - 1):
        X_FIN += 1 #find final fitting point

    if (X_FIN - X_INIT <= 400):
        X_INIT = max(0, X_FIN - 400)

    X_THIRD = int((X_FIN - X_INIT)/3)
    xh = x_het[X_INIT:X_FIN]
    het_fit = het[X_INIT:X_FIN]
    coeffs, r_sq = linear_reg(xh, np.log(het_fit))#Do fit
    return xh, het_fit, coeffs, r_sq


def make_het_comparison(het_data, ax, label, axis_fontsize):
    fstr_comparison = np.load('data/fstr_neff_det_comparison.npy')
    het_data = np.flipud(het_data)

    ax.set_yscale('log')
    ax.ticklabel_format(style='sci', scilimit=(-2,2), axis='x')
    ax.set_xlabel('time, t', fontweight='bold', fontsize=axis_fontsize)
    ax.set_ylabel('heterozygosity, H', fontweight='bold', fontsize=axis_fontsize)
    ax.text(-3000, 2.1E0, label, fontweight='bold', fontsize=12)
    ax.text(13000, 3E-1, '$H \sim e^{-\Lambda t}$', fontsize=10, color='k')
    ax.set_xticks([0, 10000, 20000])
    ax.set_yticks([1E-1, 1])
    ax.set_ylim([1E-1, 1.8E0])

    number_of_points = 20
    reduced_het_indices = (1000/number_of_points)*np.arange(number_of_points)

    for elem in het_data:
        deme = elem[0][0]
        fstr = elem[0][1]
        x_het = elem[1].T[0]
        het = elem[1].T[1]
        surv = elem[1].T[2]

        x_plot = [x_het[i] for i in reduced_het_indices]
        het_plot = [het[i] for i in reduced_het_indices]
                
        xh, het_fit, coeffs, r_sq = fit_heterozygosity(x_het, het, surv)
        fit = np.poly1d(coeffs)
        x_fit = [x_het[50], 1.1*x_het[-1]] #choose range for plotting fit
        est = np.exp(fit(x_fit))

        #Plot results
        if fstr > -0.5:
            clr = 'b'
            lbl = 'pushed'
        else:
            clr = 'r'
            lbl = 'pulled'
        ax.scatter(x_plot, het_plot, s=20, edgecolor=clr, facecolor='none', lw=1, label=lbl)
        ax.plot(x_fit, est, c=clr, lw=1)
    ax.set_xlim([0, 22000])

    legend_properties={'weight':'normal', 'size':6}
    ax.legend(loc='best', prop=legend_properties, scatterpoints=1)


def make_neff_comparison(neff_data, ax, label, axis_fontsize):
    fstr_comparison = np.load('data/fstr_neff_det_comparison.npy')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('population size, N', fontweight='bold', fontsize=axis_fontsize)
    ax.set_ylabel('rate of diversity loss, $\mathbf{\Lambda}$', fontweight='bold', fontsize=axis_fontsize)
    ax.text(4E2, 1E-3, label, fontweight='bold', fontsize=12)
    ax.text(2E5, 1E-5, '$\Lambda \sim N^{\\alpha_{\mathrm{H}}}$', fontsize=10, color='k')
    #ax.set_xticks([1E-3, 1E-5, 1E-7])
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
        #kappa_BSC = ((np.pi**2)*2*0.01)/(np.log(x_BSC)**3)

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
    #ax.set_xticks([-1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25])
    ax.set_xticks([-1.25, -0.75, -0.25, 0.25])
    ax.set_yticks([-0.25, -0.5, -0.75, -1.0])
    #ax.set_xticklabels(['-1.25', '-1.0', '-0.75', '-0.5', '-0.25', '0.0', '0.25'])
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


def fig4_Lambda_summary(het_data, neff_data, gamma_list):
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 6}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(cm2inch(17.8), cm2inch(5.9)), dpi=100)
    ax = fig.add_subplot(131)
    make_het_comparison(het_data, ax, 'A', 8)
    ax = fig.add_subplot(132)
    make_neff_comparison(neff_data, ax, 'B', 8)
    ax = fig.add_subplot(133)
    make_Lambda_phase_plot(ax, gamma_list, 'C', 8, 8, 6)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.9)
    plt.savefig('plots/FigS7_det_diversity.pdf')

def FigS7_no_demographic_noise():
    het_det_data = np.load('data/hetero_plot_det_data.npy')
    lambda_comparison_det = np.load('data/neff_comparison_det_data.npy')
    gamma_list = np.load('data/gamma_list_r_allee_det.npy')
    fig4_Lambda_summary(het_det_data, lambda_comparison_det, gamma_list)


def make_vel_comparison(ax, label, axis_fontsize):
    ax.set_xlabel('time, t', fontweight='bold', fontsize=axis_fontsize)
    ax.set_ylabel('average front position, $\langle X_{\mathrm{f}} \\rangle$', fontweight='bold', fontsize=axis_fontsize)
    #ax.text(-2000, 1.5E1, label, fontweight='bold', fontsize=12)
    #ax.text(7000, 5E-2, '$H \sim e^{-\Lambda t}$', fontsize=10, color='k')
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
    #ax.text(2E2, 8E-3, label, fontweight='bold', fontsize=12)
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
    #ax.text(2.1, -0.2, 'semi-pushed', fontsize=markings_fontsize, fontweight='bold', color='k')
    #ax.text(6.0, -0.2, 'fully-pushed', fontsize=markings_fontsize, fontweight='bold', color='k')
    ax.text(2.5, -0.2, 'semi-\n'+'pushed', fontsize=markings_fontsize, fontweight='bold', color='k')
    ax.text(6.5, -0.2, 'fully-\n'+'pushed', fontsize=markings_fontsize, fontweight='bold', color='k')


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

def Df_theory(r, m, B, N):
    D = m/2.
    x = 2./B
    lm = profile_decay(r, 2*D, B)

    const = 3/(20*np.pi*N*lm)
    #trig = (math.sin(2*np.pi/B))**2/math.sin(4*np.pi/B)
    trig = math.tan(2*np.pi/B)
    #algebr = (2*x*(1+2*x)*(2+2*x)*(3+2*x)*(4+2*x))/(x*(1+x)*(2+x))**2
    algebr = (B*(B+4.)*(3*B+4.))/((B+1.)*(B+2.))
    return const*trig*algebr

def Lambda_theory(B, N):
    return np.sqrt(0.01*B/0.25)*4*np.pi*np.tan(2*np.pi/B)/(B+4)/N

def FigS4_metastable_theory(title_size, axis_size):
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 10}
    matplotlib.rc('font', **font)

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
    ax.plot(N_fit, Lambda_theory(7.0, N_fit), lw=2, ls='-', label='exact prediction')
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
    ax.plot(N_fit, Lambda_theory(10.0, N_fit), lw=2, ls='-', label='exact prediction')
    legend_properties={'weight':'normal', 'size':10}
    ax.legend(loc='best', prop=legend_properties, scatterpoints=1)

    plt.tight_layout(pad=1.2, h_pad=2.0, w_pad=1.5)
    plt.savefig('plots/FigS4_theory_metastable.pdf')

def det_metastable(a):
    return 2*np.sqrt(a**2 + 4*(1. - a))/(a + np.sqrt(a**2 + 4*(1. - a)))

def FigS9_theory_comparison(dx):
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 12}
    matplotlib.rc('font', **font)


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
    #stochsemi_arr = - stoch_semi(-1./rhosemi_arr) 
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

    gamma_list_r = np.load('data/gamma_list_r.npy')
    gamma_list_v = np.load('data/gamma_list_v.npy')
    het_data = np.load('data/hetero_plot_data.npy')
    D_array = np.load('data/D_array.npy')
    gamma_list_r_allee = np.load('data/gamma_list_r_allee.npy')
    D_array_allee = np.load('data/D_array_allee.npy')
    lambda_comparison_arr = np.load('data/neff_comparison_data_si.npy')

    FigS1_other_models()
    FigS2_ancestry()
    FigS3_three_classes(lambda_comparison_arr, '', 12)
    FigS4_metastable_theory(14, 12)
    FigS5_metastable(gamma_list_r_allee, D_array_allee)
    FigS6_velocity_corrections()
    FigS7_no_demographic_noise()
    FigS9_theory_comparison(0.001)

    plt.show()
