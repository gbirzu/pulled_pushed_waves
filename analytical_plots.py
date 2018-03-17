import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from matplotlib.colorbar import Colorbar
from plotting_tools import *


def profile(gf, migr, fstr, x):
    D = migr/2.
    if fstr > -0.5:
        prof = 1./(1. + np.exp(np.sqrt(gf/(2.*D))*x))
    else:
        prof = 1./(1. + np.exp(np.sqrt(gf*abs(fstr)/D)*x))
    return prof

def fixation_const(gf, migr, fstr, x_min, x_max, dx):
    x_arr = np.arange(x_min, x_max, dx)
    c_arr = profile(gf, migr, fstr, x_arr)
    v = velocity(gf, migr, fstr)
    D = migr/2.

    prelim_prob = c_arr**2*np.exp(v*x_arr/D)
    const = integrate.simps(prelim_prob, x_arr)
    return const

def fixation_probability(gf, migr, fstr, x_min, x_max, dx, x):
    c = profile(gf, migr, fstr, x)
    v = velocity(gf, migr, fstr)
    D = migr/2.
    const = fixation_const(gf, migr, fstr, x_min, x_max, dx)
    prob = c**2*np.exp(v*x/D)/const
    return prob

def Fig1_growth(labels_flag, label_size):
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 8}
    matplotlib.rc('font', **font)

    gf = 0.01
    gf_pushed = 4*gf
    m = 1.25
    dx = 0.01
    x_min = -40
    x_max = 50
    f_pulled = -1.0
    f_pushed = -0.08

    min_f = -0.8
    max_f = -0.2
    f_arr = np.arange(min_f, max_f, 0.01)
    f_pulled_arr = np.arange(min_f, -0.5, 0.001)
    f_pushed_arr = np.arange(-0.5, max_f, 0.001)
    v_arr = np.array([velocity(gf, m, f) for f in f_arr])
    vF_arr = np.array([velocity_Fisher(gf, m, f) for f in f_arr])
    v_ratio_arr = v_arr/vF_arr

    min_v = 0.95*min(v_ratio_arr)
    max_v = 1.05*max(v_ratio_arr)
       
    y_pp_transition = np.arange(min_v, max_v, 0.001)
    x_pp_transition = -0.5*np.ones(len(y_pp_transition))

    x_array = np.arange(x_min, x_max, 0.01)
    pulled_profile = np.array([profile(gf, m, f_pulled, x) for x in x_array])
    pulled_growth = np.array(growth(gf, f_pulled, pulled_profile))
    pushed_profile = np.array([profile(gf_pushed, m, f_pushed, x) for x in x_array])
    pushed_growth = np.array(growth(gf_pushed, f_pushed, pushed_profile))


    fig = plt.figure(figsize=(cm2inch(17.8),cm2inch(5.8)))

    ax = fig.add_subplot(131)
    ax.set_ylim([min_v, max_v])
    ax.set_yticks([])
    ax.set_xlabel('cooperativity, B', fontsize=label_size, fontweight='bold')
    ax.set_ylabel('velocity, $\mathbf{v}$', fontsize=label_size, fontweight='bold')

    if labels_flag != 0:
        ax.set_xticks([])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8) 

        ax.set_xticks([-0.8, -0.5, -0.2])
        ax.set_xticklabels([0, 2, 4], fontsize=8)
    else:
        ax.set_xticks([])
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    ax.fill_between(f_pulled_arr, min_v, max_v, facecolor='lightsalmon', alpha=0.5)
    ax.fill_between(f_pushed_arr, min_v, max_v, facecolor='lightskyblue', alpha=0.5)

    ax.plot(f_arr, v_arr/vF_arr, ls='-', lw=2, c='k')
    ax.plot(f_arr, np.ones(len(f_arr)), ls='--', lw=2, c='k')
    ax.text(1.08*min_f, 1.007*max_v, 'A', fontsize=12, fontweight='bold', color='k')
    ax.text(-0.3, 0.98, '$v_{\mathrm{F}}$', fontsize=14, fontweight='bold', color='k')
    ax.text(-0.72, 1.010*max_v, 'pulled', fontsize=12, fontweight='bold', color='lightsalmon')
    ax.text(-0.45, 1.010*max_v, 'pushed', fontsize=12, fontweight='bold', color='lightskyblue')

    gf = 0.01
    m = 0.25
    dx = 0.01
    x_min = -30
    x_max = 30
    f_pulled = -1.0
    f_pushed = -0.08

    x_array = np.arange(x_min, x_max, 0.1)
    pulled_profile = np.array([profile(gf, m, f_pulled, x) for x in x_array])
    pulled_growth = np.array(growth(gf, f_pulled, pulled_profile))
    pushed_profile = np.array([profile(gf, m, f_pushed, x) for x in x_array])
    pushed_growth = np.array(growth(gf, f_pushed, pushed_profile))

    pulled_growth_fill = np.array([[elem]*len(pulled_profile) for elem in pulled_growth])
    pushed_growth_fill = np.array([[elem]*len(pushed_profile) for elem in pushed_growth])
    max_growth = max(pushed_growth)
    min_growth = min(pushed_growth)


    ax1 = fig.add_subplot(132)
    ax1.set_title('pulled', fontsize=12, fontweight='bold')
    ax1.set_xlabel('position, x', fontsize=label_size, fontweight='bold')
    ax1.set_ylabel('population density, n', fontsize=label_size, fontweight='bold')
    ax1.set_xticks([-20, 0, 20, 40])
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim([0.0, 1.1])
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 

    ax1.contourf(x_array, pulled_profile, pulled_growth_fill.T, 200, cmap=plt.cm.winter)
    ax1.fill_between(x_array, pulled_profile, y2=1.01*max(pulled_profile), color='w')
    ax1.text(1.20*x_min, 1.03*1.1, 'B', fontsize=12, fontweight='bold', color='k')

    ax1 = fig.add_subplot(133)
    ax1.set_title('pushed', fontsize=12, fontweight='bold')
    ax1.set_xlabel('position, x', fontsize=label_size, fontweight='bold')
    ax1.set_ylabel('population density, n', fontsize=label_size, fontweight='bold')
    ax1.set_xticks([-20, 0, 20, 40])
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_ylim([0.0, 1.1])
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 

    cax = ax1.contourf(x_array, pushed_profile, pushed_growth_fill.T, 200, cmap=plt.cm.winter)
    ax1.fill_between(x_array, pushed_profile, y2=1.01*max(pushed_profile), color='w')
    ax1.text(1.25*x_min, 1.03*1.1, 'C', fontsize=12, fontweight='bold', color='k')

    cbar = fig.colorbar(cax, ticks=[min_growth, max_growth])
    cbar.ax.set_yticklabels(['low', 'high'])

    ax1.text(40, 0.82, 'growth rate', fontsize=10, rotation=90)

    plt.tight_layout(pad=1.5, h_pad=1.0)
    plt.savefig('plots/Fig1_growth.tiff', dpi=500)



def Fig2_fixation(label_size, markings_size):
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 8}
    matplotlib.rc('font', **font)


    fig = plt.figure(figsize=(cm2inch(17.8), cm2inch(6.0)))

    gf = 3E-4
    m = 0.25
    dx = 0.01
    x_min = -100
    x_max = 400
    f = -1.0
    vt = 200.

    x_array = np.arange(x_min, x_max, 0.002)
    first_profile = np.array([profile(gf, m, f, x) for x in x_array])
    second_profile = np.array([profile(gf, m, f, x - vt) for x in x_array])

    ax = plt.subplot2grid((2, 3), (0, 0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([0.0, 1.1])
    ax.plot(x_array, first_profile, lw=1, c='k')
    ax.text(-150, 1.20, 'A', fontsize=markings_size, fontweight='bold', color='k')
    ax.text(5, 1.18, 'fixation event', fontsize=markings_size, fontweight='bold', color='k')

    ax = plt.subplot2grid((2, 3), (1, 0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([0.0, 1.1])
    ax.plot(x_array, second_profile, lw=1, c='k')
    ax.text(45, -0.25, 'position, x', fontsize=label_size, fontweight='bold', color='k')


    gf = 0.01
    gf_pushed = 0.04
    m = 1.25
    dx = 0.01
    x_min = -40
    x_max = 100
    f_pulled = -1.0
    f_pushed = -0.08

    x_array = np.arange(x_min, x_max, 0.01)
    x_fix = np.arange(x_min, x_max, 0.1)
    pulled_profile = np.array([profile(gf, m, f_pulled, x) for x in x_fix])
    pulled_fixation = np.array([fixation_probability(gf, m, f_pulled, x_min, x_max, dx, x) for x in x_fix])
    pushed_profile = np.array([profile(gf_pushed, m, f_pushed, x) for x in x_fix])
    pushed_fixation = np.array([fixation_probability(gf_pushed, m, f_pushed, x_min, x_max, dx, x) for x in x_fix])


    #ax1 = fig.add_subplot(132)
    ax1 = plt.subplot2grid((2, 3), (0, 1), rowspan=2)
    ax1.set_title('pulled', fontsize=markings_size, fontweight='bold', color='k')
    ax1.set_xlabel('position, x', fontsize=label_size, fontweight='bold')
    ax1.set_ylabel('fixation probability', color='g', fontsize=label_size, fontweight='bold')
    ax1.set_ylim([0.0, 0.05])
    ax1.set_yticks([])
    ax1.ticklabel_format(style='sci',scilimits=(-2,2),axis='y')
    ax1.get_yaxis().set_tick_params(direction='in', pad=5)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
        tick.label.set_color('g') 
    ax1.plot(x_fix, pulled_fixation, lw=4, c='g', ls='-')
    ax1.text(15, 0.014,'fixed ancestor', fontsize=markings_size, fontweight='bold', color='g')
    ax1.text(-55, 0.053, 'B', fontsize=markings_size, fontweight='bold', color='k')

    ax2 = ax1.twinx()
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylim([0.0, 1.1])
    for tick in ax2.yaxis.get_major_ticks():
        tick.label2.set_fontsize(8) 
    ax2.plot(x_fix, pulled_profile, lw=4, c='k', ls='--')
    ax2.text(-8, 0.9, 'density', fontsize=markings_size, fontweight='bold', color='k')

    ax1 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
    ax1.set_title('pushed', fontsize=markings_size, fontweight='bold', color='k')
    ax1.set_xlabel('position, x', fontsize=label_size, fontweight='bold')
    ax1.set_ylim([0.0, 0.08])
    ax1.set_yticks([])
    ax1.ticklabel_format(style='sci',scilimits=(-2,2),axis='y')
    ax1.get_yaxis().set_tick_params(direction='in', pad=5)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
        tick.label.set_color('g') 
    ax1.plot(x_fix, pushed_fixation, lw=4, c='g', ls='-')
    ax1.text(15, 0.03, 'fixed ancestor', fontsize=markings_size, fontweight='bold', color='g')
    ax1.text(-55, 0.085, 'C', fontsize=markings_size, fontweight='bold', color='k')

    ax2 = ax1.twinx()
    ax2.set_ylabel('population density', fontsize=label_size, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylim([0.0, 1.1])
    for tick in ax2.yaxis.get_major_ticks():
        tick.label2.set_fontsize(8) 
    ax2.plot(x_fix, pushed_profile, lw=4, c='k', ls='--')
    ax2.text(-5, 0.9, 'density', fontsize=markings_size, fontweight='bold', color='k')

    plt.tight_layout(pad=2.4, h_pad=0.5)
    plt.savefig('plots/Fig2_fixation.pdf')

if __name__=='__main__':
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 8}
    matplotlib.rc('font', **font)

    Fig1_growth(labels_flag=1, label_size=8)
    Fig2_fixation(label_size=9, markings_size=9)

    plt.show()
