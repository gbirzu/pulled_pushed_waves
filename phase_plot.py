import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob

from sklearn import linear_model

def cm2inch(x):
    return x/2.54

def mystrip(s): #Separate letters from words in string
    head = s.strip('-.0123456789')
    tail = s[len(head):]
    return head, tail  

def get_variables(name):
    het_name = name.split('/')[-1].split('.')#Get file name
    if 'txt' in het_name:
        het_name.remove('txt') #remove ending
    het_name = '.'.join(het_name) #reform name
    aux = [mystrip(s) for s in het_name.split('_')]
    #default values if none found
    r = 1.0
    m = 1.0
    b = 1.0
    d = 1.0
    n = 0.0
    e = 1.0
    for s in aux:
        if s[0] == 'r':
            gf = float(s[1])
        elif s[0] == 'm':
            m = float(s[1])
        elif s[0] == 'death':
            d = float(s[1])
        elif s[0] == 'nstar':
            n = float(s[1])
        elif s[0] == 'B' or s[0] == 'b':
            b = float(s[1])
        elif s[0] == 'exponent':
            e = float(s[1])

    return m, r, e, d, n, b

def get_velocity_data(model):
    file_list = glob.glob('data/distance/distance_'+model+'_*.txt')

    velocity_arr = []
    for name in file_list:

        m, r, e, d, n, b = get_variables(name)

        data_arr = np.loadtxt(name, delimiter=',')
        time_arr = data_arr.T[0][500:]
        distance_arr = 0.05*data_arr.T[1][500:]
        X = time_arr.reshape(len(time_arr), 1) 
        regr = linear_model.LinearRegression()
        regr.fit(X, distance_arr)

        if model == 'B-cooperative':
            velocity_arr.append([e, b, regr.coef_[0]])
        elif model == 'predation':
            velocity_arr.append([n, regr.coef_[0]])
        elif model == 'yeast':
            velocity_arr.append([b, regr.coef_[0]])
    return np.array(velocity_arr)
  

def plot_data(file_name, title, xlabel):
    #arr = np.loadtxt(file_name, delimiter=',')
    arr = file_name
    arr = arr[arr[:, 1].argsort()]
    v_arr = arr.T[1]/min(arr.T[1])
    print v_arr
    #vF = min(arr[:, 1])
    semipushed_ratio = (3./(2.*np.sqrt(2.)))
    if title != 'predation model':
        vF = 2.*np.sqrt(0.1)
    else:
        vF = 2.*np.sqrt(0.1*0.5)
    delta = [arr[i+1,1] - arr[i,1] for i in range(len(arr) - 1)]
    print title
    print arr.T[0][np.where(arr.T[1] > vF)[0][0]]
    print arr.T[0][np.where(arr.T[1] > semipushed_ratio*vF)[0][0]]
    print arr.T[0][np.where(arr.T[1] > semipushed_ratio*vF)[0][0]] - arr.T[0][np.where(arr.T[1] > vF)[0][0]]
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('$v/v_F$', fontsize=14)

    ax.set_xlim(0.9*min(arr.T[0]), 1.1*max(arr.T[0]))
    #ax.scatter(arr.T[0], arr.T[1], color='blue')
    #ax.plot(arr.T[0], vF*np.ones(len(arr)), color='k', ls='--')
    #ax.plot(arr.T[0], (3.*vF/(2.*np.sqrt(2.)))*np.ones(len(arr)), color='red', ls='--')
    #if title != 'predation model':
    ax.scatter(arr.T[0], v_arr, color='blue')
    ax.plot(arr.T[0], np.ones(len(arr)), color='k', ls='--')
    ax.plot(arr.T[0], semipushed_ratio*np.ones(len(arr)), color='red', ls='--')
    #else:
    #    ax.scatter(1./arr.T[0], v_arr, color='blue')
    #    ax.plot(arr.T[0], np.ones(len(arr)), color='k', ls='--')
    #    ax.plot(arr.T[0], (3./(2.*np.sqrt(2.)))*np.ones(len(arr)), color='red', ls='--')
    #plt.savefig('plots/'+title+'_phase_plot.pdf')

def phase_plot_si():
    font = {'family' : 'sans-serif', 'serif' : 'Helvetica Neue', 'weight' : 'bold', 'size' : 12}
    matplotlib.rc('font', **font)

    semipushed_ratio = (3./(2.*np.sqrt(2.)))

    fig = plt.figure(figsize=(cm2inch(17.8), cm2inch(9)))

    yeast_arr = np.load('data/velocity_yeast.npy')
    vF = min(yeast_arr.T[2])
    x_min = 5E-4
    x_max = 2.0
    y_min = 0.54
    y_max = 0.6

    ax = fig.add_subplot(122)
    ax.set_xscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_title('yeast model', fontsize=14, fontweight='bold')
    ax.set_xlabel('sucrose concentration, $\mathbf{s}$', fontsize=14, fontweight='bold')
    ax.set_ylabel('expansion velocity, $\mathbf{v}$', fontsize=14, fontweight='bold')
    ax.text(1.5E-4, 0.6025, 'B', fontsize=14, fontweight='bold')
    ax.text(0.05, 0.545, 'pulled', fontsize=12, fontweight='bold', rotation=90, color='k', transform=ax.transAxes)
    ax.text(0.0045, 0.59, 'semi-pushed', fontsize=12, fontweight='bold')
    ax.text(0.86, 0.67, 'fully-pushed', fontsize=12, fontweight='bold', rotation=90, color='k', transform=ax.transAxes)

    #ax.set_xticks([0.001, 0.008, 0.05, 0.5])
    ax.set_xticks([0.001, 0.01, 0.1, 1.0])
    #ax.set_xticklabels(['0.001', '0.008', '0.05', '0.5'])
    ax.set_yticks([0.54, 0.56, 0.58])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.fill_between([x_min, 0.004], y_min, y_max, facecolor='lightsalmon', alpha=0.5)
    ax.fill_between([0.004, 0.4], y_min, y_max, facecolor='lightgreen', alpha=0.5)
    ax.fill_between([0.4, x_max], y_min, y_max, facecolor='lightskyblue', alpha=0.5)

    ax.scatter(yeast_arr.T[1], yeast_arr.T[2], facecolor='k', edgecolor='none', s=30)
    ax.plot([x_min, x_max], vF*np.ones(2), color='r', ls='--')
    ax.plot([x_min, x_max], vF*semipushed_ratio*np.ones(2), color='r', ls='--', lw=2)


    predation_arr = get_velocity_data('predation')
    predation_arr = predation_arr[predation_arr[:, 1].argsort()]
    #v_arr = predation_arr.T[1]/min(predation_arr.T[1])
    #vF = min(predation_arr[:, 1])
    vF = min(predation_arr.T[1])
    delta = [predation_arr[i+1,1] - predation_arr[i,1] for i in range(len(predation_arr) - 1)]
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


    plt.tight_layout()
    plt.savefig('plots/phase_space_si.pdf')


if __name__=='__main__':
    #plot_data(get_velocity_data('predation'), 'predation model', '$n^*$')
    #plot_data(get_velocity_data('yeast'), 'simplyfied yeast model', 'B')
    phase_plot_si()
    '''
    file_name = 'data/velocity_B-cooperative_m0.1_r1_exponent1.txt'
    plot_data(file_name, 'cooperative model', 'B')
    file_name = 'data/velocity_B-cooperative_m0.1_r1_exponent2.txt'
    plot_data(file_name, 'cooperative model, exp=2', 'B')
    file_name = 'data/velocity_predation_m0.1_r1_death0.5.txt'
    plot_data(file_name, 'predation model', '$n^*$')
    file_name = 'data/velocity_yeast_m0.1_r1_death0.1_nstar100.txt'
    plot_data(file_name, 'simplyfied yeast model', 'B')
    '''

    plt.show()
