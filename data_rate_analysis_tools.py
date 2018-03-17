import numpy as np 
import matplotlib.pyplot as plt
import glob

from string import *

def cm2inch(size):
    return size/2.54

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
    gf = 1.0
    m = 1.0
    N = 1.0
    B = 0.0
    for s in aux:
        if s[0] == 'gf':
            gf = float(s[1])
        elif s[0] == 'migr':
            m = float(s[1])
        elif s[0] == 'N':
            N = int(s[1])
        elif s[0] == 'B':
            B = float(s[1])
    return gf, m, N, B

def linear_reg(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    X = np.vander(x, 2)
    coeffs, res, rank, sing_vals = np.linalg.lstsq(X, y)
    mx = x.sum()/len(x)
    sx = float(((x - mx)**2).sum())
    if len(x) > 2:
        r2 = 1. - res/(y.size*np.var(y))
    else:
        r2 = 0
    return coeffs, r2

def neff_BSC(N, g):
    return np.log(N)**3/(2*np.pi**2*g)

class expansion():
    '''Class containing all relevant information about an expansion experiment: g, m, f*, v, profile shape and exponen, N, kappa'''
    def __init__(self, location, term, gf_init, m_init, B_init, N_init):
        self.location = location
        self.g = gf_init
        self.m = m_init
        self.B = B_init
        self.N = N_init
        self.kappa = 'def'
        self.v = 'def'
        if term != '':
            self.termination = term
        else:
            self.termination = termination

    def create_name(self, prefix, extension):
        name = prefix + '_N' + str(self.N) + '_gf' + str(self.g) + '_migr' + str(self.m) + '_B'  + str(self.B) + '_' + extension 
        return name

    def velocity_theory(self):
        if self.B >= 2.0:
            vel = np.sqrt(self.g*self.B*(self.m/2.)/2.)*(1 + 2./self.B)
        else:
            vel = 2.*np.sqrt(self.g*(self.m/2.))
        return vel

    def profile_theory(self, x_arr):
        profile = self.N/(1. + np.exp(np.sqrt(self.g/self.m)*x_arr))
        return profile

    def find_fit_endpoint(self, x_array, y_array, x_fin, epsilon, forward_flag=0):
        '''If forward_flag = 1 the search begins at the start of the array
        forward_flag is optional; default value is 0'''
        r_sq_previous = 0.

        if forward_flag == 0:
            x_init = x_fin - min(100, x_fin/2) #necessary if nonzero array is small
        else:
            x_init = 0
        x_best = x_init
        
        while 1 - r_sq_previous > epsilon and x_init > 1:
            x_fit = x_array[x_init:x_fin]
            y_fit = y_array[x_init:x_fin]
            coeffs, r_sq = linear_reg(x_fit, y_fit)#Do fit

            if r_sq > r_sq_previous:
                x_best = x_init

            r_sq_previous = r_sq
            if forward_flag == 0:
                x_init -= 1
            else:
                x_init += 1

        return x_best, x_array[x_best:x_fin], y_array[x_best:x_fin]

    def get_data_types(self):
        h_file = 0
        v_file = 0
        p_file = 0

        name_root = 'N' + str(self.N) + '_gf' + str(self.g) + '_migr' + str(self.m) + '_B' + str(self.B)
        if len(glob.glob(self.location+'hetero_'+name_root+'_*.npy')) == 1:
            h_file = 1
        if len(glob.glob(self.location+'velocity_'+name_root+'_*.npy')) == 1:
            v_file = 1
        if len(glob.glob(self.location+'profile_'+name_root+'_*.npy')) == 1:
            p_file = 1

        self.h_file = h_file
        self.v_file = v_file
        self.p_file = p_file


    def read_het(self, plot_flag, save_flag):
        '''Computes kappa from heterozygosity data and adds information to class instance'''

        #create name for file to read
        file_name = self.create_name('hetero', self.termination)
        
        #read data from files
        het_data_arr = np.load(self.location + file_name)
        self.het_data = het_data_arr.T
        time_arr = het_data_arr[0]
        het_arr = het_data_arr[1]
        survival_arr = het_data_arr[2]

        #calculate fitiing range using sliding window
        #use as final time the point where 5% of simulations have nonzero H
        time_final = 0
        while (survival_arr[time_final] > 0.05 and time_final < len(het_arr) - 1): #final point is last point with 5% of simulations with H != 0
            time_final += 1 #find final fitting point

        epsilon = 1E-6 #set threshold for stopping

        time_initial, time_fit, loghet_fit = self.find_fit_endpoint(time_arr, np.log(het_arr), time_final, epsilon)
        het_fit = np.exp(loghet_fit)

        coeffs, r_sq = linear_reg(time_fit, np.log(het_fit))#Do fit

        self.kappa = abs(coeffs[0])

        if plot_flag != 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('N = ' + str(self.N) + ', f* = ' + str(self.B))
            ax.set_xlabel('Generations')
            ax.set_ylabel('Log(H)')

            ax.set_yscale('log')
            ax.plot(time_arr, het_arr, c='0.2', lw=2, alpha=1.0, label='Simulations')

            fit = np.poly1d(coeffs)
            est = fit(time_fit)
            ax.plot(time_fit, np.exp(est), c='g', lw=4, alpha=0.8, label='Fit')
            ax.legend()

            if save_flag != 0:
                plt.savefig(self.location + 'plots/heteroplot_N' + str(self.N) + '_f' + str(self.B) + '.pdf')

    def read_velocity(self, plot_flag):
        #create name for file to read
        file_name = self.create_name('velocity', self.termination)
        
        #read data from files
        vel_data_arr = np.load(self.location + file_name)
        time_arr = vel_data_arr.T[0]
        population_arr = vel_data_arr.T[1]

        #calculate fitiing range using sliding window
        #use final time as end point of fit
        i_fin = len(population_arr) - 1

        epsilon = 1E-6 #set threshold for stopping search

        i_init, time_fit, population_fit = self.find_fit_endpoint(time_arr, population_arr, i_fin, epsilon, forward_flag=1)

        coeffs, r_sq = linear_reg(time_fit, population_fit)#Do fit
        self.v = coeffs[0]

        if plot_flag != 0:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title('N = ' + str(self.N) + ', f* = ' + str(self.B))
            ax.set_xlabel('Generations')
            ax.set_ylabel('Population')

            ax.plot(time_arr, population_arr, c='0.2', lw=2, alpha=1.0, label='Simulations')

            fit = np.poly1d(coeffs)
            est = fit(time_fit)
            ax.plot(time_fit, est, c='g', lw=4, alpha=0.8, label='Fit')
            ax.legend()


    def read_profile(self, plot_flag):
        #create name for file to read
        file_name = self.create_name('profile', self.termination)
        
        #read data from files 
        prof_data_arr = np.load(self.location + file_name)
        self.profile = prof_data_arr
        x_arr = prof_data_arr.T[0]
        s1_arr = prof_data_arr.T[1]
        s2_arr = prof_data_arr.T[2]

        if plot_flag != 0:
            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.set_title('N = ' + str(self.N) + ', f* = ' + str(self.B))
            ax.set_xlabel('x')
            ax.set_ylabel('c(x)')
            ax.plot(x_arr, s1_arr + s2_arr, lw=2, c='0.2', alpha=0.5, label='simulations')
            ax.plot(x_arr, self.profile_theory(x_arr - 150*np.ones(len(x_arr))), lw=4, alpha=0.8, label='theory')
            ax.legend()

            ax2 = fig.add_subplot(122)
            ax2.set_yscale('log')
            ax2.plot(x_arr, s1_arr + s2_arr, lw=2, c='0.2', alpha=0.5, label='simulations')
            ax2.plot(x_arr, self.profile_theory(x_arr - 150*np.ones(len(x_arr))), lw=4, alpha=0.8, label='theory')

    def collect_all_data(self):
        self.get_data_types()
        if self.h_file == 1:
            self.read_het(plot_flag=0, save_flag=0)
        if self.v_file == 1:
            self.read_velocity(plot_flag=0)
        if self.p_file == 1:
            self.read_profile(plot_flag=0)               


class collection():
    '''Contains a collection of expansion classes and search plot functions based on various criteria'''
    def __init__(self, location, term='.npy'):
        self.g_list = []
        self.m_list = []
        self.B_list = []
        self.N_list = []
        self.experiment_list = []
        self.location = location #give data directory
        self.termination = term

    def get_variables(self, name):
        file_name = name.split('/')[-1].split('.')#Get file name

        #remove termination if one exists
        if 'txt' in file_name:
            file_name.remove('txt') #remove ending
        elif 'npy' in file_name:
            file_name.remove('npy')
        file_name = '.'.join(file_name) #reform name

        aux = [mystrip(s) for s in file_name.split('_')]
        #default values if none found
        g = 'def'
        m = 'def' 
        N = 'def' 
        B = 'def' 
        for s in aux:
            if s[0] == 'gf':
                g = float(s[1])
            elif s[0] == 'migr':
                m = float(s[1])
            elif s[0] == 'N':
                N = int(s[1])
            elif s[0] == 'B':
                B = float(s[1])

        if g == 'def' or m == 'def' or N == 'def' or B == 'def':
            error_flag = 1
        else:
            error_flag = 0
        return error_flag, g, m, N, B

    def make_collection(self):
        file_list = glob.glob(self.location + '*.npy')

        g_list = []
        m_list = []
        B_list = []
        N_list = []

        for file_name in file_list:
            error, g, m, N, B = self.get_variables(file_name)
            if error == 0:
                #add parameters to list
                g_list.append(g)
                m_list.append(m)
                B_list.append(B)
                N_list.append(N)

        self.g_list = sorted(set(g_list))
        self.m_list = sorted(set(m_list))
        self.B_list = sorted(set(B_list))
        self.N_list = sorted(set(N_list))

        for g in self.g_list:
            for m in self.m_list:
                for B in self.B_list:
                    for N in self.N_list:
                        #add expansion object to collection
                        exp = expansion(self.location, self.termination, g, m, B, N)
                        exp.collect_all_data()
                        self.experiment_list.append(exp)

    def filter_one_param(self, original_list, type_flag, type_value):
        '''returns all expansion instances with type_value at parameter type_flag'''
        new_list = []
        for elem in original_list:
            if type_flag == 'g':
                if elem.g == type_value:
                    new_list.append(elem)
            if type_flag == 'm':
                if elem.m == type_value:
                    new_list.append(elem)
            if type_flag == 'B':
                if elem.B == type_value:
                    new_list.append(elem)
            if type_flag == 'N':
                if elem.N == type_value:
                    new_list.append(elem)

        return new_list

    def filter_data(self, original_list, sort_parameters):
        '''returns all expansion instances with type_value at parameter type_flag'''
        number_of_parameters = len(sort_parameters)
        new_list = original_list
        
        for param in sort_parameters:
            p_label = param[0]
            p_value = param[1]
            new_list = self.filter_one_param(new_list, p_label, p_value)
        return new_list


    def write_data(self, file_name):
        out_file = open(self.location + file_name + '.txt', 'w')

        out_file.write(self.location+'\n')
        for elem in self.experiment_list:
            out_string = str(elem.g)+','+str(elem.m)+','+str(elem.B)+','+str(elem.N)+','+str(elem.kappa)+','+str(elem.v)
            out_file.write(out_string)
            out_file.write('\n')

            if elem.h_file == 1:
                out_file.write('heterozygosity')
                out_file.write('\n')
                for het_elem in elem.het_data:
                    out_string = str(het_elem[0])+','+str(het_elem[1])+','+str(het_elem[2])
                    out_file.write(out_string)
                    out_file.write('\n')
                out_file.write('EOD')
                out_file.write('\n')
            if elem.p_file == 1:
                out_file.write('profile')
                out_file.write('\n')
                for prof_elem in elem.profile:
                    out_string = str(prof_elem[0])+','+str(prof_elem[1])+','+str(prof_elem[2])
                    out_file.write(out_string)
                    out_file.write('\n')
                out_file.write('EOD')
                out_file.write('\n')
            out_file.write('\n')

        out_file.close()
        print 'Finished writing data file!'

    def read_data(self, file_name):
        self.experiment_list = []
        data_file = open(self.location + file_name + '.txt', 'r')
        self.location = data_file.readline()

        g_list = []
        m_list = []
        B_list = []
        N_list = []

        line = data_file.readline()
        while line != '':
            #start reading data if present
            data = line.split('\n')[0]
            data = data.split(',')
            if data[4] != 'def' and data[5] != 'def':
                g, m, B, N, kappa, v = atof(data[0]), atof(data[1]), atof(data[2]), atoi(data[3]), atof(data[4]), atof(data[5])
            elif data[4] != 'def':
                g, m, B, N, kappa = atof(data[0]), atof(data[1]), atof(data[2]), atoi(data[3]), atof(data[4])
                v = data[5]
            else:
                g, m, B, N = atof(data[0]), atof(data[1]), atof(data[2]), atoi(data[3])
                kappa = data[4]
                v = data[5]

            g_list.append(g)
            m_list.append(m)
            B_list.append(B)
            N_list.append(N)

            exp = expansion(self.location, self.termination, g, m, B, N)
            exp.kappa = kappa
            exp.v = v

            line = data_file.readline()

            while line != '\n': #if other data is available process
                if line == 'heterozygosity\n':
                    het_data = []
                    #start extracting heteroyzigosity data
                    line = data_file.readline()
                    while line != 'EOD\n':
                        line_data = line.split(',')
                        het_data.append([atof(line_data[0]), atof(line_data[1]), atof(line_data[2])]) #data in format time, heterozygosity, survival probability
                        line = data_file.readline()
                    het_data = np.array(het_data)
                    exp.het_data = het_data
                elif line == 'profile\n':
                    prof_data = []
                    #start extracting profile data
                    line = data_file.readline()
                    while line != 'EOD\n':
                        line_data = line.split(',')
                        prof_data.append([atof(line_data[0]), atof(line_data[1]), atof(line_data[2])])#data in format x, fraction species 1, fraction species 2
                        line = data_file.readline()
                    prof_data = np.array(prof_data)
                    exp.profile = prof_data

                line = data_file.readline()
                if line == 'EOD\n':#test if end of file
                    line =''
                    break
            #if exp.kappa != 'def' and exp.v != 'def':
            if exp.kappa != 'def':
                self.experiment_list.append(exp)

            print 'Finished! g='+str(g)+',m='+str(m)+',f='+str(B)+',N='+str(N)
            #go to next line
            line = data_file.readline()

        self.g_list = sorted(set(g_list))
        self.m_list = sorted(set(m_list))
        self.B_list = sorted(set(B_list))
        self.N_list = sorted(set(N_list))

        data_file.close()

class n_Allee_effect:
    def __init__(self, n, gr, Diff, rho):
        self.n = n
        self.g = gr
        self.D = Diff
        self.r = rho
    def velocity(self):
        return np.sqrt(self.g*self.D*(self.n))*(1./(self.n + 1) - self.r**self.n)
    def profile(self, x):
        return (1 + np.exp(self.n*np.sqrt(self.g/(self.D*(self.n + 1)))*x))**(-1./self.n)
    def v_F(self):
        vf = 0
        if self.r < 0.:
            vf = 2.*np.sqrt(self.g*self.D*abs(self.r))
        return vf

def v_F(r, Diff, B):
    g = r*B
    if B != 0:
        fstar = 1./B
    else:
        fstar = 1.
    if fstar <= 0: 
        vf = 2*np.sqrt(g*Diff*abs(fstar))
    else:
        vf = 0.
    return vf

def v_Fquartic(g, Diff, fstar):
    if fstar <= 0: 
        vf = 2*np.sqrt(g*Diff*abs(fstar**3))
    else:
        vf = 0.
    return vf

def v_F_cutoff(r, Diff, B, N):
    g = r*B
    if B != 0.:
        fstar = 1./B
    else:
        fstar = 1.
    if fstar < 0.5: 
        vf = 2*np.sqrt(g*Diff*abs(fstar))
    elif fstar >= 0.5:
        vf = 2*np.sqrt(g*Diff*abs(fstar)) - (np.pi**2*Diff*np.sqrt(g*abs(fstar)/Diff))/(np.log(N)**2)
    return vf

def v_Fquartic_cutoff(g, Diff, fstar, N):
    if fstar <= 0 and fstar > -0.5: 
        vf = 2*np.sqrt(g*Diff*abs(fstar**3))
    elif fstar <= -0.5:
        vf = 2*np.sqrt(g*Diff*abs(fstar**3)) - (np.pi**2*Diff*np.sqrt(g*abs(fstar**3)/Diff))/(np.log(N)**2)
    else:
        vf = 0.
    return vf

