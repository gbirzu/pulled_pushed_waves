import numpy as np
import scipy.integrate as integrate 
import time
import sys


def m_rate(arr):
    #computes the migration rate; takes in the density array and returns an array of migration rates
    #density dependent migration currently disabled
    #return m0 + m1*arr
    return m0

def migration_dd(arr):
    new_arr = m_rate(np.roll(arr, 1))*np.roll(arr, 1) - 2.*m_rate(arr)*arr + m_rate(np.roll(arr, -1))*np.roll(arr, -1)
    new_arr[0] = new_arr[1]
    new_arr[-1] = new_arr[-2]
    return new_arr

def tp1_prefactor(arr):
    new_arr = np.roll(arr, 1) + arr + np.roll(arr, -1)
    new_arr[0] = new_arr[1]
    new_arr[-1] = new_arr[-2]

    prefact = 1. + r0*dt*new_arr/3.
    return prefact

def update_tp1(arr):
    R = (dt/dx**2)
    new_arr = (R*migration(arr) + (1. + r0*dt)*arr)/tp1_prefactor(arr)
    return new_arr

def migration(arr):
    new_arr = m0*np.roll(arr, 1) - 2.*m0*arr + m0*np.roll(arr, -1)
    new_arr[0] = new_arr[1]
    new_arr[-1] = new_arr[-2]
    return new_arr

def growth_rate(arr, function, params):
    if function=='logistic':
        return r0*arr*(1. - arr)
    if function=='B-cooperative':
        B = params[0]
        return r0*arr*(1. - arr)*(1. + B*arr)
    if function=='B-power':
        [gamma, B] = params
        return r0*arr*(1. - arr)*(1. + B*(arr**gamma))
    if function=='predation':
        [d, nstar] = params
        return (r0*(1. - arr) - d/(1. + arr/nstar))*arr
    if function=='yeast':
        [d, nstar, b] = params
        return (r0 + b*arr/(arr + nstar) - d*arr)*arr

def update_naive(arr, growth_type, params):
    R = dt/dx**2
    new_arr = arr + R*migration(arr) + dt*growth_rate(arr, growth_type, params)
    return new_arr

def total_time(Lmax, r_eff, m_eff):
    #estimates the time to reach Lmax, asuming Fisher velocity
    vF = 2.*np.sqrt(r_eff*m_eff)
    return 10*Lmax/vF

def estimate_l(time, r_eff, m_eff):
    vF = 2.*np.sqrt(r_eff*m_eff)
    return 2*vF*time

def velocity_B(B):
    if B >= 2.:
        return np.sqrt(r0*m0*B/2.)*(1. + 2./B)
    else:
        return 2.*np.sqrt(r0*m0)

def initialize_arrays(x_init):
    l = int(estimate_l(total_t, r0, m0))
    x_arr = np.arange(0, l, dx)
    nx = len(x_arr) #number of points on x axis
    u_arr = np.zeros(nx)
    u_arr[np.where(x_arr < x_init)] = 1.
    return x_arr, u_arr

def run(params, cutoff=0.):
    #params gives type of growth followed by other parameters
    growth_type = params[0]
    t_arr = []
    pop_arr = []
    x_arr, u_arr = initialize_arrays(1.)

    step = 0
    while step < time_steps:
        u_arr = update_naive(u_arr, growth_type, params[1:])
        u_arr[np.where(u_arr < cutoff)] = 0.
        if step%100==0:
            pop_arr.append(integrate.simps(u_arr, x_arr))
        if step==int(time_steps/10):
            t0 = step*dt
            p0 = integrate.simps(u_arr, x_arr)
        step += 1
        t_arr.append(step*dt)
    t1 = step*dt
    p1 = integrate.simps(u_arr, x_arr)
    
    ratio_v = (p1 - p0)/(t1 - t0)
    print 'velocity = ', ratio_v
    prof_output = np.array([x_arr, u_arr])
    pop_output = np.array([t_arr, pop_arr])

    vel_stem = growth_type+'_r0-'+str(r0)
    fvel = open('velocity_'+vel_stem+'.txt', 'a')
    if growth_type!='logistic':
        fvel.write(str(params[-1])+','+str(ratio_v)+','+str(max(u_arr))+'\n')
    else:
        fvel.write(str(ratio_v)+','+str(max(u_arr))+'\n')
    fvel.close()

    fparams = open('params_'+vel_stem+'.txt', 'a')
    for index in range(1, len(params) - 1):
        fparams.write(str(params[index])+',')
    fparams.write(str(params[-1])+'\n')
    fparams.close()

    name_stem = growth_type+'_r0-'+str(r0)+'_m0-'+str(m0)+'_'+str(params[-1])
    np.save('profile_'+name_stem, prof_output)
    np.save('population_'+name_stem, pop_arr)

if __name__ == '__main__':
    m0 = 0.1
    r0 = 0.1
    dx = min(0.5*np.sqrt(2/r0), 0.05)
    dt = 0.1*dx**2/m0 # assigned for constant m_rate; should be changed for density-depnendent migration 
    total_t = 1E2 #simulation time; default value
    time_steps = int(total_t/dt)
    model = 'logistic' #default model

    #Grab arguments
    i = 1
    while i < len(sys.argv):
        flag = sys.argv[i]
        if flag=='-m':
            i += 1
            model = sys.argv[i]
        elif flag=='-d':
            i += 1
            d = float(sys.argv[i])
        elif flag=='-b':
            i += 1
            b = float(sys.argv[i])
        elif flag=='-n':
            i += 1
            nstar = float(sys.argv[i])
        elif flag=='-g':
            i += 1
            gamma = float(sys.argv[i])
        elif flag=='-r':
            i += 1
            r0 = float(sys.argv[i])

        i += 1

    if model=='logistic':
        t_init = time.clock()
        run(params=[model], cutoff=1E-10)
        t_final = time.clock()
        print 'Time spent: {0:.2f}s '.format(t_final - t_init)
    
    elif model=='B-cooperative':
        t_init = time.clock()
        run(params=[model, b], cutoff=1E-10)
        t_final = time.clock()
        print 'Time spent: {0:.2f}s '.format(t_final - t_init)

    elif model=='B-power':
        t_init = time.clock()
        run(params=[model, gamma, b], cutoff=1E-10)
        t_final = time.clock()
        print 'Time spent: {0:.2f}s '.format(t_final - t_init)

    elif model=='predation':
        t_init = time.clock()
        run(params=[model, d, nstar], cutoff=1E-10)
        t_final = time.clock()
        print 'Time spent: {0:.2f}s '.format(t_final - t_init)

    elif model=='yeast':
        t_init = time.clock()
        run(params=[model, d, nstar, b], cutoff=1E-10)
        t_final = time.clock()
        print 'Time spent: {0:.2f}s '.format(t_final - t_init)

