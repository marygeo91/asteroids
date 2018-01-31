#!/usr/bin/env python

import numpy as np
import timeit as ti
import scipy.stats as st
import scipy.integrate as ig
import matplotlib.pyplot as pl
import matplotlib.ticker as tk


def euler_forward(previous=np.array([[1.0], [0.0]]), step=1e-3):
    '''
    Using the Euler Forward Method written in the matrix form for this specific
    system of differential equations, this function receives the last step - as
    well as a step size value - and returns the following step.
    '''

    assert step > 0.0, 'Step has to be higher than zero!'

    matrix = np.array([[1.0, step], [-step, 1.0]])

    return np.dot(matrix, previous)


def euler_backward(previous=np.array([[1.0], [0.0]]), step=1e-3):
    '''
    Using the Euler Backward Method written in the matrix form for this
    specific system of differential equations, this function receives the
    last step - as well as a step size value - and returns the following step.
    '''

    assert step > 0.0, 'Step has to be higher than zero!'

    matrix = (1.0/(1.0+step**2))*np.array([[1.0, step], [-step, 1.0]])

    return np.dot(matrix, previous)


def euler_symplectic(previous=np.array([[1.0], [0.0]]), step=1e-3):
    '''
    Using the Euler Symplectic Method written in the matrix form for this
    specific system of differential equations, this function receives the
    last step - as well as a step size value - and returns the following step.
    '''

    assert step > 0.0, 'Step has to be higher than zero!'

    matrix = np.array([[1.0-step**2, step], [-step, 1.0]])

    return np.dot(matrix, previous)


def func(previous=np.array([[1.0], [0.0]])):
    '''
    Returns the vectorial function f(x, v) associated to the specific problem
    (Harmonic Oscillator), given the previous known point.
    '''
    return np.array([[previous[1][0]], [-previous[0][0]]])


def midpoint_rule(previous=np.array([[1.0], [0.0]]), step=1e-3):
    '''
    Using the Midpoint Rule, this function receives the last step - as well as
    a step size value - and returns the following step.
    '''

    assert step > 0.0, 'Step has to be higher than zero!'

    k1 = step*func(previous)
    k2 = step*func(previous+0.5*k1)

    return previous+k2


def runge_kutta(previous=np.array([[1.0], [0.0]]), step=1e-3):
    '''
    Using the fourth order Runge-Kutta, this function receives the last step -
    as well as a step size value - and returns the following step.
    '''

    assert step > 0.0, 'Step has to be higher than zero!'

    k1 = step*func(previous)
    k2 = step*func(previous+0.5*k1)
    k3 = step*func(previous+0.5*k2)
    k4 = step*func(previous+k3)

    return previous+k1/6.0+k2/3.0+k3/3.0+k4/6.0


def f_ode(previous, t):
    '''
    Same as func, but this one is specifically for the odeint integrator.
    '''
    return [previous[1], -previous[0]]


def calc(solution=np.array([[1.0], [0.0]]), method=euler_forward,
         step=1e-3, num_step=5e4):
    '''
    This function receives the initial condition, a specific solving method,
    the step size and the total number of steps and uses all these to compute
    the solution of a certain system of differential equantions.
    '''

    assert step > 0.0, 'Step has to be higher than zero!'
    assert int(num_step) >= 1, 'Number of steps has to be higher than one!'

    for i in range(int(num_step)):
        solution = np.append(solution, method(solution[:, [i]], step), axis=1)

    return solution


def plot(name='Plots/test_plot.png', title='Plotting test', data=[[[0.0, 1.0],
         [1.0, 0.8]], [[0.0, 1.0], [0.6, 0.0]]], params=[['k', '-', 1.1],
         ['r', '--', 0.9]], axes=['Time (t/s)', 'Position (x/m)'], legend=0,
         labels=['Data 1', 'Data 2'], scale='linear'):
    '''
    This function receives a group of parameters (such as the figure name, the
    figure title, the markers/lines parameters, the axes labels, the legend
    position within the figure, the data legends and the axes scale) and a set
    of data and plots them all together in the same figure.
    '''

    assert len(data) > 0, 'Data cannot be an empty array!'
    assert len(labels) == len(data), 'Incorrect amount of labels!'
    assert len(params) == len(data), 'Incorrect amount of marker parameters!'

    # Sets the size of the generated figure.
    pl.figure(figsize=(8, 4))

    # Sets the rotation of the tick labels.
    pl.xticks(rotation=15)
    pl.yticks(rotation=15)

    # Sets the title parameters.
    pl.title(title, fontsize=18, weight='bold')

    # Sets the axes labels parameters.
    pl.xlabel(axes[0], labelpad=8, fontsize=14)
    pl.ylabel(axes[1], labelpad=8, fontsize=14)

    # Sets the background grid parameters.
    pl.grid(color='gray', linestyle='-.', linewidth=0.8, alpha=0.2)

    # Sets the axes ticks parameters.
    ca = pl.gca()
    ca.tick_params(which='minor', bottom=False, left=False)
    ca.tick_params(top=True, right=True, length=5, width=1.25,
                   labelsize=10, direction='in', pad=5)
    if scale == 'log':
        ca.xaxis.set_major_formatter(tk.ScalarFormatter())
        ca.yaxis.set_major_formatter(tk.ScalarFormatter())
    if scale == 'linear':
        ca.ticklabel_format(axis='y', fontstyle='italic', style='sci',
                            useMathText=True, scilimits=(0, 0))

    # Sets the axes thickness.
    sides = ['top', 'bottom', 'left', 'right']
    [ca.spines[s].set_linewidth(1.25) for s in sides]

    # Plots the different sets of data with the received parameters.
    for i in range(len(data)):
        ca.plot(data[i][0], data[i][1], params[i][0]+params[i][1],
                label=labels[i], lw=float(params[i][2]))

    # Sets the legend parameters.
    ca.legend(loc=legend, ncol=3, columnspacing=0.65, handletextpad=0.15,
              labelspacing=0.15, handlelength=1.3, borderpad=0.25,
              borderaxespad=0.75)

    # Sets the axes offset text (scientific notation) parameters.
    xo, yo = ca.get_xaxis().get_offset_text(), ca.get_yaxis().get_offset_text()
    xo.set_fontsize(8)
    yo.set_fontsize(8)
    xo.set_fontstyle('italic')
    yo.set_fontstyle('italic')

    # Sets the axes scale.
    if scale != 'linear':
        pl.xscale(scale)
        pl.yscale(scale)

    # Saves and closes the obtained figure.
    pl.tight_layout()
    pl.savefig(name, dpi=500, facecolor='ivory', edgecolor=None)
    pl.close()


def results(method=euler_forward, method_name='Euler Forward',
            method_tag='feul', time_ana=np.arange(0.0, 10.0, 0.1),
            step=1e-3, solu_ana=np.cos(np.arange(0.0, 10.0, 0.1)),
            num_step=5e4, h_list=np.logspace(-5, 0, num=5, base=2)):
    '''
    This function receives, as a function, a computional method to solve a
    specific set of coupled first order differential equations (as well as the
    respective method name and tag for the filenames), an array with times and
    another one with the known analytic solution of the system on those points
    in time, the step size and the total number of steps used in the analytic
    solution array and the list of step sizes that will be used with the chosen
    method. Not only it creates some plots showing the difference between the
    calculated quantities considering the different step sizes, as it also
    returns those quantities for a further analysis outside this function.
    '''

    assert step > 0.0, 'Step has to be higher than zero!'
    assert len(h_list) > 0, 'List of step sizes cannot be an empty array!'
    assert int(num_step) >= 1, 'Number of steps has to be higher than one!'
    assert len(time_ana) > 0, 'Analytic times cannot be an empty array!'
    assert len(solu_ana) > 0, 'Analytic solution cannot be an empty array!'
    assert len(time_ana) == len(solu_ana), 'Unmatching analytic arrays!'

    # Creates empty arrays that will contain the calculated quantities (such as
    # the computed times, the computed solution, the computed total energy of
    # the considered system, the absolute global error in the computed position
    # when comparing to the analytic solution, the maximum absolute global
    # error for each considered step size and the maximum system energy value
    # for each considered step size as well).
    time_com, solu_com, solu_ene = [], [], []
    posi_err, maxi_err, maxi_ene = [], [], []

    # Stores the list of colors used for plotting the data corresponding to the
    # different considered step sizes.
    colors = ['g', 'y', 'c', 'b', 'm', 'r']

    # Creates and initializes the arrays that will contain all the data labels,
    # the markers/lines parameters for plotting purposes and, for every step
    # size considered, the computed system solutions, the absolute global
    # errors and the total energy of the system - all as functions of the time.
    data_labels, data_params = ['Ana. Sol.'], [['k', '-', 1.1]]
    solutions, global_errors, energies = [[time_ana, solu_ana]], [], []

    for i in range(len(h_list)):
        # Calculates the amount of steps needed to get to a specific time.
        h = h_list[i]
        N = num_step*step/h

        # Calculates the time array, the computed system solution, the absolute
        # global errors in the computed solution for the mass position and its
        # corresponding maximum value, the initial energy of the system, the
        # energy of the system relatively to its initial value and, finally,
        # the maximum of the previously mentioned array. The if/else exception
        # deals with the odeint integration differently.
        time_com.append(np.arange(0.0, num_step*step, h))
        leng = len(time_com[i])
        if method != ig.odeint:    
            solu_com.append(calc(step=h, method=method, num_step=N)[:, :leng])
        else:
            temp = method(f_ode, [1.0, 0.0], time_com[i], hmin=h, hmax=h)
            solu_com.append(np.swapaxes(temp, 0, 1))
        iene = solu_com[i][0][0]**2 + solu_com[i][1][0]**2
        solu_ene.append(np.sum(solu_com[i]**2, axis=0)-iene)
        posi_err.append(np.cos(time_com[i])-solu_com[i][0])
        maxi_ene.append(max(np.absolute(solu_ene[i])))
        maxi_err.append(max(np.absolute(posi_err[i])))

        # Adds the new data labels and respective markers/lines parameters, and
        # stores the some of the previous calculations in the previously
        # created arrays for this specific purpose.
        data_labels.append('h = %.3f' % h)
        data_params.append([colors[i], '--', 0.9])
        solutions.append([time_com[i], solu_com[i][0]])
        global_errors.append([time_com[i], posi_err[i]])
        energies.append([time_com[i], solu_ene[i]])

    # For different step sizes and the considered method, generates three
    # plots: the first one with the computed solutions as a function of time,
    # the second one with the computed absolute global errors as a function of
    # time (relatively to the known analytic solution) and the third one with
    # the total energy of the system as a function of time (relatively to its
    # initial value).
    plot(name='Plots/' + method_tag + '_sol.png', data=solutions,
         title=method_name + ' - Solution', labels=data_labels,
         params=data_params, legend=3)

    plot(name='Plots/' + method_tag + '_err.png', data=global_errors,
         title=method_name + ' - Error', labels=data_labels[1:],
         axes=['Time (t/s)', 'Absolute global error (x/m)'],
         params=data_params[1:])

    plot(name='Plots/' + method_tag + '_ene.png', data=energies,
         title=method_name + ' - Energy', labels=data_labels[1:],
         axes=['Time (t/s)', 'Total energy variation (E/J)'],
         params=data_params[1:])

    return [solutions, global_errors, energies, maxi_ene, maxi_err]


if __name__ == '__main__':
    # Tests the plotting function.
    plot()

    # Generates the analytical solution.
    step_size, number_steps = 1e-3, 5e4
    analytic_times = np.arange(0.0, number_steps*step_size, step_size)
    analytic_solution = np.cos(analytic_times)

    # Generates the array of step sizes that will be used.
    steps_list = 0.512*np.logspace(-9, -4, num=6, base=2, endpoint=True)

    # Stores the functions, names and tags (for the filenames) for each method.
    methods = [euler_forward, euler_backward, euler_symplectic,
               midpoint_rule, runge_kutta, ig.odeint]
    names = ['Euler Forward', 'Euler Backward', 'Euler Symplectic',
             'Midpoint Rule', '$4^{th}$ Runge-Kutta', 'SciPy odeint']
    tags = ['eul_for', 'eul_bac', 'eul_sym', 'mid_rul', 'run_kut', 'ode_int']
    assert len(names) == len(methods), 'Arrays must have the same length!'
    assert len(tags) == len(methods), 'Arrays must have the same length!'

    # Stores the list of colors used for plotting the data corresponding to the
    # different considered methods.
    colors = ['g', 'y', 'c', 'b', 'm', 'r']

    # Creates an array which will contain the final results of each considered
    # method and also empty arrays for some useful linear regressions - these
    # will give an idea of the rate of growth of the quantities (total energy
    # of the system and absolute global error) plotted in the y axis as a
    # function of the step size. Then it executes all the calculations for each
    # method, storing the obtained results directly into the mentioned arrays;
    # arrays for the labels and marker/lines parameters are also created.
    final, energy, errors, ene_label, err_label, pars = [], [], [], [], [], []
    for i in range(len(methods)):
        # Exception that deals with the fact that the odeint function does not
        # integrate for the three bigger step sizes that were chosen.
        if methods[i] == ig.odeint:
            slist = steps_list[:3]
        else:
            slist = steps_list
        final.append(results(method=methods[i], method_name=names[i],
                             method_tag=tags[i], time_ana=analytic_times,
                             solu_ana=analytic_solution, h_list=slist))

        energy.append([slist, final[i][3]])
        ene_label.append(names[i])
        l = st.linregress(np.log10(slist), np.log10(final[i][3]))
        temp = (10**l[1])*(slist**l[0])
        energy.append([slist, temp])
        ene_label.append(r'$\Delta E=a\times h^{%.3f}$' % l[0])

        errors.append([slist, final[i][4]])
        err_label.append(names[i])
        l = st.linregress(np.log10(slist), np.log10(final[i][4]))
        temp = (10**l[1])*(slist**l[0])
        errors.append([slist, temp])
        err_label.append(r'$GE=a\times h^{%.3f}$' % l[0])

        pars.append([colors[i], 'x', 7.5])
        pars.append([colors[i], ':', 1.1])

    # Generates two plots: the maximum value of the total energy of the system
    # as a function of the step size and the maximum absolute global error as a
    # function of the step size (for the different considered methods).
    plot(name='Plots/energy_method.png', title='Total energy vs. Step size',
         scale='log', axes=['Step size (h/m)', 'Maximum total energy (E/J)'],
         data=energy, labels=ene_label, params=pars)

    plot(name='Plots/errors_method.png', title='Global error vs. Step size',
         scale='log', axes=['Step size (h/m)', 'Maximum global error (x/m)'],
         data=errors, labels=err_label, params=pars)
