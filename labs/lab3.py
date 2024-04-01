import numpy as np
import matplotlib.pyplot as plt


#! Глобальные переменные
eps = 0.0001
a = 0.2
b = 0.2
c = 5.7
T = 100
N = 1000
steps = [10e-2, 10e-3, 10e-4]
colors = ['r', 'g', 'b']


def equation1(x, t):
    return -x


def equation2(x, t):
    return x


def equation3(x, t):
    return np.array([x[1], -x[0]])


def equation4(x, t):
    return np.array([-x[1] - x[2], x[0] + a*x[1], b + x[2]*(x[0] - c)])


def euler(func, start_t, end_t, init, h):
    num_steps = int((end_t - start_t) / h)
    x_path = [0]*(num_steps + 1)
    t_path = [0]*(num_steps + 1)
    x_path[0] = init
    t_path[0] = start_t

    for i in range(len(t_path) - 1):
        x_path[i + 1] = x_path[i] + h*func(x_path[i], t_path[i])
        t_path[i + 1] = t_path[i] + h
    t_path[-1] = end_t
    x_path[-1] = x_path[-2] + h*func(x_path[-2], end_t)
    return np.array(t_path), np.array(x_path)


def RK4(func, start_t, end_t, init, h):
    num_steps = int((end_t - start_t) / h)
    x_path = [0.]*(num_steps + 1)
    t_path = [0.]*(num_steps + 1)
    x_path[0] = init
    t_path[0] = start_t

    for i in range(len(t_path) - 1):
        k1 = func(x_path[i], t_path[i])
        k2 = func(x_path[i] + 0.5*h*k1, t_path[i] + 0.5*h)
        k3 = func(x_path[i] + 0.5*h*k2, t_path[i] + 0.5*h)
        k4 = func(x_path[i] + h*k3, t_path[i] + h)
        x_path[i + 1] = x_path[i] + h*(k1 + 2*k2 + 2*k3 + k4) / 6
        t_path[i + 1] = t_path[i] + h

    return np.array(t_path), np.array(x_path)


def error_plot(t, err, name):
    plt.yscale('log')
    for i, step in enumerate(steps):
        plt.plot(t[i], err[i], colors[i], label = 'h = ' + str(step))
    plt.xlabel('$t$')
    plt.ylabel('$log|\\xi(t)|$')
    plt.grid()
    plt.legend(loc = 'best', fontsize = 10)
    plt.savefig(name, dpi = 200)
    plt.cla()
    #plt.show()



def main():
    t_start = 0
    t_stop = 100

    #! Система 1
    err_values = []
    t_values = []
    x0 = 2
    for h in steps:
        t, x = RK4(equation1, t_start, t_stop, x0, h)
        err_values.append(np.abs(x - x0*np.exp(-t)))
        t_values.append(t)
    error_plot(t_values, err_values, 'lab3_equation1_rk.png')


    #! Система 2
    err_values = []
    t_values = []
    x0 = 2
    for h in steps:
        t, x = RK4(equation2, t_start, t_stop, x0, h)
        err_values.append(np.abs(x - x0*np.exp(t)))
        t_values.append(t)
    error_plot(t_values, err_values, 'lab3_equation2_rk.png')


    #! Система 3
    err_values = []
    t_values = []
    x0 = np.array([2, 3])
    for h in steps:
        t, x = RK4(equation3, t_start, t_stop, x0, h)
        err_values.append(np.abs(x[:, 0] - (x0[0]*np.cos(t) + x0[1]*np.sin(t))))
        t_values.append(t)
    error_plot(t_values, err_values, 'lab3_equation3_rk.png')


    #! Система 4
    err_values = []
    t_values = []
    x0 = np.array([10, 10, 10])
    for h in steps:
        t1, x1 = RK4(equation4, t_start, t_stop, x0, h)
        t2, x2 = RK4(equation4, t_start, t_stop, x0, h/2)
        err_values.append(np.sum(np.abs(x1 - x2[::2]), axis = 1))
        t_values.append(t1)
    error_plot(t_values, err_values, 'lab3_equation4_rk.png')

if __name__ == '__main__':
    main()