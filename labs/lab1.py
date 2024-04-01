import numpy as np
import matplotlib.pyplot as plt

#! Глобальные переменные
eps = 0.000001


def f1(x):
    return x**3 + x - 1


def df1(x):
    return 3*x**2 + 1


def f2(x, n, alpha):
    return x**(n + 1) + x - alpha


def df2(x, n, alpha):
    return (n + 1)*x**n + 1


def dichotomy_method(left, right, f):
    x_path = []

    while right - left > eps:
        x_mid = (left + right) / 2
        if f(x_mid) == 0 or abs(f(x_mid)) < eps:
            return x_mid, x_path
        elif f(left)*f(x_mid) < 0:
            right = x_mid
        else:
            left = x_mid
        x_path.append(f(x_mid))
        
    return (left + left) / 2, x_path


def newton_method(x0, f, df):
    x_next = x0 - f(x0) / df(x0)
    x_path = [f(x_next)]

    while abs(x0 - x_next) > eps:
        x0, x_next = x_next, x0
        x_next = x0 - f(x0) / df(x0)
        x_path.append(f(x_next))

    return x_next, x_path


def main():
    x_opt, path_dichotomy = dichotomy_method(-1, 1, f1)
    print('Метод дихотомии = {}'.format(x_opt))
    x_opt, path_newton = newton_method(0, f1, df1)
    print('Метод Ньютона = {}'.format(x_opt))

    plt.plot(range(1, len(path_dichotomy) + 1), np.abs(path_dichotomy), '-o', label='Метод дихотомии')
    plt.plot(range(1, len(path_newton) + 1), np.abs(path_newton), '-x', label='Метод Ньютона')
    plt.xlabel('Число итераций')
    plt.ylabel('$|f(x)|$')
    plt.xticks(np.arange(0, len(path_dichotomy) + 1, 1))
    plt.grid()
    plt.legend(loc = 'best', fontsize = 10)
    plt.savefig('lab1_root.png', dpi = 200)

    alphaGrid = np.linspace(0, 10)
    colors = ['r', 'g', 'b']

    plt.clf()
    plt.xlabel('$\\alpha$')
    plt.ylabel('$x$')

    for i in range(1, 4):
        roots = [newton_method(alpha, lambda x: f2(x, i*2, alpha), \
                                      lambda x: df2(x, i*2, alpha))[0] \
                                      for alpha in alphaGrid]
        plt.plot(alphaGrid, roots, colors[i - 1], label = '$ N = ' + str(i*2) + '$')
    plt.xticks(np.arange(0, 11, 1))
    plt.grid()
    plt.legend(loc = 'best', fontsize = 10)
    plt.savefig('lab1_roots.png', dpi = 200)


if __name__ == '__main__':
    main()