import numpy as np
import matplotlib.pyplot as plt

eps = 0.001

def newton_method(x0, f, df):
    x_next = x0 - f(x0) / df(x0)
    x_path = [f(x_next)]

    while abs(x0 - x_next) > eps:
        x0, x_next = x_next, x0
        x_next = x0 - f(x0) / df(x0)
        x_path.append(f(x_next))

    return x_next, x_path

def f2(x, n, alpha):
    return x**(n + 1) + x - alpha


def df2(x, n, alpha):
    return (n + 1)*x**n + 1

def main():
    alphaGrid = np.linspace(0.1, 5)
    colors = ['r', 'g', 'b']

    plt.xlabel('$\\alpha$')
    plt.ylabel('$\\tau$')

    for i in range(1, 4):
        roots = [newton_method(alpha, lambda x: f2(x, i*2, alpha), \
                                      lambda x: df2(x, i*2, alpha))[0] \
                                      for alpha in alphaGrid]
        w_values = []
        for j, x0 in enumerate(roots):
            rootArg = (i*2*(1 - x0 / alphaGrid[j]))**2 - 1.
            if rootArg > 0:
                w_values.append(np.sqrt(rootArg))
            else:
                w_values.append(np.nan)

        t_values = []
        for j, w in enumerate(w_values):
            acosArg = -1/((i*2)*(1 - roots[j] / alphaGrid[j]))
            if np.abs(acosArg) < 1:
                t_values.append(np.arccos(acosArg) \
                                / w if w is not np.nan else np.inf)
            else:
                t_values.append(np.nan)
    
        plt.plot(alphaGrid, t_values, colors[i - 1] + '-o', \
                 label = '$N = ' + str(i*2) + '$')

    plt.grid()
    plt.legend(loc = 'best', fontsize = 10)
    plt.savefig('lab2_biffur.png', dpi = 200)

if __name__ == '__main__':
    main()
