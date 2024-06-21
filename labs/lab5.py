import numpy as np
import matplotlib.pyplot as plt

def delete_similar(array, value):
    sort_arr = []
    for x in array:
        if abs(x - value) > 0.001:
            sort_arr.append(x)
    return sort_arr

def comput_lapun(x_0, r, n):
    x = x_0
    lamd_lapun = 0
    for i in range(n):
        x = x*r*(1 - x)
        log = np.log(abs(r - 2*r*x))
        lamd_lapun += log
    return x, lamd_lapun / n

def sort_array(array):
    sort_arr = array
    for i in range(int(len(array) / 2 + 1)):
        sort_arr = delete_similar(sort_arr, array[i])
        sort_arr.append(array[i])
    return sort_arr

def main():
    #np.random.seed(100)
    n_iter = 1600
    impuls = 100
    r_list = np.linspace(1e-3, 4, 1000)
    x_start = np.random.rand(impuls)
    l_const = np.zeros((len(r_list), 1))
    x_finish = np.zeros((len(r_list), impuls))

    for j in range(impuls):
        for i, r in enumerate(r_list):
            x_end, lam = comput_lapun(x_start[j], r, n_iter)
            x_finish[i][j] = x_end
            l_const[i] += lam

    l_const /= impuls
    r_chaos = r_list[np.argmax(l_const >= 0.)]
    print('Граничное значение: r = {}'.format(r_chaos))

    plt.xlabel('$r$')
    plt.ylabel('$x^*$')

    for i, r in enumerate(r_list):
        tmp = np.array(sort_array(x_finish[i]))
        plt.plot([r], tmp.reshape((1, len(tmp))), 'o', markersize = 0.3, color='black')

    plt.axvline(x=r_chaos, color='b', linestyle='--')
    plt.grid()
    plt.savefig('bifur_diag.png', format = 'png')
    plt.clf()

    plt.xlabel('$r$')
    plt.ylabel(r'$\lambda$')
    plt.plot(r_list, l_const)
    plt.grid()
    plt.savefig('lapun_lamb.png', format = 'png')

if __name__ == '__main__':
    main()
