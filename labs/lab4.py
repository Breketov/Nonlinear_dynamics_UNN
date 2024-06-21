import time
import numpy as np
import matplotlib.pyplot as plt


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


def Bernoulli_solver(init, start_t, end_t, increment, decrement):
    time = start_t
    x = init.copy()
    x_s = [init]
    step = 1 / decrement[0](x) / 100
    times = [time]
    while time < end_t:
        a = decrement[0](x)
        r = np.random.random()
        if r < a*step:
            x[0] -= 1
        time += step
        x_s.append(x.copy())
        times.append(time)
        if x[0] <= 0:
            break

    return times, x_s


def Gillespie_solver(init, start_t, end_t, increment, decrement):
    time = start_t
    x = init.copy()
    x_s = [init]
    times = [time]
    while time < end_t:
        v_plus = []
        for rule in increment:
            v_plus.append(rule(x))
        v_minus = []
        for rule in decrement:
            v_minus.append(rule(x))
        a_0 = np.sum(np.array(v_plus + v_minus))
        if a_0 <= 0:
            break
        
        r1 = max(np.random.random(), 1e-12)
        tao = np.log(1 / r1) / a_0
        time += tao
        prob = np.array(v_plus + v_minus) / a_0
        idx = np.random.choice(2*len(init), p=prob)

        if idx >= len(increment):
            x[idx % len(decrement)] -= 1
        else:
            x[idx] += 1
        x_s.append(x.copy())
        times.append(time)

    return times, x_s


def main():
    np.random.seed(1)

    #! Первая часть
    init1 = 100
    gamma = 0.01
    time_st1 = 0
    time_fi1 = 1000
    t_rk4, x_rk4 = RK4(lambda x, t: -gamma*x, time_st1, time_fi1, init1, 1e-4)

    start = time.time()
    t_ber, x_ber= Bernoulli_solver([init1], time_st1, time_fi1, [lambda x: 0], [lambda x: gamma*x[0]])
    end = time.time()
    print('Время исполнения метода Бернулли ', end - start)

    start = time.time()
    t_gil, x_gil = Gillespie_solver([init1], time_st1, time_fi1, [lambda x: 0], [lambda x: gamma*x[0]])
    end = time.time()
    print('Время исполнения метода Gillespie ', end - start)

    plt.xlabel('$t$')
    plt.ylabel('$x(t)$')
    plt.plot(t_rk4, x_rk4, 'r-', label='Среднее значение')
    plt.plot(t_ber, np.array(x_ber).reshape(-1), 'b-o', markersize=2, label='Метод Бернулли')
    plt.plot(t_gil, np.array(x_gil).reshape(-1), 'g-x', markersize=2, label='Метод Gillespie')
    plt.grid()
    plt.legend()
    plt.savefig('Lab4_1.png')
    plt.clf()

    #! Вторая часть
    init2 = [30, 15]
    n = 6
    alpha = 20
    betta = 1
    time_st2 = 0
    time_fi2 = 50
    t_rk4, x_rk4 = RK4(lambda x, t: np.array([alpha/(1 + x[1]**n) - x[0], betta*(x[0] - x[1])]),
                       time_st2, time_fi2, np.array(init2), 1e-3)

    start = time.time()
    t_gil, x_gil = Gillespie_solver(init2, time_st2, time_fi2, [lambda x: alpha/(1 + x[1]**n), lambda x: betta*x[0]],
                            [lambda x: x[0], lambda x: betta*x[1]])
    end = time.time()
    print('Время исполнения метода Gillespie ', end - start)

    plt.xlabel('$t$')
    plt.ylabel('$m(t)$')

    plt.plot(t_rk4, np.array(x_rk4)[:, 0], 'b-', markersize=2, label='Среднее значение')
    plt.plot(t_gil, np.array(x_gil)[:, 0], 'r-o', markersize=2, label='Метод Gillespie')
    plt.grid()
    plt.legend()
    plt.savefig('Lab4_2.1.png')
    plt.clf()

    plt.xlabel('$t$')
    plt.ylabel('$x(t)$')
    plt.plot(t_rk4, np.array(x_rk4)[:, 1], 'b-', markersize=2, label='Среднее значение')
    plt.plot(t_gil, np.array(x_gil)[:, 1], 'r-o', markersize=2, label='Метод Gillespie')
    plt.grid()
    plt.legend()
    plt.savefig('Lab4_2.2.png')


if __name__ == '__main__':
    main()
