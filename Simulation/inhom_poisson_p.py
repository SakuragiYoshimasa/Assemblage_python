#coding: utf-8
# inhomogeneous Poisson process
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def lambdaT(t):
    return 20.0 * (1.0 + math.sin(2.0 * math.pi * 3.0 * t))

def output(events):
    print('Event Count %d' % len(events))
    y = [0.5] * len(events)
    b = plt.bar(events, y, width=0.001)
    plt.legend("t", loc='upper left')
    plt.xlim(-0.2, 3.2)
    plt.ylim(0, 0.7)
    plt.xlabel('Time(s)')
    plt.savefig('result.pdf')
    plt.show()

def main():
    interval = 0.0001
    t_i = 0
    t_i_1 = 0
    events = []

    while t_i < 3.0:
        xi = np.random.exponential(1.0)
        while xi >= integrate.quad(lambdaT,t_i_1, t_i)[0]:
            t_i += interval
            if t_i > 3.0:
                output(events)
                return
        events.append(t_i)
        t_i_1 = t_i

if __name__ == '__main__':
    main()
