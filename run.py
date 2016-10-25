import matplotlib.pyplot as plt
import numpy as np


def test_func(x):
    return 4*x**3 - 3*x**2 + 15*x


def dtest_func(x):
    return 12*x**2 - 6*x + 15


def df_scr_1st(f_m1, f_p1, f_p2, h_m1, h_p1, h_p2):
    tmp = (h_m1**2 - (h_p1 + h_p2)**2) / (h_p1**2 - (h_p1 + h_p2)**2)
    a = 1/(-h_m1 - h_p1*tmp - (h_p1 + h_p2)*(1 - tmp))
    b = -tmp*a
    c = -(a + b)
    out = a*f_m1 + b*f_p1 + c*f_p2
    return out


def test_df_scr_1st():
    nxvec = np.linspace(10, int(1e5), 60)
    nxvec = np.array([10, 30, 80, 200, 500, 1000])
    nxvec = np.array([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240])
    nxvec = np.logspace(1, 4, 60).astype(int)
    errors = np.zeros(nxvec.shape)
    hvec = np.zeros(nxvec.shape)
    for indx, nx in np.ndenumerate(nxvec):
        xvec = np.linspace(-6, 12, nx)
        h = np.diff(xvec)[0]
        hvec[indx] = h
        perturb = 0.95*h*(np.random.random(nx) - 0.5)
        xvec_p = xvec + perturb
        df = dtest_func(xvec_p)[1:-2]
        h_p = np.diff(xvec_p)
        f = test_func(xvec_p)
        df_num = df_scr_1st(f[:-3], f[2:-1], f[3:],
                            h_p[:-2], h_p[1:-1], h_p[2:])
        errors[indx] = np.sqrt(np.mean((df - df_num)**2))
    fig, ax = plt.subplots()
    log_h = np.log(hvec)
    log_err = np.log(errors)
    logpoly = np.polyfit(log_h, log_err, 1)
    print(logpoly)
    ax.plot(log_h, log_err, 'or', label="Errors")
    ax.plot(log_h, logpoly[0]*log_h + logpoly[1],
            label="{m:1.2f}x + {b:1.2f}".format(m=logpoly[0], b=logpoly[1]))
    ax.legend(loc="best")
    ax.set_xlabel(r"$\ln\left(h\right)$")
    ax.set_ylabel(r"$\ln\left(e\right)$")
    plt.show()
    return


def main():
    test_df_scr_1st()
    return

if __name__ == "__main__":
    main()
