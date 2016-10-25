import matplotlib.pyplot as plt
import numpy as np


def test_func(x):
    return 4*x**3 - 3*x**2 + 15*x


def dtest_func(x):
    return 12*x**2 - 6*x + 15


def df_cent_1st(fvec, hvec):
    f_m1, f_0, f_p1 = fvec
    h_m1, h_p1 = hvec
    tmp = 1 + h_p1/h_m1
    c = 1/(h_p1*tmp)
    a = -h_p1**2/h_m1**2 * c
    b = -(a + c)
    out = a*f_m1 + b*f_0 + c*f_p1
    return out


def df_right_1st(fvec, hvec):
    f_0, f_p1, f_p2 = fvec
    h_p1, h_p2 = hvec
    c = -h_p1/(h_p2*(h_p1 + h_p2))
    b = (h_p1 + h_p2)/(h_p1*h_p2)
    a = -(b + c)
    out = a*f_0 + b*f_p1 + c*f_p2
    return out


def df_left_1st(fvec, hvec):
    f_0, f_m1, f_m2 = fvec
    h_m1, h_m2 = hvec
    c = h_m1/(h_m2*(h_m1 + h_m2))
    b = -(h_m1 + h_m2)/(h_m1*h_m2)
    a = -(b + c)
    out = a*f_0 + b*f_m1 + c*f_m2
    return out


def df_scr_1st(fvec, hvec):
    f_m1, f_p1, f_p2 = fvec
    h_m1, h_p1, h_p2 = hvec
    tmp = (h_m1**2 - (h_p1 + h_p2)**2) / (h_p1**2 - (h_p1 + h_p2)**2)
    a = 1/(-h_m1 - h_p1*tmp - (h_p1 + h_p2)*(1 - tmp))
    b = -tmp*a
    c = -(a + b)
    out = a*f_m1 + b*f_p1 + c*f_p2
    return out


def test_df_cent_1st():
    ttl = "centered, 1st deriv"
    xslice = slice(1, -1)
    fslices = [slice(0, -2), slice(1, -1), slice(2, None)]
    hslices = [slice(0, -1), slice(1, None)]
    test_df_num(df_cent_1st, xslice, fslices, hslices, ttl)
    return


def test_df_right_1st():
    ttl = "right-sided, 1st deriv"
    xslice = slice(0, -2)
    fslices = [slice(0, -2), slice(1, -1), slice(2, None)]
    hslices = [slice(0, -1), slice(1, None)]
    test_df_num(df_right_1st, xslice, fslices, hslices, ttl)
    return


def test_df_left_1st():
    ttl = "left-sided, 1st deriv"
    xslice = slice(2, None)
    fslices = [slice(2, None), slice(1, -1), slice(0, -2)]
    hslices = [slice(1, None), slice(0, -1)]
    test_df_num(df_left_1st, xslice, fslices, hslices, ttl)
    return


def test_df_scr_1st():
    ttl = "semi-centered-right, 1st deriv"
    xslice = slice(1, -2)
    fslices = [slice(0, -3), slice(2, -1), slice(3, None)]
    hslices = [slice(0, -2), slice(1, -1), slice(2, None)]
    test_df_num(df_scr_1st, xslice, fslices, hslices, ttl)
    return


def test_df_num(dfunc_num, xslice, fslices, hslices, ttl):
    make_plot = False
    nxvec = np.logspace(1, 4, 60).astype(int)
    errors = np.zeros(nxvec.shape)
    hvec = np.zeros(nxvec.shape)
    for indx, nx in np.ndenumerate(nxvec):
        xvec = np.linspace(-6, 12, nx)
        h = np.diff(xvec)[0]
        hvec[indx] = h
        perturb = 0.95*h*(np.random.random(nx) - 0.5)
        xvec_p = xvec + perturb
        df = dtest_func(xvec_p)[xslice]
        h_p = np.diff(xvec_p)
        f = test_func(xvec_p)
        fvecs = [f[fslice] for fslice in fslices]
        hvecs = [h_p[hslice] for hslice in hslices]
        df_num = dfunc_num(fvecs, hvecs)
        errors[indx] = np.sqrt(np.mean((df - df_num)**2))
    log_h = np.log(hvec)
    log_err = np.log(errors)
    logpoly = np.polyfit(log_h, log_err, 1)
    if make_plot:
        fig, ax = plt.subplots()
        ax.plot(log_h, log_err, 'or', label="Errors")
        ax.plot(log_h, logpoly[0]*log_h + logpoly[1],
                label="{m:1.2f}x + {b:1.2f}".format(m=logpoly[0], b=logpoly[1]))
        ax.legend(loc="best")
        ax.set_xlabel(r"$\ln\left(h\right)$")
        ax.set_ylabel(r"$\ln\left(e\right)$")
        ax.set_title(ttl)
    else:
        print(ttl)
        print("Convergence order:", logpoly[0])
    return


def main():
    test_df_cent_1st()
    test_df_left_1st()
    test_df_right_1st()
    test_df_scr_1st()
    plt.show()
    return

if __name__ == "__main__":
    main()
