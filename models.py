import numpy as np


def model1(pars):
    Rf, delta, xf = pars.T

    Zs_abs, Zr_abs, Zs_ang, Zr_ang, ks, kr = [15, 50, 87, 87, 1, 1]

    U = 110e3 / np.sqrt(3)
    Z0 = 0.1 + 0.4j

    [Zr_ang, Zs_ang, delta] = [arr * np.pi / 180 for arr in [Zr_ang, Zs_ang, delta]]
    Zr, Zs = Zr_abs * np.exp(1j * Zr_ang), Zs_abs * np.exp(1j * Zs_ang)
    Es, Er = U * np.exp(1j * delta) * ks, U * kr

    Zw1 = Z0 * 100 * xf
    Zw2 = Z0 * 100 * (1 - xf)

    I = (Es * Rf - Er * Rf + Es * Zr + Es * Zw2) / (
        Rf * Zr
        + Rf * Zs
        + Rf * Zw1
        + Rf * Zw2
        + Zr * Zs
        + Zr * Zw1
        + Zs * Zw2
        + Zw1 * Zw2
    )
    U = Es - I * Zs

    ang = np.angle(U)  # angle of base vector
    [U, I] = np.array([U, I]) * np.exp(-1j * ang)  # rotate all to base

    Y = I / U

    return np.c_[Y.real, Y.imag]


def model2(pars):
    Rf, delta, xf = pars.T

    Zs_abs, Zr_abs, Zs_ang, Zr_ang, ks, kr = [15, 50, 87, 87, 1, 1]

    U = 110e3 / np.sqrt(3)
    Z0 = 0.1 + 0.4j

    [Zr_ang, Zs_ang, delta] = [arr * np.pi / 180 for arr in [Zr_ang, Zs_ang, delta]]
    Zr, Zs = Zr_abs * np.exp(1j * Zr_ang), Zs_abs * np.exp(1j * Zs_ang)
    Es, Er = U * np.exp(1j * delta) * ks, U * kr

    Zw1 = Z0 * 100 * xf
    Zw2 = Z0 * 100 * (1 - xf)

    I = (Es * Rf - Er * Rf + Es * Zr + Es * Zw2) / (
        Rf * Zr
        + Rf * Zs
        + Rf * Zw1
        + Rf * Zw2
        + Zr * Zs
        + Zr * Zw1
        + Zs * Zw2
        + Zw1 * Zw2
    )
    U = Es - I * Zs

    ang = np.angle(U)  # angle of base vector
    [U, I] = np.array([U, I]) * np.exp(-1j * ang)  # rotate all to base

    Y = I / U

    return np.c_[Y.real, Y.imag]
