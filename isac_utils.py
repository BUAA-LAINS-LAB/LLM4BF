import numpy as np


class ISACConfig:
    def __init__(self):
        self.Nt = 12
        self.Nr = 10
        self.L = 10
        self.lambda_ = 1.0
        self.d = self.lambda_ / 2
        self.alpha = 0.5


        self.sigma2C_dBm = 0.0
        self.sigma2R_dBm = 0.0


        self.scale_factor_W = 100.0
        self.scale_factor_CRB = 100.0

        self.n_t = np.arange(-(self.Nt - 1) / 2.0, (self.Nt - 1) / 2.0 + 1.0, 1.0, dtype=np.float64).reshape(-1, 1)
        self.n_r = np.arange(-(self.Nr - 1) / 2.0, (self.Nr - 1) / 2.0 + 1.0, 1.0, dtype=np.float64).reshape(-1, 1)

    @property
    def k0(self):
        return 2.0 * np.pi / self.lambda_

    @property
    def sigma2C(self):
        return 10.0 ** ((self.sigma2C_dBm - 30.0) / 10.0)

    @property
    def sigma2R(self):
        return 10.0 ** ((self.sigma2R_dBm - 30.0) / 10.0)

    def PT(self, P_dB):
        return 10.0 ** ((P_dB - 30.0) / 10.0)

    def Gamma(self, Gamma):
        return 10.0 ** (Gamma / 10.0)


def compact_vector_to_W(w_vec):
    Nt = w_vec.size // 2
    w_real = w_vec[:Nt]
    w_imag = w_vec[Nt:]


    w = w_real + 1j * w_imag


    W = np.outer(w, np.conjugate(w))

    return W


def vectors_to_W_stack(w_pred_vectors_scaled, config, K):
    Nt = config.Nt
    scale_factor_W = config.scale_factor_W
    arr = np.asarray(w_pred_vectors_scaled, dtype=float)


    W_stack = np.zeros((K, Nt, Nt), dtype=np.complex128)
    for k in range(K):
        base_index = k * 2 * Nt
        w_unscaled = arr[base_index: base_index + 2 * Nt] / scale_factor_W
        Wk = compact_vector_to_W(w_unscaled)
        W_stack[k] = Wk
    return W_stack


def compute_channel_H(input_obj):
    H_real = np.asarray(input_obj["H_real"], dtype=float)
    H_imag = np.asarray(input_obj["H_imag"], dtype=float)
    H = H_real + 1j * H_imag
    return H


def compute_radar_A(config, theta):
    k0 = config.k0
    d = config.d

    n_t = config.n_t
    n_r = config.n_r

    a_theta = np.exp(1j * k0 * d * n_t * np.sin(theta))
    b_theta = np.exp(1j * k0 * d * n_r * np.sin(theta))


    A = b_theta @ a_theta.conj().T


    a_dot = 1j * k0 * d * n_t * np.cos(theta) * a_theta
    b_dot = 1j * k0 * d * n_r * np.cos(theta) * b_theta
    Ad = b_dot @ a_theta.conj().T + b_theta @ a_dot.conj().T

    return A, Ad


def compute_crb_for_sample(config, theta, W_stack):
    L = config.L
    alpha = config.alpha

    sigma2R = config.sigma2R


    RX = np.sum(W_stack, axis=0)


    A, Ad = compute_radar_A(config, theta)


    num_CRB = sigma2R * np.trace(A.conj().T @ A @ RX)


    term1 = np.trace(Ad.conj().T @ Ad @ RX)
    term2 = np.trace(A.conj().T @ A @ RX)
    term3 = np.trace(Ad.conj().T @ A @ RX)
    den_CRB = 2.0 * (abs(alpha) ** 2) * L * (term1 * term2 - abs(term3) ** 2)

    num_real = float(np.real(num_CRB))
    den_real = float(np.real(den_CRB))

    if den_real <= 0 or abs(den_real) < 1e-12:
        CRB_rad = float("inf")
    else:
        CRB_rad = num_real / den_real


    CRB_deg = CRB_rad * (180.0 / np.pi) ** 2
    CRB_deg2 = float(np.sqrt(max(CRB_deg, 0.0)))


    crb_scaled = CRB_deg2 * config.scale_factor_CRB

    crb_scaled = float(np.round(crb_scaled, 3))

    return crb_scaled
