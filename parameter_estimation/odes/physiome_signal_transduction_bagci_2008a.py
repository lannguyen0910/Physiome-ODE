import torch
import torch.nn as nn
import math


class SignalTransductionBagci2008aODE(nn.Module):
    """
    PyTorch translation of signal_transduction/bagci_2008a/model.py.
    Dynamic states: 9; independent constants: 28; derived constants used by rates: 2.
    Original constant slots: 30; algebraic slots: 19.
    """
    source_file = 'signal_transduction/bagci_2008a/model.py'
    source_state_dim = 9
    source_constant_dim = 30
    algebraic_dim = 19
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 9
        self.parameter_dim = 28

        self.parameters = (
            parameters
            if parameters is not None
            else {'O_2': 35, 'k_1NO': 1, 'k_4NO': 6700, 'k_12aNO': 6e-06, 'k_12bNOp': 1100, 'k_12bNOm': 81000, 'k_14NO': 0.0002, 'k_15NO': 100, 'k_16NO': 1.21, 'SOD': 10, 'k_2NO': 0.1, 'k_5NO': 2400, 'k_10NO': 0.0006, 'GPX': 5.8, 'CO_2': 1000, 'Cyt_c': 400, 'k_6NO': 0.00135, 'k_7NO': 2, 'k_8NO': 0.058, 'k_9NO': 0.025, 'FeLn_0': 0.05, 'GSH_0': 10000, 'k_11NO': 66, 'v_m': 320, 'k_m': 50, 'k_17NO': 66, 'k_13NO': 1600, 'k_17bNO': 0.0002}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'NO': 0, 'O_2m': 0, 'NO_2': 0, 'N2O3': 0, 'GSNO': 0, 'CcOX': 0.1, 'FeLn': 0.05, 'ONOO_m': 0, 'GSH': 10000}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['O_2', 'k_1NO', 'k_4NO', 'k_12aNO', 'k_12bNOp', 'k_12bNOm', 'k_14NO', 'k_15NO', 'k_16NO', 'SOD', 'k_2NO', 'k_5NO', 'k_10NO', 'GPX', 'CO_2', 'Cyt_c', 'k_6NO', 'k_7NO', 'k_8NO', 'k_9NO', 'FeLn_0', 'GSH_0', 'k_11NO', 'v_m', 'k_m', 'k_17NO', 'k_13NO', 'k_17bNO']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['NO', 'O_2m', 'NO_2', 'N2O3', 'GSNO', 'CcOX', 'FeLn', 'ONOO_m', 'GSH']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (9)
        NO = y[:, 0]
        O_2m = y[:, 1]
        NO_2 = y[:, 2]
        N2O3 = y[:, 3]
        GSNO = y[:, 4]
        CcOX = y[:, 5]
        FeLn = y[:, 6]
        ONOO_m = y[:, 7]
        GSH = y[:, 8]

        # Independent parameters (source constant order)
        O_2 = theta[:, 0]  # constants[0]
        k_1NO = theta[:, 1]  # constants[1]
        k_4NO = theta[:, 2]  # constants[2]
        k_12aNO = theta[:, 3]  # constants[3]
        k_12bNOp = theta[:, 4]  # constants[4]
        k_12bNOm = theta[:, 5]  # constants[5]
        k_14NO = theta[:, 6]  # constants[6]
        k_15NO = theta[:, 7]  # constants[7]
        k_16NO = theta[:, 8]  # constants[8]
        SOD = theta[:, 9]  # constants[9]
        k_2NO = theta[:, 10]  # constants[10]
        k_5NO = theta[:, 11]  # constants[11]
        k_10NO = theta[:, 12]  # constants[12]
        GPX = theta[:, 13]  # constants[13]
        CO_2 = theta[:, 14]  # constants[14]
        Cyt_c = theta[:, 15]  # constants[15]
        k_6NO = theta[:, 16]  # constants[16]
        k_7NO = theta[:, 17]  # constants[17]
        k_8NO = theta[:, 18]  # constants[18]
        k_9NO = theta[:, 19]  # constants[19]
        FeLn_0 = theta[:, 20]  # constants[20]
        GSH_0 = theta[:, 21]  # constants[21]
        k_11NO = theta[:, 22]  # constants[22]
        v_m = theta[:, 23]  # constants[23]
        k_m = theta[:, 24]  # constants[24]
        k_17NO = theta[:, 25]  # constants[25]
        k_13NO = theta[:, 26]  # constants[26]
        k_17bNO = theta[:, 27]  # constants[27]

        # Derived constants from the original initConsts()
        r_1NO = k_1NO  # constants[28]
        r_2NO = k_2NO  # constants[29]

        # Initialize rates and algebraic variables, as in computeRates()
        d_NO = torch.zeros_like(y[:, 0])
        d_O_2m = torch.zeros_like(y[:, 0])
        d_NO_2 = torch.zeros_like(y[:, 0])
        d_N2O3 = torch.zeros_like(y[:, 0])
        d_GSNO = torch.zeros_like(y[:, 0])
        d_CcOX = torch.zeros_like(y[:, 0])
        d_FeLn = torch.zeros_like(y[:, 0])
        d_ONOO_m = torch.zeros_like(y[:, 0])
        d_GSH = torch.zeros_like(y[:, 0])
        alg_0_r_4NO = torch.zeros_like(y[:, 0])
        alg_1_r_12aNO = torch.zeros_like(y[:, 0])
        alg_2_r_12bNOp = torch.zeros_like(y[:, 0])
        alg_3_r_12bNOm = torch.zeros_like(y[:, 0])
        alg_4_r_14NO = torch.zeros_like(y[:, 0])
        alg_5_r_15NO = torch.zeros_like(y[:, 0])
        alg_6_r_16NO = torch.zeros_like(y[:, 0])
        alg_7_r_5NO = torch.zeros_like(y[:, 0])
        alg_8_r_10NO = torch.zeros_like(y[:, 0])
        alg_9_r_6NO = torch.zeros_like(y[:, 0])
        alg_10_r_7NO = torch.zeros_like(y[:, 0])
        alg_11_FeLnNO = torch.zeros_like(y[:, 0])
        alg_12_r_8NO = torch.zeros_like(y[:, 0])
        alg_13_GSSG = torch.zeros_like(y[:, 0])
        alg_14_r_9NO = torch.zeros_like(y[:, 0])
        alg_15_r_11NO = torch.zeros_like(y[:, 0])
        alg_16_r_m = torch.zeros_like(y[:, 0])
        alg_17_r_13NO = torch.zeros_like(y[:, 0])
        alg_18_r_17NO = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_1_r_12aNO = k_12aNO * NO ** 2.0 * O_2
        alg_2_r_12bNOp = k_12bNOp * NO_2 * NO
        alg_3_r_12bNOm = k_12bNOm * N2O3
        d_NO_2 = torch.zeros_like(y[:, 0]) + (2.0 * alg_1_r_12aNO - alg_2_r_12bNOp + alg_3_r_12bNOm)
        alg_5_r_15NO = k_15NO * CcOX * NO
        d_CcOX = torch.zeros_like(y[:, 0]) + (-alg_5_r_15NO)
        alg_0_r_4NO = k_4NO * NO * O_2m
        alg_4_r_14NO = k_14NO * GSNO
        alg_6_r_16NO = k_16NO * FeLn * NO
        d_NO = torch.zeros_like(y[:, 0]) + (r_1NO - alg_0_r_4NO - 2.0 * alg_1_r_12aNO - alg_2_r_12bNOp + alg_3_r_12bNOm + alg_4_r_14NO - alg_5_r_15NO - alg_6_r_16NO)
        alg_7_r_5NO = k_5NO * SOD * O_2m
        alg_8_r_10NO = k_10NO * GSNO ** 2.0 * O_2m
        d_O_2m = torch.zeros_like(y[:, 0]) + (r_2NO - alg_0_r_4NO - alg_7_r_5NO - alg_8_r_10NO)
        alg_9_r_6NO = k_6NO * ONOO_m * GSH
        alg_10_r_7NO = k_7NO * ONOO_m * GPX
        alg_12_r_8NO = k_8NO * ONOO_m * CO_2
        alg_14_r_9NO = k_9NO * ONOO_m * Cyt_c
        d_ONOO_m = torch.zeros_like(y[:, 0]) + (alg_0_r_4NO - alg_9_r_6NO - alg_10_r_7NO - alg_12_r_8NO - alg_14_r_9NO)
        alg_15_r_11NO = k_11NO * N2O3 * GSH
        alg_17_r_13NO = k_13NO * N2O3
        d_N2O3 = torch.zeros_like(y[:, 0]) + (-alg_15_r_11NO + alg_2_r_12bNOp - alg_3_r_12bNOm - alg_17_r_13NO)
        alg_13_GSSG = (GSH_0 - GSH - GSNO) / 2.0
        alg_16_r_m = v_m * alg_13_GSSG / (k_m + alg_13_GSSG)
        alg_11_FeLnNO = FeLn_0 - FeLn
        alg_18_r_17NO = k_17NO * alg_11_FeLnNO * GSH
        d_GSH = torch.zeros_like(y[:, 0]) + (-alg_9_r_6NO - alg_15_r_11NO + 2.0 * alg_16_r_m - alg_18_r_17NO)
        d_GSNO = torch.zeros_like(y[:, 0]) + (alg_9_r_6NO - 2.0 * alg_8_r_10NO + alg_15_r_11NO - alg_4_r_14NO + alg_18_r_17NO)
        d_FeLn = torch.zeros_like(y[:, 0]) + (-alg_6_r_16NO + alg_18_r_17NO)

        return torch.stack(
            [d_NO, d_O_2m, d_NO_2, d_N2O3, d_GSNO, d_CcOX, d_FeLn, d_ONOO_m, d_GSH],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
