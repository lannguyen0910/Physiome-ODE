import torch
import torch.nn as nn
import math


class CircadianRhythmsUedaHagiwaraKitano2001ODE(nn.Module):
    """
    PyTorch translation of circadian_rhythms/ueda_hagiwara_kitano_2001/model.py.
    Dynamic states: 10; independent constants: 55; derived constants used by rates: 0.
    Original constant slots: 55; algebraic slots: 0.
    """
    source_file = 'circadian_rhythms/ueda_hagiwara_kitano_2001/model.py'
    source_state_dim = 10
    source_constant_dim = 55
    algebraic_dim = 0
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 10
        self.parameter_dim = 55

        self.parameters = (
            parameters
            if parameters is not None
            else {'B1': 0, 'C1': 0, 'S1': 1.45, 'D1': 0.94, 'L1': 0.3, 'R1': 1.02, 'A1': 0.45, 'D0': 0.012, 'a': 1, 'r': 4, 'S2': 0.48, 'D2': 0.44, 'L2': 0.2, 'Dbt_c': 1, 'V1': 1.45, 'V2': 1.45, 'B2': 0, 'C2': 0, 'S3': 1.45, 'D3': 0.94, 'L3': 0.3, 'R2': 1.02, 'A2': 0.45, 'S4': 0.48, 'D4': 0.44, 'L4': 0.2, 'D5': 0.44, 'L5': 0.2, 'K1': 2, 'K2': 2, 'T1': 1.73, 'T2': 0.72, 'D6': 0.29, 'L6': 0.2, 'B3': 0.6, 'C3': 0, 'S5': 1.63, 'D7': 0.54, 'L7': 0.13, 'R3': 0.89, 'A3': 0.8, 'S6': 0.47, 'D8': 0.6, 'L8': 0.2, 'V3': 1.63, 'V4': 1.63, 'Cyc_c': 1, 'D9': 0.6, 'L9': 0.2, 'K3': 2, 'K4': 2, 'T3': 1.63, 'T4': 0.52, 'D10': 0.3, 'L10': 0.2}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'Per_m': 0.5, 'PT_n': 1, 'CC_n': 0.4, 'Per_c': 0.6, 'Tim_c': 0.8, 'PT_c': 0.9, 'Tim_m': 0.7, 'Clk_m': 0.1, 'Clk_c': 0.2, 'CC_c': 0.3}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['B1', 'C1', 'S1', 'D1', 'L1', 'R1', 'A1', 'D0', 'a', 'r', 'S2', 'D2', 'L2', 'Dbt_c', 'V1', 'V2', 'B2', 'C2', 'S3', 'D3', 'L3', 'R2', 'A2', 'S4', 'D4', 'L4', 'D5', 'L5', 'K1', 'K2', 'T1', 'T2', 'D6', 'L6', 'B3', 'C3', 'S5', 'D7', 'L7', 'R3', 'A3', 'S6', 'D8', 'L8', 'V3', 'V4', 'Cyc_c', 'D9', 'L9', 'K3', 'K4', 'T3', 'T4', 'D10', 'L10']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['Per_m', 'PT_n', 'CC_n', 'Per_c', 'Tim_c', 'PT_c', 'Tim_m', 'Clk_m', 'Clk_c', 'CC_c']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (10)
        Per_m = y[:, 0]
        PT_n = y[:, 1]
        CC_n = y[:, 2]
        Per_c = y[:, 3]
        Tim_c = y[:, 4]
        PT_c = y[:, 5]
        Tim_m = y[:, 6]
        Clk_m = y[:, 7]
        Clk_c = y[:, 8]
        CC_c = y[:, 9]

        # Independent parameters (source constant order)
        B1 = theta[:, 0]  # constants[0]
        C1 = theta[:, 1]  # constants[1]
        S1 = theta[:, 2]  # constants[2]
        D1 = theta[:, 3]  # constants[3]
        L1 = theta[:, 4]  # constants[4]
        R1 = theta[:, 5]  # constants[5]
        A1 = theta[:, 6]  # constants[6]
        D0 = theta[:, 7]  # constants[7]
        a = theta[:, 8]  # constants[8]
        r = theta[:, 9]  # constants[9]
        S2 = theta[:, 10]  # constants[10]
        D2 = theta[:, 11]  # constants[11]
        L2 = theta[:, 12]  # constants[12]
        Dbt_c = theta[:, 13]  # constants[13]
        V1 = theta[:, 14]  # constants[14]
        V2 = theta[:, 15]  # constants[15]
        B2 = theta[:, 16]  # constants[16]
        C2 = theta[:, 17]  # constants[17]
        S3 = theta[:, 18]  # constants[18]
        D3 = theta[:, 19]  # constants[19]
        L3 = theta[:, 20]  # constants[20]
        R2 = theta[:, 21]  # constants[21]
        A2 = theta[:, 22]  # constants[22]
        S4 = theta[:, 23]  # constants[23]
        D4 = theta[:, 24]  # constants[24]
        L4 = theta[:, 25]  # constants[25]
        D5 = theta[:, 26]  # constants[26]
        L5 = theta[:, 27]  # constants[27]
        K1 = theta[:, 28]  # constants[28]
        K2 = theta[:, 29]  # constants[29]
        T1 = theta[:, 30]  # constants[30]
        T2 = theta[:, 31]  # constants[31]
        D6 = theta[:, 32]  # constants[32]
        L6 = theta[:, 33]  # constants[33]
        B3 = theta[:, 34]  # constants[34]
        C3 = theta[:, 35]  # constants[35]
        S5 = theta[:, 36]  # constants[36]
        D7 = theta[:, 37]  # constants[37]
        L7 = theta[:, 38]  # constants[38]
        R3 = theta[:, 39]  # constants[39]
        A3 = theta[:, 40]  # constants[40]
        S6 = theta[:, 41]  # constants[41]
        D8 = theta[:, 42]  # constants[42]
        L8 = theta[:, 43]  # constants[43]
        V3 = theta[:, 44]  # constants[44]
        V4 = theta[:, 45]  # constants[45]
        Cyc_c = theta[:, 46]  # constants[46]
        D9 = theta[:, 47]  # constants[47]
        L9 = theta[:, 48]  # constants[48]
        K3 = theta[:, 49]  # constants[49]
        K4 = theta[:, 50]  # constants[50]
        T3 = theta[:, 51]  # constants[51]
        T4 = theta[:, 52]  # constants[52]
        D10 = theta[:, 53]  # constants[53]
        L10 = theta[:, 54]  # constants[54]

        # Initialize rates and algebraic variables, as in computeRates()
        d_Per_m = torch.zeros_like(y[:, 0])
        d_PT_n = torch.zeros_like(y[:, 0])
        d_CC_n = torch.zeros_like(y[:, 0])
        d_Per_c = torch.zeros_like(y[:, 0])
        d_Tim_c = torch.zeros_like(y[:, 0])
        d_PT_c = torch.zeros_like(y[:, 0])
        d_Tim_m = torch.zeros_like(y[:, 0])
        d_Clk_m = torch.zeros_like(y[:, 0])
        d_Clk_c = torch.zeros_like(y[:, 0])
        d_CC_c = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_Per_m = torch.zeros_like(y[:, 0]) + (C1 + S1 * (((CC_n / A1) ** a + B1) / (1.0 + (PT_n / R1) ** r + (CC_n / A1) ** a + B1)) - (D1 * (Per_m / (L1 + Per_m)) + D0 * Per_m))
        d_Per_c = torch.zeros_like(y[:, 0]) + (S2 * Per_m + V2 * PT_c - (V1 * Per_c * Tim_c + D2 * Dbt_c * (Per_c / (L2 + Per_c)) + D0 * Per_c))
        d_Tim_m = torch.zeros_like(y[:, 0]) + (C2 + S3 * (((CC_n / A2) ** a + B2) / (1.0 + (PT_n / R2) ** r + (CC_n / A2) ** a + B2)) - (D3 * (Tim_m / (L3 + Tim_m)) + D0 * Tim_m))
        d_Tim_c = torch.zeros_like(y[:, 0]) + (S4 * Tim_m + V2 * PT_c - (V1 * Per_c * Tim_c + D4 * (Tim_c / (L4 + Tim_c)) + D0 * Tim_c))
        d_PT_c = torch.zeros_like(y[:, 0]) + (V1 * Per_c * Tim_c + T2 * (PT_n / (K2 + PT_n)) - (V2 * PT_c + T1 * (PT_c / (K1 + PT_c)) + D5 * (PT_c / (L5 + PT_c)) + D0 * PT_c))
        d_PT_n = torch.zeros_like(y[:, 0]) + (T1 * (PT_c / (K1 + PT_c)) - (T2 * (PT_n / (K2 + PT_n)) + D6 * (PT_n / (L6 + PT_n)) + D0 * PT_n))
        d_Clk_m = torch.zeros_like(y[:, 0]) + (C3 + S5 * (((PT_n / A3) ** a + B3) / (1.0 + (CC_n / R3) ** r + (PT_n / A3) ** a + B3)) - (D7 * (Clk_m / (L7 + Clk_m)) + D0 * Clk_m))
        d_Clk_c = torch.zeros_like(y[:, 0]) + (S6 * Clk_m + V4 * CC_c - (V3 * Clk_c * Cyc_c + D8 * (Clk_c / (L8 + Clk_c)) + D0 * Clk_c))
        d_CC_c = torch.zeros_like(y[:, 0]) + (V3 * Clk_c * Cyc_c + T4 * (CC_n / (K4 + CC_n)) - (V4 * CC_c + T3 * (CC_c / (K3 + CC_c)) + D9 * (CC_c / (L9 + CC_c)) + D0 * CC_c))
        d_CC_n = torch.zeros_like(y[:, 0]) + (T3 * (CC_c / (K3 + CC_c)) - (T4 * (CC_n / (K4 + CC_n)) + D10 * (CC_n / (L10 + CC_n)) + D0 * CC_n))

        return torch.stack(
            [d_Per_m, d_PT_n, d_CC_n, d_Per_c, d_Tim_c, d_PT_c, d_Tim_m, d_Clk_m, d_Clk_c, d_CC_c],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
