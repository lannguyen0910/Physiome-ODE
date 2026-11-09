import torch
import torch.nn as nn
import math


class CellCycleBarberisKlippVanoniAlberghina2007ODE(nn.Module):
    """
    PyTorch translation of cell_cycle/barberis_klipp_vanoni_alberghina_2007/model.py.
    Dynamic states: 34; independent constants: 55; derived constants used by rates: 0.
    Original constant slots: 55; algebraic slots: 2.
    """
    source_file = 'cell_cycle/barberis_klipp_vanoni_alberghina_2007/model.py'
    source_state_dim = 34
    source_constant_dim = 55
    algebraic_dim = 2
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 34
        self.parameter_dim = 55

        self.parameters = (
            parameters
            if parameters is not None
            else {'k_growth': 0.0051, 'k_50': 0.6, 'k_10': 0.12, 'k_51': 0.6, 'k_11': 0.12, 'k_1': 0.03523, 'k_2': 0.03523, 'k_39': 1, 'k_34': 8.46, 'k_35': 0.0005, 'k_6': 1e-05, 'k_43': 0.005, 'k_15': 0.01, 'k_3': 0.32, 'k_26': 2.82, 'k_27': 0.55, 'k_12': 0.1, 'k_4': 0.32, 'k_28': 2.82, 'k_29': 0.55, 'k_13': 0.35, 'k_7': 0.01, 'k_44': 0.005, 'k_49': 0.001, 'k_16': 0.03, 'k_24': 2.82, 'k_25': 0.55, 'k_20': 0.01, 'k_21': 0, 'k_46': 0.1, 'k_53': 0.001, 'k_33': 0.55, 'k_32': 84.6, 'k_48': 0.012, 'k_41': 1, 'k_31': 0.55, 'k_30': 42300, 'k_40': 1, 'k_36': 4363.6, 'k_37': 4363.6, 'k_47': 1, 'k_38': 4363.6, 'k_9': 5e-05, 'k_18': 0.0008, 'k_8': 4e-05, 'k_45': 0.005, 'k_17': 0.01, 'k_5': 4.2e-05, 'k_42': 0.005, 'k_14': 0.01, 'k_52': 0.005, 'k_22': 0.01, 'k_19': 0.01, 'k_23': 0.01, 'constant_54': 0}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'v_c': 0.5, 'v_n': 0.5, 'mcln2_c': 0, 'mcln2_n': 0, 'mclb5_c': 0, 'mclb5_n': 0, 'sbf_n': 0, 'sbfwhi5p_n': 0, 'whi5_n': 0, 'sbfwhi5_n': 0.025544, 'cln3_c': 0.000485, 'cln2_c': 0, 'cdk1_c': 0.333333, 'cdk1cln2_c': 0, 'clb5_c': 0, 'cdk1clb5_c': 0, 'cdk1_n': 0.0074127, 'cln3_n': 0, 'cdk1cln3_n': 0, 'cdk1cln2_n': 0, 'cdk1clb5sic1_c': 0, 'sic1_c': 0.039234, 'cdk1clb5_n': 0, 'cdk1clb5sic1p_n': 0, 'cdk1cln3far1_n': 0, 'far1_n': 0, 'cdk1cln3far1p_n': 0, 'cdk1clb5sic1_n': 0, 'whi5_c': 0.073564, 'far1_c': 0.0037926, 'whi5p_c': 0, 'whi5p_n': 0, 'sic1P_n': 0, 'far1P_n': 0}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['k_growth', 'k_50', 'k_10', 'k_51', 'k_11', 'k_1', 'k_2', 'k_39', 'k_34', 'k_35', 'k_6', 'k_43', 'k_15', 'k_3', 'k_26', 'k_27', 'k_12', 'k_4', 'k_28', 'k_29', 'k_13', 'k_7', 'k_44', 'k_49', 'k_16', 'k_24', 'k_25', 'k_20', 'k_21', 'k_46', 'k_53', 'k_33', 'k_32', 'k_48', 'k_41', 'k_31', 'k_30', 'k_40', 'k_36', 'k_37', 'k_47', 'k_38', 'k_9', 'k_18', 'k_8', 'k_45', 'k_17', 'k_5', 'k_42', 'k_14', 'k_52', 'k_22', 'k_19', 'k_23', 'constant_54']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['v_c', 'v_n', 'mcln2_c', 'mcln2_n', 'mclb5_c', 'mclb5_n', 'sbf_n', 'sbfwhi5p_n', 'whi5_n', 'sbfwhi5_n', 'cln3_c', 'cln2_c', 'cdk1_c', 'cdk1cln2_c', 'clb5_c', 'cdk1clb5_c', 'cdk1_n', 'cln3_n', 'cdk1cln3_n', 'cdk1cln2_n', 'cdk1clb5sic1_c', 'sic1_c', 'cdk1clb5_n', 'cdk1clb5sic1p_n', 'cdk1cln3far1_n', 'far1_n', 'cdk1cln3far1p_n', 'cdk1clb5sic1_n', 'whi5_c', 'far1_c', 'whi5p_c', 'whi5p_n', 'sic1P_n', 'far1P_n']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (34)
        v_c = y[:, 0]
        v_n = y[:, 1]
        mcln2_c = y[:, 2]
        mcln2_n = y[:, 3]
        mclb5_c = y[:, 4]
        mclb5_n = y[:, 5]
        sbf_n = y[:, 6]
        sbfwhi5p_n = y[:, 7]
        whi5_n = y[:, 8]
        sbfwhi5_n = y[:, 9]
        cln3_c = y[:, 10]
        cln2_c = y[:, 11]
        cdk1_c = y[:, 12]
        cdk1cln2_c = y[:, 13]
        clb5_c = y[:, 14]
        cdk1clb5_c = y[:, 15]
        cdk1_n = y[:, 16]
        cln3_n = y[:, 17]
        cdk1cln3_n = y[:, 18]
        cdk1cln2_n = y[:, 19]
        cdk1clb5sic1_c = y[:, 20]
        sic1_c = y[:, 21]
        cdk1clb5_n = y[:, 22]
        cdk1clb5sic1p_n = y[:, 23]
        cdk1cln3far1_n = y[:, 24]
        far1_n = y[:, 25]
        cdk1cln3far1p_n = y[:, 26]
        cdk1clb5sic1_n = y[:, 27]
        whi5_c = y[:, 28]
        far1_c = y[:, 29]
        whi5p_c = y[:, 30]
        whi5p_n = y[:, 31]
        sic1P_n = y[:, 32]
        far1P_n = y[:, 33]

        # Independent parameters (source constant order)
        k_growth = theta[:, 0]  # constants[0]
        k_50 = theta[:, 1]  # constants[1]
        k_10 = theta[:, 2]  # constants[2]
        k_51 = theta[:, 3]  # constants[3]
        k_11 = theta[:, 4]  # constants[4]
        k_1 = theta[:, 5]  # constants[5]
        k_2 = theta[:, 6]  # constants[6]
        k_39 = theta[:, 7]  # constants[7]
        k_34 = theta[:, 8]  # constants[8]
        k_35 = theta[:, 9]  # constants[9]
        k_6 = theta[:, 10]  # constants[10]
        k_43 = theta[:, 11]  # constants[11]
        k_15 = theta[:, 12]  # constants[12]
        k_3 = theta[:, 13]  # constants[13]
        k_26 = theta[:, 14]  # constants[14]
        k_27 = theta[:, 15]  # constants[15]
        k_12 = theta[:, 16]  # constants[16]
        k_4 = theta[:, 17]  # constants[17]
        k_28 = theta[:, 18]  # constants[18]
        k_29 = theta[:, 19]  # constants[19]
        k_13 = theta[:, 20]  # constants[20]
        k_7 = theta[:, 21]  # constants[21]
        k_44 = theta[:, 22]  # constants[22]
        k_49 = theta[:, 23]  # constants[23]
        k_16 = theta[:, 24]  # constants[24]
        k_24 = theta[:, 25]  # constants[25]
        k_25 = theta[:, 26]  # constants[26]
        k_20 = theta[:, 27]  # constants[27]
        k_21 = theta[:, 28]  # constants[28]
        k_46 = theta[:, 29]  # constants[29]
        k_53 = theta[:, 30]  # constants[30]
        k_33 = theta[:, 31]  # constants[31]
        k_32 = theta[:, 32]  # constants[32]
        k_48 = theta[:, 33]  # constants[33]
        k_41 = theta[:, 34]  # constants[34]
        k_31 = theta[:, 35]  # constants[35]
        k_30 = theta[:, 36]  # constants[36]
        k_40 = theta[:, 37]  # constants[37]
        k_36 = theta[:, 38]  # constants[38]
        k_37 = theta[:, 39]  # constants[39]
        k_47 = theta[:, 40]  # constants[40]
        k_38 = theta[:, 41]  # constants[41]
        k_9 = theta[:, 42]  # constants[42]
        k_18 = theta[:, 43]  # constants[43]
        k_8 = theta[:, 44]  # constants[44]
        k_45 = theta[:, 45]  # constants[45]
        k_17 = theta[:, 46]  # constants[46]
        k_5 = theta[:, 47]  # constants[47]
        k_42 = theta[:, 48]  # constants[48]
        k_14 = theta[:, 49]  # constants[49]
        k_52 = theta[:, 50]  # constants[50]
        k_22 = theta[:, 51]  # constants[51]
        k_19 = theta[:, 52]  # constants[52]
        k_23 = theta[:, 53]  # constants[53]
        constant_54 = theta[:, 54]  # constants[54]

        # Initialize rates and algebraic variables, as in computeRates()
        d_v_c = torch.zeros_like(y[:, 0])
        d_v_n = torch.zeros_like(y[:, 0])
        d_mcln2_c = torch.zeros_like(y[:, 0])
        d_mcln2_n = torch.zeros_like(y[:, 0])
        d_mclb5_c = torch.zeros_like(y[:, 0])
        d_mclb5_n = torch.zeros_like(y[:, 0])
        d_sbf_n = torch.zeros_like(y[:, 0])
        d_sbfwhi5p_n = torch.zeros_like(y[:, 0])
        d_whi5_n = torch.zeros_like(y[:, 0])
        d_sbfwhi5_n = torch.zeros_like(y[:, 0])
        d_cln3_c = torch.zeros_like(y[:, 0])
        d_cln2_c = torch.zeros_like(y[:, 0])
        d_cdk1_c = torch.zeros_like(y[:, 0])
        d_cdk1cln2_c = torch.zeros_like(y[:, 0])
        d_clb5_c = torch.zeros_like(y[:, 0])
        d_cdk1clb5_c = torch.zeros_like(y[:, 0])
        d_cdk1_n = torch.zeros_like(y[:, 0])
        d_cln3_n = torch.zeros_like(y[:, 0])
        d_cdk1cln3_n = torch.zeros_like(y[:, 0])
        d_cdk1cln2_n = torch.zeros_like(y[:, 0])
        d_cdk1clb5sic1_c = torch.zeros_like(y[:, 0])
        d_sic1_c = torch.zeros_like(y[:, 0])
        d_cdk1clb5_n = torch.zeros_like(y[:, 0])
        d_cdk1clb5sic1p_n = torch.zeros_like(y[:, 0])
        d_cdk1cln3far1_n = torch.zeros_like(y[:, 0])
        d_far1_n = torch.zeros_like(y[:, 0])
        d_cdk1cln3far1p_n = torch.zeros_like(y[:, 0])
        d_cdk1clb5sic1_n = torch.zeros_like(y[:, 0])
        d_whi5_c = torch.zeros_like(y[:, 0])
        d_far1_c = torch.zeros_like(y[:, 0])
        d_whi5p_c = torch.zeros_like(y[:, 0])
        d_whi5p_n = torch.zeros_like(y[:, 0])
        d_sic1P_n = torch.zeros_like(y[:, 0])
        d_far1P_n = torch.zeros_like(y[:, 0])
        alg_0_dv_c_dt = torch.zeros_like(y[:, 0])
        alg_1_k_volume = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_v_n = torch.zeros_like(y[:, 0]) + (constant_54)
        d_mcln2_n = torch.zeros_like(y[:, 0]) + (k_1 * sbf_n - k_50 * mcln2_n)
        d_mclb5_n = torch.zeros_like(y[:, 0]) + (k_2 * sbf_n - k_51 * mclb5_n)
        d_sbf_n = torch.zeros_like(y[:, 0]) + (k_39 * sbfwhi5p_n - k_34 * sbf_n * whi5_n + k_35 * sbfwhi5_n)
        d_cdk1cln3_n = torch.zeros_like(y[:, 0]) + (k_24 * cln3_n * cdk1_n - k_25 * cdk1cln3_n + k_31 * cdk1cln3far1_n - k_30 * far1_n * cdk1cln3_n + k_40 * cdk1cln3far1p_n)
        d_sbfwhi5_n = torch.zeros_like(y[:, 0]) + (k_34 * sbf_n * whi5_n - k_35 * sbfwhi5_n - k_36 * sbfwhi5_n * cdk1cln3_n)
        d_sbfwhi5p_n = torch.zeros_like(y[:, 0]) + (k_36 * sbfwhi5_n * cdk1cln3_n - k_39 * sbfwhi5p_n)
        d_cdk1cln3far1_n = torch.zeros_like(y[:, 0]) + (k_30 * far1_n * cdk1cln3_n - k_31 * cdk1cln3far1_n - k_37 * cdk1cln3far1_n * cdk1cln2_n)
        d_cdk1clb5sic1p_n = torch.zeros_like(y[:, 0]) + (k_38 * cdk1clb5sic1_n * cdk1cln2_n - k_41 * cdk1clb5sic1p_n)
        d_cdk1cln3far1p_n = torch.zeros_like(y[:, 0]) + (k_37 * cdk1cln3far1_n * cdk1cln2_n - k_40 * cdk1cln3far1p_n)
        d_sic1P_n = torch.zeros_like(y[:, 0]) + (k_41 * cdk1clb5sic1p_n)
        d_far1P_n = torch.zeros_like(y[:, 0]) + (k_40 * cdk1cln3far1p_n)
        d_whi5p_n = torch.zeros_like(y[:, 0]) + (k_39 * sbfwhi5p_n - k_52 * whi5p_n - k_23 * whi5p_n)
        alg_0_dv_c_dt = k_growth * v_c
        d_v_c = torch.zeros_like(y[:, 0]) + (alg_0_dv_c_dt)
        d_cln3_c = torch.zeros_like(y[:, 0]) + (k_6 - k_43 * cln3_c - k_15 * cln3_c - alg_0_dv_c_dt / v_c * cln3_c)
        d_cln2_c = torch.zeros_like(y[:, 0]) + (k_3 * mcln2_c - k_26 * cdk1_c * cln2_c + k_27 * cdk1cln2_c - k_12 * cln2_c - alg_0_dv_c_dt / v_c * cln2_c)
        d_clb5_c = torch.zeros_like(y[:, 0]) + (k_4 * mclb5_c - k_28 * clb5_c * cdk1_c + k_29 * cdk1clb5_c - k_13 * clb5_c - alg_0_dv_c_dt / v_c * clb5_c)
        d_cdk1_c = torch.zeros_like(y[:, 0]) + (k_7 - k_44 * cdk1_c + k_49 * cdk1_n + k_27 * cdk1cln2_c - k_26 * cdk1_c * cln2_c - k_28 * cdk1_c * clb5_c + k_29 * cdk1clb5_c - k_16 * cdk1_c - alg_0_dv_c_dt / v_c * cdk1_c)
        d_cdk1clb5_c = torch.zeros_like(y[:, 0]) + (k_28 * cdk1_c * clb5_c - k_29 * cdk1clb5_c + k_33 * cdk1clb5sic1_c - k_32 * sic1_c * cdk1clb5_c - k_48 * cdk1clb5_c - alg_0_dv_c_dt / v_c * cdk1clb5_c)
        d_cdk1clb5sic1_c = torch.zeros_like(y[:, 0]) + (k_32 * sic1_c * cdk1clb5_c - k_33 * cdk1clb5sic1_c - k_47 * cdk1clb5sic1_c - alg_0_dv_c_dt / v_c * cdk1clb5sic1_c)
        d_sic1_c = torch.zeros_like(y[:, 0]) + (k_9 - k_32 * sic1_c * cdk1clb5_c + k_33 * cdk1clb5sic1_c - k_18 * sic1_c - alg_0_dv_c_dt / v_c * sic1_c)
        d_whi5_c = torch.zeros_like(y[:, 0]) + (k_8 - k_45 * whi5_c - k_17 * whi5_c - alg_0_dv_c_dt / v_c * whi5_c)
        d_far1_c = torch.zeros_like(y[:, 0]) + (k_5 - k_42 * far1_c - k_14 * far1_c - alg_0_dv_c_dt / v_c * far1_c)
        alg_1_k_volume = v_n / v_c
        d_mcln2_c = torch.zeros_like(y[:, 0]) + (k_50 * mcln2_n * alg_1_k_volume - k_10 * mcln2_c - alg_0_dv_c_dt / v_c * mcln2_c)
        d_mclb5_c = torch.zeros_like(y[:, 0]) + (k_51 * mclb5_n * alg_1_k_volume - k_11 * mclb5_c - alg_0_dv_c_dt / v_c * mclb5_c)
        d_cln3_n = torch.zeros_like(y[:, 0]) + (k_43 * cln3_c / alg_1_k_volume - k_24 * cln3_n * cdk1_n + k_25 * cdk1cln3_n - k_20 * cln3_n)
        d_cdk1_n = torch.zeros_like(y[:, 0]) + (k_44 * cdk1_c / alg_1_k_volume - k_49 * cdk1_n - k_24 * cln3_n * cdk1_n + k_25 * cdk1cln3_n - k_21 * cdk1_n)
        d_cdk1cln2_c = torch.zeros_like(y[:, 0]) + (k_26 * cdk1_c * cln2_c - k_27 * cdk1cln2_c - k_46 * cdk1cln2_c + k_53 * cdk1cln2_n * alg_1_k_volume - alg_0_dv_c_dt / v_c * cdk1cln2_c)
        d_cdk1cln2_n = torch.zeros_like(y[:, 0]) + (k_46 * cdk1cln2_c / alg_1_k_volume - k_53 * cdk1cln2_n)
        d_cdk1clb5_n = torch.zeros_like(y[:, 0]) + (k_41 * cdk1clb5sic1p_n + k_48 * cdk1clb5_c / alg_1_k_volume)
        d_cdk1clb5sic1_n = torch.zeros_like(y[:, 0]) + (k_47 * cdk1clb5sic1_c / alg_1_k_volume - k_38 * cdk1clb5sic1_n * cdk1cln2_n)
        d_whi5p_c = torch.zeros_like(y[:, 0]) + (k_52 * whi5p_n * alg_1_k_volume)
        d_whi5_n = torch.zeros_like(y[:, 0]) + (k_45 * whi5_c / alg_1_k_volume - k_34 * whi5_n * sbf_n - k_22 * whi5_n)
        d_far1_n = torch.zeros_like(y[:, 0]) + (k_42 * far1_c / alg_1_k_volume - k_30 * far1_n * cdk1cln3_n + k_31 * cdk1cln3far1_n - k_19 * far1_n)

        return torch.stack(
            [d_v_c, d_v_n, d_mcln2_c, d_mcln2_n, d_mclb5_c, d_mclb5_n, d_sbf_n, d_sbfwhi5p_n, d_whi5_n, d_sbfwhi5_n, d_cln3_c, d_cln2_c, d_cdk1_c, d_cdk1cln2_c, d_clb5_c, d_cdk1clb5_c, d_cdk1_n, d_cln3_n, d_cdk1cln3_n, d_cdk1cln2_n, d_cdk1clb5sic1_c, d_sic1_c, d_cdk1clb5_n, d_cdk1clb5sic1p_n, d_cdk1cln3far1_n, d_far1_n, d_cdk1cln3far1p_n, d_cdk1clb5sic1_n, d_whi5_c, d_far1_c, d_whi5p_c, d_whi5p_n, d_sic1P_n, d_far1P_n],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
