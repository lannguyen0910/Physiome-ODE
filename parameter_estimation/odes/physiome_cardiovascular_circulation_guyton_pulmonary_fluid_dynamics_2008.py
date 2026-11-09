import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonPulmonaryFluidDynamics2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_pulmonary_fluid_dynamics_2008/model.py.
    Dynamic states: 2; independent constants: 7; derived constants used by rates: 1.
    Original constant slots: 8; algebraic slots: 13.
    """
    source_file = 'cardiovascular_circulation/guyton_pulmonary_fluid_dynamics_2008/model.py'
    source_state_dim = 2
    source_constant_dim = 8
    algebraic_dim = 13
    requires_dae_solver = False

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 2
        self.parameter_dim = 7

        self.parameters = (
            parameters
            if parameters is not None
            else {'PPC': 29.9941, 'PPA': 15.6376, 'PLA': 2, 'CPP': 71.9719, 'RPV': 1.55719, 'RPA': 1.5683, 'CPF': 0.0003}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'VPF1': 0.0123238, 'PPR1': 0.419998}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['PPC', 'PPA', 'PLA', 'CPP', 'RPV', 'RPA', 'CPF']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['VPF1', 'PPR1']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (2)
        VPF1 = y[:, 0]
        PPR1 = y[:, 1]

        # Independent parameters (source constant order)
        PPC = theta[:, 0]  # constants[0]
        PPA = theta[:, 1]  # constants[1]
        PLA = theta[:, 2]  # constants[2]
        CPP = theta[:, 3]  # constants[3]
        RPV = theta[:, 4]  # constants[4]
        RPA = theta[:, 5]  # constants[5]
        CPF = theta[:, 6]  # constants[6]

        # Derived constants from the original initConsts()
        PCP = (PPA - PLA) * RPV / (RPV + RPA) + PLA  # constants[7]

        # Initialize rates and algebraic variables, as in computeRates()
        d_VPF1 = torch.zeros_like(y[:, 0])
        d_PPR1 = torch.zeros_like(y[:, 0])
        alg_0_VPF = torch.zeros_like(y[:, 0])
        alg_1_PPI = torch.zeros_like(y[:, 0])
        alg_2_PPR = torch.zeros_like(y[:, 0])
        alg_3_CPN = torch.zeros_like(y[:, 0])
        alg_4_POS = torch.zeros_like(y[:, 0])
        alg_5_PFI = torch.zeros_like(y[:, 0])
        alg_6_PPN = torch.zeros_like(y[:, 0])
        alg_7_PLF = torch.zeros_like(y[:, 0])
        alg_8_DFZ = torch.zeros_like(y[:, 0])
        alg_9_PPO = torch.zeros_like(y[:, 0])
        alg_10_DFP = torch.zeros_like(y[:, 0])
        alg_11_PPZ = torch.zeros_like(y[:, 0])
        alg_12_PPD = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        alg_0_VPF = torch.where(VPF1 < 0.001, 0.001, VPF1)
        alg_2_PPR = torch.where(PPR1 < 0.025, 0.025, PPR1)
        alg_3_CPN = alg_2_PPR / alg_0_VPF
        alg_4_POS = alg_3_CPN * 0.4
        alg_1_PPI = 2.0 - 0.15 / alg_0_VPF
        alg_5_PFI = (PCP - alg_1_PPI + alg_4_POS - PPC) * CPF
        alg_7_PLF = (alg_1_PPI + 11.0) * 0.0003
        alg_8_DFZ = alg_5_PFI - alg_7_PLF
        alg_10_DFP = alg_8_DFZ
        d_VPF1 = torch.zeros_like(y[:, 0]) + (alg_10_DFP)
        alg_9_PPO = alg_7_PLF * alg_3_CPN
        alg_6_PPN = (CPP - alg_3_CPN) * 0.000225
        alg_11_PPZ = alg_6_PPN - alg_9_PPO
        alg_12_PPD = alg_11_PPZ
        d_PPR1 = torch.zeros_like(y[:, 0]) + (alg_12_PPD)

        return torch.stack(
            [d_VPF1, d_PPR1],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
