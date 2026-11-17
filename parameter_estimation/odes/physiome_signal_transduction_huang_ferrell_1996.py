import torch
import torch.nn as nn
import math


class SignalTransductionHuangFerrell1996ODE(nn.Module):
    """
    PyTorch translation of signal_transduction/huang_ferrell_1996/model.py.
    Dynamic states: 18; independent constants: 37; derived constants used by rates: 0.
    Original constant slots: 37; algebraic slots: 0.
    """
    source_file = 'signal_transduction/huang_ferrell_1996/model.py'
    source_state_dim = 18
    source_constant_dim = 37
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

        self.state_dim = 18
        self.parameter_dim = 37

        self.parameters = (
            parameters
            if parameters is not None
            else {'mostot': 0.003, 'e1tot': 0.0003, 'e2tot': 0.0003, 'mektot': 1.2, 'mekpasetot': 0.0003, 'mapktot': 1.2, 'mapkpasetot': 0.12, 'a1': 1000, 'a2': 1000, 'a3': 1000, 'a4': 1000, 'a5': 1000, 'a6': 1000, 'a7': 1000, 'a8': 1000, 'a9': 1000, 'a10': 1000, 'd1': 150, 'd2': 150, 'd3': 150, 'd4': 150, 'd5': 150, 'd6': 150, 'd7': 150, 'd8': 150, 'd9': 150, 'd10': 150, 'k1': 150, 'k2': 150, 'k3': 150, 'k4': 150, 'k5': 150, 'k6': 150, 'k7': 150, 'k8': 150, 'k9': 150, 'k10': 150}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'mos': 0.003, 'mosstar': 0, 'mose1': 0, 'mosstare2': 0, 'mekstarmosstar': 0, 'mekmosstar': 0, 'mekstar': 0, 'mekstarstar': 0, 'mekstarmekpase': 0, 'mekstarstarmekpase': 0, 'mapkmekstarstar': 0, 'mapkstarmekstarstar': 0, 'mek': 1.2, 'mapkstar': 0, 'mapkstarstar': 0, 'mapkstarmapkpase': 0, 'mapkstarstarmapkpase': 0, 'mapk': 1.2}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['mostot', 'e1tot', 'e2tot', 'mektot', 'mekpasetot', 'mapktot', 'mapkpasetot', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['mos', 'mosstar', 'mose1', 'mosstare2', 'mekstarmosstar', 'mekmosstar', 'mekstar', 'mekstarstar', 'mekstarmekpase', 'mekstarstarmekpase', 'mapkmekstarstar', 'mapkstarmekstarstar', 'mek', 'mapkstar', 'mapkstarstar', 'mapkstarmapkpase', 'mapkstarstarmapkpase', 'mapk']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        # States (18)
        mos = y[:, 0]
        mosstar = y[:, 1]
        mose1 = y[:, 2]
        mosstare2 = y[:, 3]
        mekstarmosstar = y[:, 4]
        mekmosstar = y[:, 5]
        mekstar = y[:, 6]
        mekstarstar = y[:, 7]
        mekstarmekpase = y[:, 8]
        mekstarstarmekpase = y[:, 9]
        mapkmekstarstar = y[:, 10]
        mapkstarmekstarstar = y[:, 11]
        mek = y[:, 12]
        mapkstar = y[:, 13]
        mapkstarstar = y[:, 14]
        mapkstarmapkpase = y[:, 15]
        mapkstarstarmapkpase = y[:, 16]
        mapk = y[:, 17]

        # Independent parameters (source constant order)
        mostot = theta[:, 0]  # constants[0]
        e1tot = theta[:, 1]  # constants[1]
        e2tot = theta[:, 2]  # constants[2]
        mektot = theta[:, 3]  # constants[3]
        mekpasetot = theta[:, 4]  # constants[4]
        mapktot = theta[:, 5]  # constants[5]
        mapkpasetot = theta[:, 6]  # constants[6]
        a1 = theta[:, 7]  # constants[7]
        a2 = theta[:, 8]  # constants[8]
        a3 = theta[:, 9]  # constants[9]
        a4 = theta[:, 10]  # constants[10]
        a5 = theta[:, 11]  # constants[11]
        a6 = theta[:, 12]  # constants[12]
        a7 = theta[:, 13]  # constants[13]
        a8 = theta[:, 14]  # constants[14]
        a9 = theta[:, 15]  # constants[15]
        a10 = theta[:, 16]  # constants[16]
        d1 = theta[:, 17]  # constants[17]
        d2 = theta[:, 18]  # constants[18]
        d3 = theta[:, 19]  # constants[19]
        d4 = theta[:, 20]  # constants[20]
        d5 = theta[:, 21]  # constants[21]
        d6 = theta[:, 22]  # constants[22]
        d7 = theta[:, 23]  # constants[23]
        d8 = theta[:, 24]  # constants[24]
        d9 = theta[:, 25]  # constants[25]
        d10 = theta[:, 26]  # constants[26]
        k1 = theta[:, 27]  # constants[27]
        k2 = theta[:, 28]  # constants[28]
        k3 = theta[:, 29]  # constants[29]
        k4 = theta[:, 30]  # constants[30]
        k5 = theta[:, 31]  # constants[31]
        k6 = theta[:, 32]  # constants[32]
        k7 = theta[:, 33]  # constants[33]
        k8 = theta[:, 34]  # constants[34]
        k9 = theta[:, 35]  # constants[35]
        k10 = theta[:, 36]  # constants[36]

        # Initialize rates and algebraic variables, as in computeRates()
        d_mos = torch.zeros_like(y[:, 0])
        d_mosstar = torch.zeros_like(y[:, 0])
        d_mose1 = torch.zeros_like(y[:, 0])
        d_mosstare2 = torch.zeros_like(y[:, 0])
        d_mekstarmosstar = torch.zeros_like(y[:, 0])
        d_mekmosstar = torch.zeros_like(y[:, 0])
        d_mekstar = torch.zeros_like(y[:, 0])
        d_mekstarstar = torch.zeros_like(y[:, 0])
        d_mekstarmekpase = torch.zeros_like(y[:, 0])
        d_mekstarstarmekpase = torch.zeros_like(y[:, 0])
        d_mapkmekstarstar = torch.zeros_like(y[:, 0])
        d_mapkstarmekstarstar = torch.zeros_like(y[:, 0])
        d_mek = torch.zeros_like(y[:, 0])
        d_mapkstar = torch.zeros_like(y[:, 0])
        d_mapkstarstar = torch.zeros_like(y[:, 0])
        d_mapkstarmapkpase = torch.zeros_like(y[:, 0])
        d_mapkstarstarmapkpase = torch.zeros_like(y[:, 0])
        d_mapk = torch.zeros_like(y[:, 0])

        # Algebraic calculations and state derivatives
        d_mos = torch.zeros_like(y[:, 0]) + (-a1 * (mostot - mosstar - mose1 - mosstare2 - mekmosstar - mekstarmosstar) * (e1tot - mose1) + d1 * mose1 + k2 * mosstare2)
        d_mose1 = torch.zeros_like(y[:, 0]) + (a1 * (mostot - mosstar - mose1 - mosstare2 - mekmosstar - mekstarmosstar) * (e1tot - mose1) - (d1 + k1) * mose1)
        d_mosstar = torch.zeros_like(y[:, 0]) + (-a2 * mosstar * (e2tot - mosstare2) + d2 * mosstare2 + k1 * mose1 + (k3 + d3) * mekmosstar - a3 * mosstar * (mektot - mekstar - mekstarstar - mekstarmekpase - mekstarstarmekpase - mekmosstar - mekstarmosstar - mapkmekstarstar - mapkstarmekstarstar) + (k5 + d5) * mekstarmosstar - a5 * mekstar * mosstar)
        d_mosstare2 = torch.zeros_like(y[:, 0]) + (a2 * mosstar * (e2tot - mosstare2) - (d2 + k2) * mosstare2)
        d_mek = torch.zeros_like(y[:, 0]) + (-a3 * (mektot - mekstar - mekstarstar - mekstarmekpase - mekstarstarmekpase - mekmosstar - mekstarmosstar - mapkmekstarstar - mapkstarmekstarstar) * mosstar + d3 * mekmosstar + k4 * mekstarmekpase)
        d_mekmosstar = torch.zeros_like(y[:, 0]) + (a3 * (mektot - mekstar - mekstarstar - mekstarmekpase - mekstarstarmekpase - mekmosstar - mekstarmosstar - mapkmekstarstar - mapkstarmekstarstar) * mosstar - (d3 + k3) * mekmosstar)
        d_mekstar = torch.zeros_like(y[:, 0]) + (-a4 * mekstar * (mekpasetot - mekstarmekpase - mekstarstarmekpase) + d4 * mekstarmekpase + k3 * mekmosstar + k6 * mekstarstarmekpase + d5 * mekstarmosstar - a5 * mekstar * mosstar)
        d_mekstarmekpase = torch.zeros_like(y[:, 0]) + (a4 * mekstar * (mekpasetot - mekstarmekpase - mekstarstarmekpase) - (d4 + k4) * mekstarmekpase)
        d_mekstarmosstar = torch.zeros_like(y[:, 0]) + (a5 * mekstar * mosstar - (d5 + k5) * mekstarmosstar)
        d_mekstarstar = torch.zeros_like(y[:, 0]) + (k5 * mekstarmosstar - a6 * mekstarstar * (mekpasetot - mekstarmekpase - mekstarstarmekpase) + d6 * mekstarstarmekpase - a7 * mekstarstar * (mapktot - mapkstar - mapkstarstar - mapkstarmapkpase - mapkstarstarmapkpase - mapkmekstarstar - mapkstarmekstarstar) + (d7 + k7) * mapkmekstarstar + (d9 + k9) * mapkstarmekstarstar - a9 * mapkstar * mekstarstar)
        d_mekstarstarmekpase = torch.zeros_like(y[:, 0]) + (a6 * mekstarstar * (mekpasetot - mekstarmekpase - mekstarstarmekpase) - (d6 + k6) * mekstarstarmekpase)
        d_mapk = torch.zeros_like(y[:, 0]) + (-a7 * (mapktot - mapkstar - mapkstarstar - mapkstarmapkpase - mapkstarstarmapkpase - mapkmekstarstar - mapkstarmekstarstar) * mekstarstar + d7 * mapkmekstarstar + k8 * mapkstarmapkpase)
        d_mapkmekstarstar = torch.zeros_like(y[:, 0]) + (a7 * (mapktot - mapkstar - mapkstarstar - mapkstarmapkpase - mapkstarstarmapkpase - mapkmekstarstar - mapkstarmekstarstar) * mekstarstar - (d7 + k7) * mapkmekstarstar)
        d_mapkstar = torch.zeros_like(y[:, 0]) + (k7 * mapkmekstarstar - a8 * mapkstar * (mapkpasetot - mapkstarmapkpase - mapkstarstarmapkpase) + d8 * mapkstarmapkpase - a9 * mapkstar * mekstarstar + d9 * mapkstarmekstarstar + k10 * mapkstarstarmapkpase)
        d_mapkstarmekstarstar = torch.zeros_like(y[:, 0]) + (a9 * mapkstar * mekstarstar - (d9 + k9) * mapkstarmekstarstar)
        d_mapkstarstar = torch.zeros_like(y[:, 0]) + (-a10 * mapkstarstar * (mapkpasetot - mapkstarmapkpase - mapkstarstarmapkpase) + d10 * mapkstarstarmapkpase + k9 * mapkstarmekstarstar)
        d_mapkstarmapkpase = torch.zeros_like(y[:, 0]) + (a8 * mapkstar * (mapkpasetot - mapkstarmapkpase - mapkstarstarmapkpase) - (d8 + k8) * mapkstarmapkpase)
        d_mapkstarstarmapkpase = torch.zeros_like(y[:, 0]) + (a10 * mapkstarstar * (mapkpasetot - mapkstarmapkpase - mapkstarstarmapkpase) - (d10 + k10) * mapkstarstarmapkpase)

        return torch.stack(
            [d_mos, d_mosstar, d_mose1, d_mosstare2, d_mekstarmosstar, d_mekmosstar, d_mekstar, d_mekstarstar, d_mekstarmekpase, d_mekstarstarmekpase, d_mapkmekstarstar, d_mapkstarmekstarstar, d_mek, d_mapkstar, d_mapkstarstar, d_mapkstarmapkpase, d_mapkstarstarmapkpase, d_mapk],
            dim=-1,
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
