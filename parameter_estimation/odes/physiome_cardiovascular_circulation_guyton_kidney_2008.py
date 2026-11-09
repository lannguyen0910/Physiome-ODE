import torch
import torch.nn as nn
import math


class CardiovascularCirculationGuytonKidney2008ODE(nn.Module):
    """
    PyTorch translation of cardiovascular_circulation/guyton_kidney_2008/model.py.
    Dynamic states: 4; independent constants: 62; derived constants used by rates: 4.
    Original constant slots: 72; algebraic slots: 56.
    """
    source_file = 'cardiovascular_circulation/guyton_kidney_2008/model.py'
    source_state_dim = 4
    source_constant_dim = 72
    algebraic_dim = 56
    requires_dae_solver = True

    def __init__(
        self,
        parameters=None,
        initial_state=None,
        parameter_names=None,
        state_names=None,
    ):
        super().__init__()

        self.state_dim = 4
        self.parameter_dim = 62

        self.parameters = (
            parameters
            if parameters is not None
            else {'ADHMK': 1, 'AMK': 1.037, 'AMNA': 1, 'ANM': 0.987545, 'ANPX': 1, 'AUM': 1.00066, 'CKE': 4.44092, 'CNA': 142.035, 'HM1': 0.39984739, 'MYOGRS': 1, 'PA': 103.525, 'PAMKRN': 1, 'PPC': 29.9941, 'VTW': 39.8952, 'GBL': 0, 'RAPRSP': 0, 'RFCDFT': 0, 'RCDFPC': 0, 'RCDFDP': 2000, 'RNAUGN': 0.6, 'RNAULL': 0.3, 'RNAUUL': 10, 'RNAUAD': 0, 'ARF': 0.5, 'ANMAM': 1.4, 'ANMARL': 0.86, 'AARK': 1, 'ANPXAF': 1.5, 'AARLL': 4, 'AUMK1': 0.3, 'ANMEM': 1.6, 'EFAFR': 0, 'EARK': 1, 'EARLL': 24, 'REK': 1, 'GPPD': 1, 'GLPCA': 1, 'PXTP': 8, 'GFLC': 0.0208333, 'GFNLL': 0.001, 'MDFL1': 10, 'RTPPR': 0.9, 'RTPPRS': 15.2, 'URFORM': 0.24, 'RFABX': 0.8, 'RVRS': 19.167, 'RTSPRS': 6, 'RABSC': 0.5, 'RFABDP': 1, 'RFABDM': 0.3, 'DTNAR': 0.675, 'DIURET': 1, 'AHMNAR': 0.3, 'DTNARL': 1e-06, 'ANMNAM': 1, 'RFABKM': 0.03, 'MDFLKM': 0.667, 'ANMKEM': 2, 'ANMKEL': 0.3, 'CKEEX': 4, 'RNAGTC': 15, 'GFNDMP': 3}
        )

        self.initial_state = (
            initial_state
            if initial_state is not None
            else {'PAR1': 103.525, 'RNAUG3': 0, 'PLUR': 159.549, 'DTKA': 0.0367573}
        )

        self.parameter_names = (
            list(parameter_names)
            if parameter_names is not None
            else ['ADHMK', 'AMK', 'AMNA', 'ANM', 'ANPX', 'AUM', 'CKE', 'CNA', 'HM1', 'MYOGRS', 'PA', 'PAMKRN', 'PPC', 'VTW', 'GBL', 'RAPRSP', 'RFCDFT', 'RCDFPC', 'RCDFDP', 'RNAUGN', 'RNAULL', 'RNAUUL', 'RNAUAD', 'ARF', 'ANMAM', 'ANMARL', 'AARK', 'ANPXAF', 'AARLL', 'AUMK1', 'ANMEM', 'EFAFR', 'EARK', 'EARLL', 'REK', 'GPPD', 'GLPCA', 'PXTP', 'GFLC', 'GFNLL', 'MDFL1', 'RTPPR', 'RTPPRS', 'URFORM', 'RFABX', 'RVRS', 'RTSPRS', 'RABSC', 'RFABDP', 'RFABDM', 'DTNAR', 'DIURET', 'AHMNAR', 'DTNARL', 'ANMNAM', 'RFABKM', 'MDFLKM', 'ANMKEM', 'ANMKEL', 'CKEEX', 'RNAGTC', 'GFNDMP']
        )

        self.state_names = (
            list(state_names)
            if state_names is not None
            else ['PAR1', 'RNAUG3', 'PLUR', 'DTKA']
        )

    def dynamics(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        theta: torch.Tensor,
    ):

        raise NotImplementedError(
            'This model requires a differentiable algebraic/DAE solve: computeRates() calls an implicit rootfind function'
        )

    def forward(self, t, y, theta):
        return self.dynamics(t, y, theta)
