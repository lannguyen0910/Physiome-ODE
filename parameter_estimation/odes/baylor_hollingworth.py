
import torch
import torch.nn as nn


class BaylorHollingworthODE(nn.Module):
    """
    Baylor-Hollingworth ODE.

    States:
        Ca      : free calcium
        Trop    : free troponin
        CaTrop  : calcium-troponin complex

    Parameters:
        k1  : association rate
        k1_ : dissociation rate

    Expected shapes:
        y     : (..., 3)
        theta : (..., 2)
    """

    def forward(self, t, y, theta):
        Ca = y[..., 0]
        Trop = y[..., 1]
        CaTrop = y[..., 2]

        k1 = theta[..., 0]
        k1_ = theta[..., 1]

        # Reaction: Ca + Trop <-> CaTrop
        association = k1 * Ca * Trop
        dissociation = k1_ * CaTrop

        dCa = dissociation - association
        dTrop = dissociation - association
        dCaTrop = association - dissociation

        return torch.stack(
            [dCa, dTrop, dCaTrop],
            dim=-1,
        )
