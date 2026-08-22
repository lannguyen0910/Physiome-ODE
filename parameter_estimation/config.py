from pathlib import Path

import yaml
import torch


class Config:
    def __init__(self, path="config.yaml"):
        self.path = Path(path)
        with open(self.path, "r") as f:
            self.cfg = yaml.safe_load(f)

    @property
    def seed(self):
        return self.cfg["experiment"]["seed"]

    @property
    def data_path(self):
        return self.cfg["data"]["path"]

    @property
    def batch_size(self):
        return self.cfg["data"]["batch_size"]

    @property
    def num_workers(self):
        return self.cfg["data"]["num_workers"]

    @property
    def fold(self):
        return self.cfg["data"].get("fold", 0)

    @property
    def state_dim(self):
        return self.cfg["model"]["state_dim"]

    @property
    def parameter_dim(self):
        return self.cfg["model"]["parameter_dim"]

    @property
    def hidden_dim(self):
        return self.cfg["model"]["hidden_dim"]

    @property
    def num_gru_layers(self):
        return self.cfg["model"]["num_gru_layers"]

    @property
    def drop_out(self):
        return self.cfg["model"]["drop_out"]

    @property
    def epochs(self):
        return self.cfg["training"]["epochs"]

    @property
    def learning_rate(self):
        return self.cfg["training"]["learning_rate"]

    @property
    def weight_decay(self):
        return self.cfg["training"]["weight_decay"]

    @property
    def parameter_loss_weight(self):
        return self.cfg["training"]["parameter_loss_weight"]

    @property
    def trajectory_loss_weight(self):
        return self.cfg["training"]["trajectory_loss_weight"]

    @property
    def patience(self):
        return self.cfg["training"]["patience"]

    @property
    def gradient_clip(self):
        return self.cfg["training"]["gradient_clip"]

    @property
    def ode_parameters(self):
        return self.cfg["ode"]["parameters"]

    @property
    def initial_state(self):
        return self.cfg["ode"]["initial_state"]

    @property
    def parameter_names(self):
        return self.cfg["ode"]["parameter_names"]

    @property
    def state_names(self):
        return self.cfg["ode"]["state_names"]

    @property
    def checkpoint_path(self):
        return self.cfg["output"]["checkpoint"]

    @property
    def results_path(self):
        return self.cfg["output"]["results"]

    @property
    def figures_path(self):
        return self.cfg["output"]["figures"]

    @property
    def history_path(self):
        return self.cfg["output"].get(
            "history",
            "results/history.pt"
        )

    @property
    def device(self):
        configured_device = self.cfg.get(
            "experiment", {}).get("device", "cuda")

        if configured_device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")
