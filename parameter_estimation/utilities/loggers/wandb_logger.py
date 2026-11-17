from typing import Optional, Set
from pathlib import Path
from typing import Dict

try:
    import wandb as wandb_logger
except ModuleNotFoundError:
    pass

import os.path as osp

import torch

from .observer import LoggerObserver, LoggerSubscriber

LOGGER = LoggerObserver.getLogger("main")


class WandbLogger(LoggerSubscriber):
    """
    Logger for wandb intergration
    :param log_dir: Path to save checkpoint
    """

    def __init__(
        self,
        unique_id: str,
        username: str,
        project_name: str,
        run_name: str,
        group_name: str = None,
        save_dir: str = None,
        config_dict: Dict = None,
        **kwargs,
    ):
        self.project_name = project_name
        self.username = username
        self.run_name = run_name
        self.config_dict = config_dict
        self.id = unique_id
        self.save_dir = save_dir
        self.group_name = group_name

        wandb_logger.init(
            id=self.id,
            dir=self.save_dir,
            config=config_dict,
            entity=username,
            group=self.group_name,
            project=project_name,
            name=run_name,
            resume="allow",
            reinit=kwargs.get("reinit", False),
            tags=kwargs.get("tags", None),
        )

        wandb_logger.watch_called = False

    def load_state_dict(self, path):
        if wandb_logger.run.resumed:
            state_dict = torch.load(wandb_logger.restore(path))
            return state_dict
        else:
            return None

    def log_file(self, tag, value, base_folder=None, **kwargs):
        """
        Write a file to wandb
        :param tag: (str) tag
        :param value: (str) path to file

        :param base_folder: (str) folder to save file to
        """
        wandb_logger.save(value, base_path=base_folder)

    def log_scalar(self, tag, value, step=0, **kwargs):
        """
        Write a log to specified directory
        :param tags: (str) tag for log
        :param values: (number) value for corresponding tag
        :param step: (int) logging step
        """

        if step is None:
            step = 0

        wandb_logger.define_metric("iterations")
        wandb_logger.define_metric(
            tag,
            step_metric="iterations",
        )

        wandb_logger.log(
            {
                tag: float(value),
                "iterations": int(step),
            }
        )

    def log_figure(self, tag, value, step=0, **kwargs):
        """
        Write a matplotlib fig to wandb
        :param tags: (str) tag for log
        :param value: (image) image to log. torch.Tensor or plt.fire.Figure
        :param step: (int) logging step
        """

        if isinstance(value, torch.Tensor):
            image = wandb_logger.Image(value)
            wandb_logger.log({tag: image, "iterations": step})
        else:
            wandb_logger.log({tag: value, "iterations": step})

    def log_torch_module(self, tag, value, log_freq, **kwargs):
        """
        Write a model graph to wandb
        :param value: (nn.Module) torch model
        :param inputs: sample tensor
        """
        wandb_logger.watch(value, log="gradients", log_freq=log_freq)

    def log_spec_text(self, tag, value, step, **kwargs):
        """
        Write a text to wandb
        :param value: (str) captions
        """
        texts = wandb_logger.Html(value)
        wandb_logger.log({tag: texts, "iterations": step})

    def log_table(self, tag, value, columns, step, **kwargs):
        """
        Write a table to wandb
        :param value: list of column values
        :param columns: list of column names

        Examples:
        value = [
            [0, fig1, 0],
            [1, fig2, 8],
            [2, fig3, 7],
            [3, fig4, 1]
        ]
        columns=[
            "id",
            "image",
            "prediction"
        ]
        """

        # Workaround for tensor image, have not figured out how to use plt.Figure :<
        new_value = []
        for record in value:
            new_record = []
            for val in record:
                if isinstance(val, torch.Tensor):
                    val = wandb_logger.Image(val)
                new_record.append(val)
            new_value.append(new_record)

        table = wandb_logger.Table(data=new_value, columns=columns)
        wandb_logger.log({tag: table, "iterations": step})

    def log_video(self, tag, value, step, fps, **kwargs):
        """
        Write a video to wandb
        :param value: numpy array (time, channel, height, width)
        :param fps: int
        """
        # axes are
        wandb_logger.log({tag: wandb_logger.Video(
            value, fps=fps), "iterations": step})

    def log_html(self, tag, value, step=0, **kwargs):
        """
        Display a html
        :param value: path to html file
        """
        table = wandb_logger.Table(columns=[tag])
        table.add_data(wandb_logger.Html(value))
        wandb_logger.log({tag: table, "iterations": step})

    def log_embedding(
        self,
        tag,
        value,
        label_img=None,
        step=0,
        metadata=None,
        metadata_header=None,
        **kwargs,
    ):
        """
        Write a embedding projection to tensorboard
        :param value: embeddings array (N, D)
        :param label_img: (torch.Tensor) normalized image tensors (N, 3, H, W)
        :param metadata: (List) zipped list of metadata
        :param metadata_header: (List) list of metadata names according to the metadata provided
        """

        import pandas as pd

        df_dict = {"embeddings": [e for e in value.tolist()]}
        if metadata is not None and metadata_header is not None:
            for meta in metadata:
                for idx, item in enumerate(meta):
                    if metadata_header[idx] not in df_dict.keys():
                        df_dict[metadata_header[idx]] = []
                    df_dict[metadata_header[idx]].append(item)
        if label_img is not None:
            df_dict["images"] = [wandb_logger.Image(
                i.values) for i in label_img]

        df = pd.DataFrame(df_dict)

        table = wandb_logger.Table(
            columns=df.columns.to_list(), data=df.values)
        wandb_logger.log({tag: table, "iterations": step})

    def __del__(self):
        wandb_logger.finish()


class WandbFigureLogger:
    """
    Upload locally saved figures to the active Weights & Biases run.

    Local figures are kept unchanged. Each image is uploaded using a
    key based on its path relative to the figures directory.

    Example:
        figures/eda/train/missing_values.png
        -> figures/eda/train/missing_values.png

        figures/training_loss.png
        -> figures/training_loss.png
    """

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".gif",
    }

    def __init__(self, figure_dir, logger=None):
        self.figure_dir = Path(figure_dir)
        self.logger = logger
        self._uploaded_files: Set[str] = set()

    def _log(self, message):
        """Use the existing project logger when available."""
        if self.logger is not None:
            try:
                self.logger.text(message)
                return
            except Exception:
                pass

        print(message)

    def upload_all(self, directory: Optional[str] = None):
        """
        Scan a directory recursively and upload all supported image files.

        Args:
            directory:
                Directory to scan. Defaults to the complete figure directory.

        Returns:
            Number of newly uploaded images.
        """
        try:
            import wandb
        except ImportError:
            self._log(
                "[W&B FIGURES] wandb is not installed; "
                "skipping figure upload."
            )
            return 0

        # W&B must have an active run, normally created by WandbLogger.
        if wandb.run is None:
            self._log(
                "[W&B FIGURES] No active W&B run. "
                "Figures remain saved locally."
            )
            return 0

        root = Path(directory) if directory is not None else self.figure_dir

        if not root.exists():
            return 0

        uploaded_count = 0

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue

            if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            # Do not upload the same unchanged file repeatedly.
            resolved_path = str(path.resolve())

            if resolved_path in self._uploaded_files:
                continue

            try:
                relative_path = path.relative_to(self.figure_dir).as_posix()

                # Keep folder structure in W&B so figures remain organized.
                wandb_key = f"figures/{relative_path}"

                wandb.log({
                    wandb_key: wandb.Image(str(path))
                })

                self._uploaded_files.add(resolved_path)
                uploaded_count += 1

                self._log(
                    f"[W&B FIGURES] Uploaded: {relative_path}"
                )

            except Exception as exc:
                self._log(
                    f"[W&B FIGURES] Failed to upload "
                    f"{path}: {exc}"
                )

        if uploaded_count:
            self._log(
                f"[W&B FIGURES] Uploaded {uploaded_count} new figure(s)."
            )

        return uploaded_count


def find_run_id(dirname):
    """
    Read a .txt file which contains wandb run id
    """

    wandb_id_file = osp.join(dirname, "wandb_id.txt")

    if not osp.isfile(wandb_id_file):
        raise ValueError(f"Wandb ID file not found in {wandb_id_file}")
    else:
        with open(wandb_id_file, "r") as f:
            wandb_id = f.read().rstrip()
        return wandb_id
