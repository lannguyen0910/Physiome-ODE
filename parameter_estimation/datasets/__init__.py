from registry import Registry

from .dataset import PhysiomeDataset
from .datamodule import PhysiomeDataModule
from .eda import DatasetEDA

DATASET_REGISTRY = Registry("DATASET")


DATASET_REGISTRY.register(PhysiomeDataset)
DATASET_REGISTRY.register(PhysiomeDataModule)


__all__ = [
    "DATASET_REGISTRY",
    "PhysiomeDataset",
    "PhysiomeDataModule",
]
