from registry import Registry

from .dataset import PhysiomeDataset
from .datamodule import PhysiomeDataModule
from .eda import DatasetEDA

DATASET_REGISTRY = Registry("DATASET")


DATASET_REGISTRY.register(
    PhysiomeDataset,
    name="physiome_dataset",
)

DATASET_REGISTRY.register(
    PhysiomeDataModule,
    name="physiome_datamodule",
)


__all__ = [
    "DATASET_REGISTRY",
    "PhysiomeDataset",
    "PhysiomeDataModule",
]
