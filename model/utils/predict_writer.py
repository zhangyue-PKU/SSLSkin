import os
import pandas
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import *


class PredictWriter(BasePredictionWriter):
    
    def __init__(self, write_interval, output_dir, name, index, columns=[]):
        super().__init__(write_interval)
        self.output_dir: str = output_dir
        self.name: str = name
        self.index = index
        self.result: pandas.DataFrame = pandas.DataFrame(columns=columns)
        
        
    def write_on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", prediction: Any, batch_indices: Optional[Sequence[int]], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        
        if trainer.global_rank == 0:
            for img_name, target in zip(*prediction):
                target = target[self.index].item()
                self.result = self.result.append(pandas.DataFrame({"image_name": [img_name], "target": [target]}))
        
    def write_on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", predictions: Sequence[Any], batch_indices: Optional[Sequence[Any]]) -> None:
        
        if trainer.global_rank == 0:
            self.result.to_csv(os.path.join(self.output_dir, self.name), index=False)