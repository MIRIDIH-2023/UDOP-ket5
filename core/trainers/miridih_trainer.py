from transformers import Trainer, TrainingArguments
import math
from typing import Dict, Union, Any
import torch
from torch import nn, Tensor

# logarithmic ascend
def compute_new_ratio(epoch):
    return 0.2 * math.log2(epoch+4) if epoch < 28 else 1.0

class MIRIDIH_Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_epoch = -1
        self.ratio = 0.4

    def compute_new_ratio(self, epoch):
        self.save_epoch = epoch
        return 0.2 * math.log2(epoch+4) if epoch < 28 else 1.0

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.save_epoch == (epoch-1):
            print(f'save_epoch = {self.save_epoch}')
            self.ratio = self.compute_new_ratio(epoch) 
            print(f'ratio = {self.ratio}')
            self.train_dataset.set_lm_ratio(self.ratio)
        
        super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
      
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
            
        if self.ratio is not None :
            logs["masking ratio"] = self.ratio

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)