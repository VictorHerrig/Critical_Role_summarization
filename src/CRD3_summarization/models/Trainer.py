""""""
import logging
import os
from typing import Optional

import torch
from tokenizers import Tokenizer
from torch import Tensor
from torch import nn
from torch.nn.modules import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig()
logger = logging.Logger(__name__)


# TODO: Deal with parameter redundancy/overriding
class Trainer:
    def __init__(
            self,
            model: Module,
            train_dataloader: DataLoader,
            loss_fn: _Loss,
            optimizer: Optimizer,
            device: str = 'cpu',
            test_dataloader: Optional[DataLoader] = None,
            val_dataloader: Optional[DataLoader] = None,
            scheduler: Optional[_LRScheduler] = None,
            writer: Optional[SummaryWriter] = None,
            tokenizer: Optional[Tokenizer] = None,  # TODO: Use to output examples
            save_every: Optional[int] = 1000,
            checkpoint_path: Optional[str] = './checkpoint',
            model_name: Optional[str] = 'CRD3_summarizer',
            log_level: Optional[int] = 30
    ) -> None:
        """Class that handles simple model training, validation and testing.

        Parameters
        ----------
        model: pytorch.nn.Module
            Model to train.
        train_dataloader: pytorch.utils.DataLoader
            Dataloader that supplies training data.
        loss_fn: pytorch.nn.modules.loss._Loss
            ...
        optimizer: pytorch.optim.Optimizer
            ...
        device: str, optional
            Device to use. (default = 'cpu')
        test_dataloader: pytorch.utils.DataLoader, optional
            Dataloader that supplies testing data. (default = None)
        val_dataloader: pytorch.utils.DataLoader, optional
            Dataloader that supplies validation data. (default = None)
        scheduler: torch.optim.lr_scheduler._LRScheduler, optional
            Learning rate scheduler to use. (default = None)
        """
        self._model: Module = model
        self._train_loader: DataLoader = train_dataloader
        self._optim: Optimizer = optimizer
        self._loss_fn: _Loss = loss_fn
        self._device: device = device
        self._test_loader: DataLoader = test_dataloader
        self._val_loader: DataLoader = val_dataloader
        self._scheduler: _LRScheduler = scheduler
        self._writer = writer
        self._save_every = save_every
        self._checkpoint_path = checkpoint_path
        self._model_name = model_name
        logger.setLevel(log_level)

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self._model.to(device)

    def train(
            self,
            n_step: Optional[int] = None,
            n_epoch: Optional[int] = None,
            val_every: Optional[int] = None,
            n_val: Optional[int] = None,
            continue_from: Optional[int] = None,
            grad_norm: Optional[float] = None
    ) -> None:
        """Trains the model for a specified number of steps or epochs.

        Parameters
        ----------
        n_step: int, optional
            Number of steps for which to train the model. Exactly one of n_step or n_epoch much be not None.
            (default = None)
        n_epoch: int, optional
            Number of epochs for which to train the model. Exactly one of n_step or n_epoch much be not None.
            (default = None)
        val_every: int, optional
            Number of training steps after which to run a validation loop. (default = None)
        n_val: int, optional
            Number of validation samples to iterate over for a single validation loop. (default = None)
        """
        if (n_step is None and n_epoch is None) or (n_step is not None and n_epoch is not None):
            raise ValueError('Must give only one of n_step or n_epoch')
        self._model.train()

        # step: int = 0 if continue_from is None else continue_from
        step: int = continue_from or 0
        epoch: int = 0
        while epoch < n_epoch if n_epoch is not None else True:
            epoch_loss = 0.
            epoch_start_step = step
            for source, speaker, targets, src_mask, tgt_mask in self.train_loader:
                if step >= n_step if n_step is not None else False:
                    # Stop if we've reached the step quota
                    return
                step += 1

                # move input to CUDA and run through the model
                source = source.to(self.device)
                speaker = speaker.to(self.device)
                targets = targets.to(self.device)
                src_mask = src_mask.to(self.device)
                tgt_mask = tgt_mask.to(self.device)
                step_loss = self._train_step(source, speaker, targets, src_mask, tgt_mask, grad_norm)
                epoch_loss += step_loss

                # Log train loss
                logger.info(f'Step {step} - Train loss : {step_loss}')
                if self.writer is not None:
                    self.writer.add_scalar('Train loss', step_loss, global_step=step)

                # Run a validation is necessary
                if step % val_every == 0:
                    val_loss: float = self._validate(n_val)
                    # Log val loss in tensorboard
                    logger.info(f'Step {step} - Validation loss : {val_loss}')
                    if self.writer is not None:
                        self.writer.add_scalar('Validation loss', val_loss, global_step=step)
                    self._model.train()

                # Save the model if necessary
                if step % self.save_every == 0:
                    filename = f'{self.model_name}_step-{step}.pth'
                    filepath = os.path.join(self.checkpoint_path, filename)
                    torch.save(self._model.state_dict(), filepath)
                    logger.info(f'Step {step} - Checkpoint Saved to {filepath}')

            epoch += 1
            logger.warning(f'Epoch {epoch} - Train loss: {epoch_loss / float(step - epoch_start_step):.2f}')

    def _train_step(
            self,
            source: Tensor,
            speaker: Tensor,
            targets: Tensor,
            src_mask: Tensor,
            tgt_mask: Tensor,
            grad_norm: Optional[float] = None
    ) -> float:
        """ Trains a single batch/step and return the loss value.

        Parameters
        ----------
        source: pytorch.Tensor
            Tensor of source representations.
        speaker: pytorch.Tensor
            Tensor of source representations.
        targets: pytorch.Tensor
            Tensor of target representations.
        src_mask: pytorch.Tensor
            Tensor of source padding mask.
        tgt_mask: pytorch.Tensor
            Tensor of target padding mask.

        Returns
        -------
        float
            Loss value of the training step.
        """
        self._optim.zero_grad()
        output: Tensor = self._model.forward(source, speaker, targets, src_mask, tgt_mask)
        loss: Tensor = self._loss_fn(output.view(-1, output.size(-1)))
        loss.backward()

        if grad_norm is not None:
            nn.utils.clip_grad_norm(self._model.parameters(), grad_norm)

        # TODO: Log gradients

        self._optim.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # TODO: Log sample_sentence

        return loss.item()

    def _validate(self, n_val: Optional[int] = None) -> float:
        assert self.val_loader is not None

        self._model.eval()

        with torch.no_grad():
            n_batches = float(n_val) / self.val_loader.batch_size
            step = 0
            loss = 0.
            val_iter = iter(self.val_loader)
            while step < n_batches:
                source, speaker, targets, src_mask, tgt_mask = next(val_iter)
                step += 1
                source = source.to(self.device)
                speaker = speaker.to(self.device)
                targets = targets.to(self.device)
                src_mask = src_mask.to(self.device)
                tgt_mask = tgt_mask.to(self.device)

                output = self._model.forward(source, speaker, targets, src_mask, tgt_mask)
                loss += self._loss_fn(output.view(-1, output.size(-1)))

        return loss / float(n_batches)

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def test_loader(self):
        return self._test_loader

    @property
    def val_loader(self):
        return self._val_loader

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def device(self):
        return self._device

    @property
    def writer(self):
        return self._writer

    @property
    def save_every(self):
        return self._save_every

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    @property
    def model_name(self):
        return self._model_name
