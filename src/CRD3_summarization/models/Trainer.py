""""""
import logging
import os
from collections import defaultdict
from typing import Optional, Tuple

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
# TODO: Continue from ...
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
            grad_norm: Optional[float] = None,
            grad_every: Optional[int] = 16,
            example_every: Optional[int] = None
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
        #example_every = 1  # TODO: FIX
        # print([k for k, _ in self._model.named_parameters()])

        # step: int = 0 if continue_from is None else continue_from
        step: int = continue_from or 0
        grad_step = continue_from or 0
        grad_step_loss = 0
        epoch: int = 0
        while epoch < n_epoch if n_epoch is not None else True:
            epoch_loss = 0.
            epoch_start_step = step
            for source, speaker, targets, src_mask, tgt_mask in self.train_loader:
                if grad_step >= n_step if n_step is not None else False:
                    # Stop if we've reached the step quota
                    return
                step += 1

                # move input to CUDA and run through the model
                source = source.to(self.device)
                speaker = speaker.to(self.device)
                targets = targets.to(self.device)
                src_mask = src_mask.to(self.device) if src_mask is not None else None
                tgt_mask = tgt_mask.to(self.device) if tgt_mask is not None else None
                step_loss, step_output = self._train_step(source, speaker, targets, src_mask, tgt_mask, grad_norm)
                grad_step_loss += step_loss
                epoch_loss += step_loss

                if step % grad_every == 0:
                    grad_step += 1
                    if grad_norm is not None:
                        nn.utils.clip_grad_norm_(self._model.parameters(), grad_norm)

                    # Log train loss and gradients
                    logger.info(f'Step {grad_step} - Train loss : {grad_step_loss}')
                    if self.writer is not None:
                        self.writer.add_scalar('Train loss', grad_step_loss, global_step=grad_step)

                        grad_dict = defaultdict(lambda: 0.)
                        for param_name, val in self._model.named_parameters():
                            layer_name = None
                            if 'decoder.layers' in param_name:
                                layer_name = f'decoder.layer.{param_name.split("decoder.layers.")[1].split(".")[0]}'
                            elif 'decoder_linear' in param_name:
                                layer_name = 'decoder_linear'
                            elif 'encoder' in param_name:
                                layer_name = f'encoder.{param_name.split("._encoder._")[1].split(".")[0]}'
                            elif 'speaker_linear' in param_name:
                                layer_name = 'speaker_linear'
                            elif 'embedding' in param_name:
                                layer_name = 'embedding'
                            if layer_name is not None:
                                grad_dict[layer_name] += val.grad.sum().item()

                        self.writer.add_scalars('gradients', grad_dict, global_step=grad_step)

                    self._optim.step()

                    if self.scheduler is not None:
                        self.scheduler.step()

                    if grad_step % example_every == 0:
                        eg_input = self.train_loader.dataset.construct_string(source.to('cpu')[:, 0, :].argmax(1))
                        self.writer.add_text('Example input', eg_input, global_step=grad_step)
                        eg_speaker = self.train_loader.dataset.construct_speaker_string(speaker.to('cpu')[:, 0, :].argmax(1))
                        self.writer.add_text('Example speakers', eg_speaker, global_step=grad_step)
                        eg_target = self.train_loader.dataset.construct_string(targets.to('cpu')[:, 0, :].argmax(1))
                        self.writer.add_text('Example target output', eg_target, global_step=grad_step)
                        eg_summ = self.train_loader.dataset.construct_string(step_output.to('cpu')[:, 0, :].argmax(1))
                        self.writer.add_text('Example model output', eg_summ, global_step=grad_step)
                        self.writer.add_text('Example target mask', f"[{','.join([str(int(b)) for b in tgt_mask[0, :].to('cpu').tolist()])}]", global_step=grad_step)

                    # Run a validation is necessary
                    if grad_step % val_every == 0:
                        val_loss: float = self._validate(n_val)
                        # Log val loss in tensorboard
                        logger.info(f'Step {grad_step} - Validation loss : {val_loss}')
                        if self.writer is not None:
                            self.writer.add_scalar('Validation loss', val_loss, global_step=grad_step)
                        self._model.train()

                    # Save the model if necessary
                    if grad_step % self.save_every == 0:
                        filename = f'{self.model_name}_step-{grad_step}.pth'
                        filepath = os.path.join(self.checkpoint_path, filename)
                        torch.save(self._model.state_dict(), filepath)
                        logger.info(f'Step {grad_step} - Checkpoint Saved to {filepath}')
                    grad_step_loss = 0

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
    ) -> Tuple[float, Tensor]:
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
        loss: Tensor = self._loss_fn(output.view(-1, output.size(-1)), targets.view(-1, targets.size(-1)))
        loss.backward()

        return loss.item(), output

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
                src_mask = src_mask.to(self.device) if src_mask is not None else None
                tgt_mask = tgt_mask.to(self.device) if tgt_mask is not None else None

                output = self._model.forward(source, speaker, targets, src_mask, tgt_mask)
                loss += self._loss_fn(output.view(-1, output.size(-1)), targets.view(-1, targets.size(-1)))

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
