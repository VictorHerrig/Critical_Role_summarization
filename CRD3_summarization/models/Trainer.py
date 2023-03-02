from typing import Optional

import torch
from torch import device
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
            self,
            model: Module,
            train_dataloader: DataLoader,
            loss_fn: _Loss,
            optimizer: Optimizer,
            cuda_device: int = None,
            test_dataloader: DataLoader = None,
            val_dataloader: DataLoader = None,
            scheduler: _LRScheduler = None
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
        cuda_device: int, optional
            Number of cuda device to use. (default = None)
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
        self._use_cuda: bool = cuda_device is not None and isinstance(cuda_device, int)
        self._cuda_device: device = device('cuda', cuda_device)
        self._test_loader: DataLoader = test_dataloader
        self._val_loader: DataLoader = val_dataloader
        self._scheduler: _LRScheduler = scheduler

        if self._use_cuda:
            self._model.cuda(self.cuda_device)

    def train(
            self,
            n_step: Optional[int] = None,
            n_epoch: Optional[int] = None,
            val_every: Optional[int] = None,
            val_prop: Optional[float] = None
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
        val_prop: float, optional
            Proportion of the validation set to iterate over for a single validation loop. (default = None)
        """
        if (n_step is None and n_epoch is None) or (n_step is not None and n_epoch is not None):
            raise ValueError('Must give only one of n_step or n_epoch')
        self._model.train()

        step: int = 0
        epoch: int = 0
        while epoch < n_epoch if n_epoch is not None else True:
            for data, targets in self.train_loader:
                if step >= n_step if n_step is not None else False:
                    # Stop if we've reached the step quota
                    return
                if self._use_cuda:
                    data = data.cuda(self.cuda_device)
                    targets = targets.cuda(self.cuda_device)
                step_loss = self._train_step(data, targets)
                # TODO: Log train loss in tensorboard
                step += 1
                if step % val_every == 0:
                    # Run a validation loop if we are on the right step
                    val_loss: float = self._validate(val_prop)
                    # TODO: Log val loss in tensorboard

            epoch += 1

    def _train_step(
            self,
            data: Tensor,
            targets: Tensor
    ) -> float:
        """ Trains a single batch/step and return the loss value.

        Parameters
        ----------
        data: pytorch.Tensor
            Tensor of training examples.
        targets: pytorch.Tensor
            Tensor of training labels.

        Returns
        -------
        float
            Loss value of the training step.
        """
        self._optim.zero_grad()
        output: Tensor = self._model.forward(data, targets)
        loss: Tensor = self._loss_fn(output)
        loss.backward()
        self._optim.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def _validate(self, val_prop: Optional[float] = None) -> float:
        ...

    def test(self):
        ...

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
    def cuda_device(self):
        return self._cuda_device
