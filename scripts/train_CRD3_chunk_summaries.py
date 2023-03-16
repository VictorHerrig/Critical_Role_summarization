import sys
from argparse import ArgumentParser
from typing import Optional

from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../src/')
try:
    from loaders import CRD3Dataset
    from models import Trainer, CRD3SummarizationModel
except ImportError as e:
    print('Must be run from the scripts/ directory!')
    raise e


def main(
        device: Optional[str] = 'cpu',
        batch_size: Optional[int] = 8,
        tensorboard_logdir: Optional[str] = None
):
    model_dim = 2048
    train_dataset = CRD3Dataset('../src/loaders/train.yaml')
    val_dataset = CRD3Dataset('../src/loaders/train.yaml')
    total_steps = 100000
    warmup_steps = 8000

    model = CRD3SummarizationModel(
            vocab_size=train_dataset.vocab_size,
            speaker_size=train_dataset.speaker_vocab_size,
            model_dim=model_dim,
            pad_token_idx=train_dataset.pad_token,
            bos_token_idx=train_dataset.bos_token,
            eos_token_idx=train_dataset.eos_token,
            max_len=10000,
            max_tgt_seq_len=500,
            device=device)
    optim = Adam(model.parameters())
    writer = SummaryWriter(log_dir=tensorboard_logdir) if tensorboard_logdir is not None else None
    loss_fn = CrossEntropyLoss()
    scheduler = OneCycleLR(
        optimizer=optim,
        max_lr=model_dim ** -0.5 * warmup_steps ** -0.5,
        total_steps=total_steps,
        pct_start=float(warmup_steps) / float(total_steps)
    )

    trainer = Trainer(
        model=model,
        train_dataloader=DataLoader(train_dataset, batch_size=batch_size),
        loss_fn=loss_fn,
        optimizer=optim,
        device=device,
        val_dataloader=DataLoader(val_dataset, batch_size=batch_size),
        scheduler=scheduler,
        writer=writer,
        tokenizer=train_dataset.tokenizer,
        log_level=20
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg-file', required=True, type=str, help='Path to configuration yaml')
    args = parser.parse_args()
    cfg_file = vars(args)['cfg_file']
    main(**cfg_file)
