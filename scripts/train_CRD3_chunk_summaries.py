import yaml
from argparse import ArgumentParser
from typing import Optional

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from CRD3_summarization.loaders.CRD3Dataset import CRD3Dataset, CRD3BatchCollator
from CRD3_summarization.models.Trainer import Trainer
from CRD3_summarization.models.CRD3SummarizationModel import CRD3SummarizationModel

# sys.path.append('../src/CRD3_summarization')
# try:
#     from loaders.CRD3Dataset import CRD3Dataset, CRD3BatchCollator
#     from models.Trainer import Trainer
#     from models.CRD3SummarizationModel import CRD3SummarizationModel
# except ImportError as e:
#     print('Must be run from the scripts/ directory!')
#     raise e


def main(
        device: Optional[str] = 'cpu',
        batch_size: Optional[int] = 8,
        tensorboard_logdir: Optional[str] = None
):
    # TODO: Parameterize
    model_dim = 512
    window_size = 256
    total_steps = 100000
    warmup_steps = 8000
    grad_norm = 1.
    n_workers_loader = 8

    # Create dataloaders
    train_dataset = CRD3Dataset('../src/CRD3_summarization/loaders/CRD3Dataset_train.yaml')
    train_collator = CRD3BatchCollator(train_dataset.pad_token)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_collator, num_workers=n_workers_loader)
    val_dataset = CRD3Dataset('../src/CRD3_summarization/loaders/CRD3Dataset_val.yaml')
    val_collator = CRD3BatchCollator(val_dataset.pad_token)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_collator, num_workers=n_workers_loader)

    # Create model
    model = CRD3SummarizationModel(
            vocab_size=train_dataset.vocab_size,
            speaker_size=train_dataset.speaker_vocab_size,
            model_dim=model_dim,
            local_self_attn_window_size=window_size,
            pad_token_idx=train_dataset.pad_token,
            bos_token_idx=train_dataset.bos_token,
            eos_token_idx=train_dataset.eos_token,
            max_len=5000,
            max_tgt_seq_len=200,
            device=device)

    # Training bits
    optim = Adam(model.parameters())
    loss_fn = CrossEntropyLoss()
    scheduler = OneCycleLR(
        optimizer=optim,
        max_lr=model_dim ** -0.5 * warmup_steps ** -0.5,
        total_steps=total_steps,
        pct_start=float(warmup_steps) / float(total_steps)
    )

    # Tensorboard writer if desired
    writer = SummaryWriter(log_dir=tensorboard_logdir) if tensorboard_logdir is not None else None

    # And the trainer class itself
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        loss_fn=loss_fn,
        optimizer=optim,
        device=device,
        val_dataloader=val_dataloader,
        scheduler=scheduler,
        writer=writer,
        tokenizer=train_dataset.tokenizer,
        log_level=20
    )

    trainer.train(
        n_step=total_steps,
        val_every=1000,
        n_val=10,
        grad_norm=grad_norm
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg-file', required=True, type=str, help='Path to configuration yaml')
    args = parser.parse_args()
    with open(vars(args)['cfg_file'], 'r') as f:
        cfg = yaml.safe_load(f)
    main(**cfg)
