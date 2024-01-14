import yaml
from argparse import ArgumentParser
from typing import Optional

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from CRD3_summarization.loaders.CRD3Dataset import BaseCRD3Dataset, CRD3BatchCollator
from CRD3_summarization.models.Trainer import Trainer
from CRD3_summarization.models.CRD3SummarizationModel import CRD3SummarizationModel


def main(
        device: Optional[str] = 'cpu',
        batch_size: Optional[int] = 8,
        tensorboard_logdir: Optional[str] = None
):
    # TODO: Parameterize
    # Use BART base size
    model_dim = 768  #  512
    feedforward_dim = 3072
    num_local_self_attn = 4
    num_segment_full_self_attn = 2
    num_top_down_blocks = 2
    num_decoder_layers = 6  # 12
    window_size = 256
    total_steps = 100000
    warmup_steps = 8000
    grad_norm = 5.
    n_workers_loader = 4  # 8
    val_every = 25
    n_val = 50
    save_every = 10000
    example_every = 50
    bart_model_path = '../../code/models/bart.base/model.pt'

    # Create dataloaders
    train_dataset = BaseCRD3Dataset('../src/CRD3_summarization/loaders/CRD3Dataset_train.yaml')
    train_collator = CRD3BatchCollator(train_dataset.pad_token_id, window_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_collator, num_workers=n_workers_loader)
    val_dataset = BaseCRD3Dataset('../src/CRD3_summarization/loaders/CRD3Dataset_val.yaml')
    val_collator = CRD3BatchCollator(val_dataset.pad_token_id, window_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_collator, num_workers=n_workers_loader)

    # Create model
    model = CRD3SummarizationModel(
        vocab_size=train_dataset.vocab_size,
        speaker_size=train_dataset.speaker_vocab_size,
        model_dim=model_dim,
        local_self_attn_window_size=window_size,
        pad_token_idx=train_dataset.pad_token_id,
        bos_token_idx=train_dataset.bos_token_id,
        eos_token_idx=train_dataset.eos_token_id,
        feedforward_dim=feedforward_dim,
        num_local_self_attn=num_local_self_attn,
        num_segment_full_self_attn=num_segment_full_self_attn,
        num_top_down_blocks=num_top_down_blocks,
        num_decoder_layers=num_decoder_layers,
        max_len=4096,
        max_tgt_seq_len=200,
        device=device,
        initialize_from_bart=bart_model_path
    )

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
        log_level=20,
        save_every=save_every
    )

    trainer.train(
        n_step=total_steps,
        val_every=val_every,
        n_val=n_val,
        grad_norm=grad_norm,
        example_every=example_every
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cfg-file', required=True, type=str, help='Path to configuration yaml')
    args = parser.parse_args()
    with open(vars(args)['cfg_file'], 'r') as f:
        cfg = yaml.safe_load(f)
    main(**cfg)
