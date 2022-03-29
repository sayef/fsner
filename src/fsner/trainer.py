import argparse
import os
import warnings
from fractions import Fraction
from pathlib import Path

from pytorch_lightning import (Trainer)
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from fsner import FSNERModel, FSNERDataModule, load_dataset
from fsner import FSNERTokenizerUtils


def init_trainer_parser(parser):
    parser.add_argument('--ignore-warnings', help='Ignore warning messages', action='store_true', default=True)
    parser.add_argument('--pretrained-model', help='Pretrained model name or path',
                        default="sayef/fsner-bert-base-uncased")
    parser.add_argument('--mode', help='Run for training or evaluation', choices=['train', 'eval'], default='eval')
    parser.add_argument('--seed', help='Seed value for all random operations', default=42, type=int)
    parser.add_argument('--checkpoints-dir', help='Checkpoints directory', default="checkpoints")
    parser.add_argument('--save-dir', help='Model directory, Default: checkpoints/model', default="checkpoints/model")
    parser.add_argument('--train-data', help='Train data as json file', default="train.json")
    parser.add_argument('--val-data', help='Validation data as json file', default="val.json")
    parser.add_argument('--train-batch-size', help='Training batch size', default=6, type=int)
    parser.add_argument('--val-batch-size', help='Validation batch size', default=6, type=int)
    parser.add_argument('--n-examples-per-entity', help='Number of examples per entity', default=10, type=int)
    parser.add_argument('--neg-example-batch-ratio',
                        help='Negative example batch sampling ratio, i.e., 1/3 would mean every 3rd batch is a negative example set.',
                        default=1 / 3, type=Fraction)
    parser.add_argument('--max-epochs', help='Maximum number of training epochs', default=25, type=int)
    parser.add_argument('--device', help='cpu or gpu', default="gpu")
    parser.add_argument('--gpus', help='If device is gpu, specify gpu ids if required, Default: -1 (all)', default="-1")
    parser.add_argument('--strategy', help='If there are multiple gpus, specify multi-gpu training strategy',
                        default='ddp')

    return parser


def trainer_main(args):
    if args.ignore_warnings:  # comment out to see the warnings in the console
        warnings.filterwarnings("ignore")

    # create checkpoint directory if not exists
    Path(args.checkpoints_dir).mkdir(parents=True, exist_ok=True)

    # set seed for reproducibility of experiment results
    seed_everything(args.seed)

    # instantiate tokenizer
    tokenizer = FSNERTokenizerUtils(args.pretrained_model)

    # read from json file, convert to dict and filter out long examples
    train_data_dict = load_dataset(args.train_data if args.mode == 'train' else args.val_data, tokenizer)
    val_data_dict = load_dataset(args.val_data, tokenizer)

    # instantiate datamodule
    datamodule = FSNERDataModule(train_data_dict=train_data_dict,
                                 val_data_dict=val_data_dict,
                                 tokenizer=tokenizer,
                                 train_batch_size=args.train_batch_size,
                                 val_batch_size=args.val_batch_size,
                                 n_examples_per_entity=args.n_examples_per_entity,
                                 negative_examples_ratio=args.neg_example_batch_ratio)
    datamodule.setup("fit")

    # instantiate model
    model = FSNERModel(
        model_name_or_path=args.pretrained_model,
        epoch_steps=datamodule.epoch_steps,
        token_embeddings_size=len(tokenizer.tokenizer)
    )

    class FSNERTrainer(Trainer):
        def save_checkpoint(self, filepath, weights_only=False):
            dirpath = os.path.split(filepath)[0]
            self.lightning_module.model.save_pretrained(dirpath)
            self.datamodule.tokenizer.tokenizer.save_pretrained(dirpath)

    gpu_related_params = {"gpus": args.gpus, "strategy": args.strategy}

    # instantiate trainer
    trainer = FSNERTrainer(accelerator=args.device,
                           **gpu_related_params if args.device == "gpu" else {},
                           callbacks=[
                               ModelCheckpoint(monitor="val_loss",
                                               dirpath=Path(args.checkpoints_dir).joinpath("model"),
                                               save_top_k=1,
                                               mode="min"
                                               )
                           ],
                           enable_checkpointing=True,
                           default_root_dir=args.checkpoints_dir,
                           max_epochs=args.max_epochs,
                           logger=False,
                           num_sanity_val_steps=0,
                           reload_dataloaders_every_n_epochs=1
                           )

    if args.mode == "train":
        trainer.fit(model, datamodule)
    else:
        trainer.validate(model, datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSNER Trainer Script')
    parser = init_trainer_parser(parser)
    args = parser.parse_args()

    print("Parameters:")
    print("=" * 50)
    for k, v in vars(args).items():
        v = str(v)
        if str(k) == 'func': continue
        print(f"{k:<30}{v:>20}")
    print("=" * 50)

    trainer_main(args)
