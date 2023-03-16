import os
import glob
import torch
import argparse
from src.models.gain import GAIN
from src.data.datasets import DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def main(args):
    # Parse arguments
    alpha = args.alpha
    dataset = args.data_name
    hint_rate = args.hint_rate
    miss_rate = args.miss_rate
    batch_size = args.batch_size
    iterations = args.iterations

    # Load data
    dm = DataModule(dataset=dataset, batch_size=batch_size, prop_missing=miss_rate, normalize=True)
    dm.setup()

    # Load model
    model = GAIN(
        input_size=dm.input_size(),
        alpha=alpha,
        hint_rate=hint_rate,
    )

    print(f'''
        -------------------------- Experiment --------------------------
        Dataset: {dataset}
        Missing rate: {miss_rate}
        Batch size: {batch_size}
        Hint rate: {hint_rate}
        Alpha: {alpha}
        Iterations: {iterations}
        
        Generator: {model.generator}
        Discriminator: {model.discriminator}
        ---------------------------------------------------------------
        ''')

    # Train model
    exp_logger = TensorBoardLogger('reports/logs_experiments', name=dataset)
    trainer = Trainer(
        max_steps=iterations,
        default_root_dir='reports/logs_experiments',
        logger=exp_logger,
        accelerator='gpu',
        devices=1,
    )

    trainer.fit(model, datamodule=dm)

    trainer.test(model, datamodule=dm)

    # Save model
    files = glob.glob(f'./reports/logs_experiments/{dataset}/*')
    newest = max(files, key=os.path.getctime)
    torch.save(model.state_dict(), f'{newest}/model.pt')


if __name__ == '__main__':
    # Inputs for the experiment
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['credit', 'spam', 'letter', 'breast', 'news'],
        default='spam',
        type=str)
    parser.add_argument(
        '--miss_rate',
        help='missing data probability',
        default=0.2,
        type=float)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=100,
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training iterations',
        default=10000,
        type=int)

    args = parser.parse_args()

    main(args)
