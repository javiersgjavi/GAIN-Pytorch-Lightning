import numpy as np
import pandas as pd
import dataframe_image as dfi
from src.models.gain import GAIN
from src.data.datasets import DataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    results = pd.DataFrame(columns=['Mean', 'Std'])
    for dataset in ['breast', 'spam', 'letter', 'credit', 'news']:
        res_dataset = []
        for iteration in range(10):
            dm = DataModule(dataset=dataset, batch_size=128, prop_missing=0.2, normalize=True)
            dm.setup()

            model = GAIN(
                input_size=dm.input_size(),
                alpha=100,
                hint_rate=0.9,
            )

            exp_logger = TensorBoardLogger('reports/replication_table1', name=dataset)
            trainer = Trainer(
                max_steps=10000,
                default_root_dir='reports/replication_table1',
                logger=exp_logger,
                accelerator='gpu',
                devices=1,
            )

            trainer.fit(model, datamodule=dm)

            res = trainer.test(model, datamodule=dm)[0]
            res_dataset.append(res['rmse'])

        results.loc[dataset] = [np.round(np.mean(res_dataset), 4), np.round(np.std(res_dataset), 4)]

    print(results)
    dfi.export(results, 'reports/replication_table1/results.png', table_conversion='matplotlib')


if __name__ == '__main__':
    main()
