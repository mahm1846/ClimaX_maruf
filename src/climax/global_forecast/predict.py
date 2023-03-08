# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from climax.global_forecast.datamodule import GlobalForecastDataModule
from climax.global_forecast.module import GlobalForecastModule
from pytorch_lightning.cli import LightningCLI


def main():

    import warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*save_config_overwrite*")
    warnings.filterwarnings("ignore", ".*LightningCLI.auto_registry*")
    warnings.filterwarnings("ignore", ".*Trainer already configured with model summary callbacks*")
    warnings.filterwarnings("ignore", ".*Lightning couldn't infer*")

    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = LightningCLI(
        model_class=GlobalForecastModule,
        datamodule_class=GlobalForecastDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    os.makedirs(cli.trainer.default_root_dir, exist_ok=True)

    normalization = cli.datamodule.output_transforms
    mean_norm, std_norm = normalization.mean, normalization.std
    mean_denorm, std_denorm = -mean_norm / std_norm, 1 / std_norm
    cli.model.set_denormalization(mean_denorm, std_denorm)
    cli.model.set_lat_lon(*cli.datamodule.get_lat_lon())
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)
    #cli.model.set_val_clim(cli.datamodule.val_clim)
    #cli.model.set_test_clim(cli.datamodule.test_clim)
    cli.model.set_pred_clim(cli.datamodule.pred_clim)

    # fit() runs the training
    #cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    # test the trained model
    #cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")

    # pred
    #x, y, predictions = 
    pred = cli.trainer.predict(cli.model, datamodule=cli.datamodule, return_predictions=True)
    path='/scratch/fp0/shared/climax_train_global/pred'
    import torch
    torch.save(pred[0][0], path+ '/x.pt')
    torch.save(pred[0][1], path+ '/y.pt')
    torch.save(pred[0][2], path+ '/preds.pt')
    ####################################



if __name__ == "__main__":
    main()
