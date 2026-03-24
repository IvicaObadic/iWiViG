import os
import datetime
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from util import *
from models.gnn_models import init_model
from graph_inference import GSATViGReasoning

import bagnets.pytorchnet

from pytorch_lightning import Trainer
import wandb
wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def graph_training(dataset,
                   grid_approach,
                   gnn_model,
                   encoder_model=None,
                   encoder_downsample_wo_overlap=True,
                   encoder_backbone_wig_blocks=3,
                   encoder_window_size=8,
                   encoder_graph_conv="mr",
                   graph_bottleneck_layers=2,
                   gsat_r=0.7,
                   lamda_gsat_loss=1.0,
                   lambda_graph_redundancy_loss=.0,
                   lambda_gsat_weights_variance_loss=1.0,
                   sparsity_budget=None,
                   lambda_sparsity=0.1,
                   use_patch_predictions=False,
                   learn_edge_att=True,
                   run_time=None,
                   checkpoint_path=None):
    
    batch_size = 256
    if dataset == "Liveability":
        task = "regression"
        num_classes, image_size, train_data_loader, val_data_loader, test_data_loader = get_liveability_data_loaders(dataset, batch_size)
    else:
        task = "classification"
        if dataset == "sun397":
            num_classes, image_size, train_data_loader, val_data_loader, test_data_loader = getSUN397(batch_size)
        else:
            num_classes, image_size, train_data_loader, val_data_loader, test_data_loader = get_resisc45_data_loaders(dataset, batch_size)

    if gnn_model == "pvig" or encoder_model == "pvig":
        stem_channels = 48
    else:
        stem_channels = 192

    model, model_label = init_model(grid_approach,
                                    stem_channels,
                                    gnn_model,
                                    image_size,
                                    num_classes,
                                    encoder_model,
                                    encoder_downsample_wo_overlap,
                                    encoder_backbone_wig_blocks,
                                    encoder_window_size,
                                    encoder_graph_conv,
                                    graph_bottleneck_layers,
                                    gsat_r,
                                    lamda_gsat_loss,
                                    use_patch_predictions,
                                    learn_edge_att)
    
    
    model_base_output_dir = "/home/results/graph_image_understanding/{}/{}".format(dataset, model_label)
    if lambda_graph_redundancy_loss > 0:
        model_base_output_dir = os.path.join(model_base_output_dir, "w_graph_redundancy_loss")

    if lambda_gsat_weights_variance_loss > 0:
        model_base_output_dir = os.path.join(model_base_output_dir, "w_gsat_weights_variance_loss_{}".format(lambda_gsat_weights_variance_loss))
        if sparsity_budget is not None:
            model_base_output_dir = os.path.join(model_base_output_dir, "sparsity_{}".format(sparsity_budget))
    
    if use_patch_predictions:
        model_base_output_dir = os.path.join(model_base_output_dir, "with_patch_predictions")

    if run_time == None:
        run_time = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

    learning_rate = 0.001
    weight_decay = 0.05
    num_epochs = 300
    fine_tuning = ""
    if checkpoint_path is not None:
        print("Loading model from checkpoint")
        state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("model.", "")
            if k.startswith("gnn_bottleneck.gnn_"):
                k = k.replace("gnn_bottleneck.gnn_", "gnn_bottleneck.gnn_model.")
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        fine_tuning = "fine_tuned"
        learning_rate = 1e-5
        num_epochs = 200

    model_base_output_dir = os.path.join(model_base_output_dir, run_time, fine_tuning)

    graph_image_model = GSATViGReasoning(model_base_output_dir,
                                         model,
                                         num_classes,
                                         batch_size,
                                         learning_rate,
                                         weight_decay,
                                         num_epochs,
                                         lamda_gsat_loss,
                                         lambda_graph_redundancy_loss,
                                         lambda_gsat_weights_variance_loss,
                                         sparsity_budget=sparsity_budget,
                                         lambda_sparsity=lambda_sparsity)

    wandb_logger = WandbLogger(project='{}_{}'.format(gnn_model, dataset))
    wandb_logger.experiment.config["batch_size"] = batch_size
    wandb_logger.experiment.config["learning_rate"] = learning_rate
    wandb_logger.experiment.config["weight_decay"] = weight_decay
    wandb_logger.experiment.config["num_epochs"] = num_epochs
    wandb_logger.experiment.config["grid_approach"] = grid_approach
    wandb_logger.experiment.config["gnn_model"] = gnn_model
    wandb_logger.experiment.config["visual_encoder"] = encoder_model
    wandb_logger.experiment.config["num_layers"] = graph_bottleneck_layers
    wandb_logger.experiment.config["graph_conv"] = encoder_graph_conv
    wandb_logger.experiment.config["window_size"] = encoder_window_size
    wandb_logger.experiment.config["lambda_gsat_loss"] = lamda_gsat_loss
    wandb_logger.experiment.config["lambda_graph_redundancy_loss"] = lambda_graph_redundancy_loss
    wandb_logger.experiment.config["lambda_gsat_weights_variance_loss"] = lambda_gsat_weights_variance_loss
    wandb_logger.experiment.config["gsat_r"] = gsat_r

    if task == "regression":
        metric_to_monitor = "val_R2_entire_set"
        checkpoint_file = "vig_{}".format(dataset) + "-{epoch:02d}-{val_R2_entire_set:.2f}"
    else:
        metric_to_monitor = "val_accuracy"
        checkpoint_file = "vig_{}".format(dataset) + "-{epoch:02d}-{val_accuracy:.2f}"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        monitor=metric_to_monitor,
        mode="max",
        dirpath=model_base_output_dir,
        filename=checkpoint_file,
        save_last=True)

    if checkpoint_path is not None:
        checkpoint_path = os.path.join(model_base_output_dir, checkpoint_path)
        
    trainer = Trainer(max_epochs=num_epochs,
                      check_val_every_n_epoch=30,
                      callbacks=[checkpoint_callback],
                      default_root_dir=model_base_output_dir,
                      log_every_n_steps=1,
                      logger=wandb_logger)

    trainer.fit(graph_image_model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
    trainer.test(graph_image_model, dataloaders=test_data_loader, ckpt_path="best")

if __name__ == '__main__':
    graph_training("resisc45",
                   "vig_stem",
                   "iWiViG",
                   encoder_model="WIGNN",
                   graph_bottleneck_layers=3,
                   encoder_backbone_wig_blocks=2,
                   encoder_window_size=4,
                   lamda_gsat_loss=.0,
                   lambda_graph_redundancy_loss=0,
                   lambda_gsat_weights_variance_loss=0.01,
                   encoder_downsample_wo_overlap=True,
                   gsat_r=0.5,
                   sparsity_budget=0.7,
                   lambda_sparsity=0.1,
                   use_patch_predictions=False,
                   learn_edge_att=True)
