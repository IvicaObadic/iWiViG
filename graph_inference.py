import os.path

import pandas as pd
import torch
from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torchmetrics.functional.regression import r2_score as torch_metrics_r2_score
from torchmetrics.functional.classification import accuracy
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, mean_absolute_error
from scipy.stats import kendalltau
from models.gsat_impl.gsat import gsat_loss
from models.gnn_models import GSATViG
from torch import nn
import torchvision
from torchvision.transforms import v2


def regression_labels_getter(batch):
   print(batch[1].shape)
    # 'batch' here would typically be (images, labels) or a dict
    # If your DataLoader returns (images, labels) tuple:
   return batch[1]  # Ensure labels are in the correct shape for regression



class GSATViGReasoning(pl.LightningModule):
    def __init__(self, model_output_dir, model, num_classes, batch_size, learning_rate, weight_decay, num_epochs,
                 lambda_gsat_loss,
                 lambda_graph_redundancy_loss,
                 lambda_gsat_weights_variance_loss,
                 sparsity_budget=None,
                 lambda_sparsity=0.1) -> None:
        super().__init__()
        self.model_output_dir = model_output_dir
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

        self.validation_step_preds = []
        self.validation_labels = []
        self.validation_ids = []
        self.lambda_gsat_loss = lambda_gsat_loss
        self.lambda_graph_redundancy_loss = lambda_graph_redundancy_loss
        self.lambda_gsat_weights_variance_loss = lambda_gsat_weights_variance_loss
        self.lambda_sparsity = lambda_sparsity
        self.sparsity_budget = sparsity_budget
        self.num_classes = num_classes

        self.task = "classification"
        if self.num_classes == 1:
            self.task_loss_fn = nn.MSELoss()
            # cutmix = v2.CutMix(num_classes=None, labels_getter=regression_labels_getter)
            # mixup = v2.MixUp(alpha=0.8, num_classes=None, labels_getter=regression_labels_getter)
            self.cutmix_or_mixup = None
            self.task = "regression"
        else:
            cutmix = v2.CutMix(num_classes=self.num_classes)
            mixup = v2.MixUp(alpha=0.8, num_classes=self.num_classes)
            self.cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
            self.task_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            print(f"DEBUG: cutmix_transform.num_classes AFTER instantiation: {cutmix.num_classes}")
            print(f"DEBUG: Type of cutmix_transform.num_classes: {type(mixup.num_classes)}")

        self.save_hyperparameters(ignore=["model"]) 
        self.automatic_optimization = False


    def _predict(self, images):
        result = self.model(images)
        raw_edge_att = None if "raw_edge_att" not in result else result["raw_edge_att"]
        edge_att = None if "edge_att" not in result else result["edge_att"]
        node_embeddings = None if "node_embeddings" not in result else result["node_embeddings"]
        node_embeddings_graph_processing = None if "node_embeddings_graph_processing" not in result else result["node_embeddings_graph_processing"]
        patch_predictions = None if "patch_predictions" not in result else result["patch_predictions"].double()
        prediction = result["prediction"]
        return edge_att, raw_edge_att,  node_embeddings, node_embeddings_graph_processing, patch_predictions, prediction.double()

    def calc_patch_pred_loss(self, labels, patch_predictions):
        batch_size, num_classes, W, H = patch_predictions.shape
        num_patches = W * H
        patch_predictions = patch_predictions.reshape((batch_size, num_classes, -1)).transpose(1, 2)
        patch_predictions = patch_predictions.reshape((batch_size * num_patches, num_classes))
        labels_rep = torch.repeat_interleave(labels, num_patches, dim=0)
        print("labels_rep.shape", labels_rep.shape)
        print("patch_predictions.shape", patch_predictions.shape)
        loss_patch_predictions = self.task_loss_fn(patch_predictions, labels_rep)
        return loss_patch_predictions
    

    def Loss_cosine(self, h_emb, eps=1e-8):
        # h_emb (B, Tokens, dims * heads)
        # normalize
        #target_h_emb = h_emb.reshape
        #hshape = target_h_emb.shape 
        #target_h_emb = target_h_emb.reshape(hshape[0], hshape[1], -1)
        print("h_emb.shape", h_emb.shape)
        a_n = h_emb.norm(dim=2).unsqueeze(2)
        a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

        # patch-wise absolute value of cosine similarity
        sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm.transpose(1,2))
        loss_cos = sim_matrix.mean()

        return loss_cos

    def Loss_contrastive(self, h1_emb, hl_emb, eps=1e-8):
    
        h1_emb_target = h1_emb
        hl_emb_target = hl_emb

        hshape = h1_emb_target.shape 
        # h1_emb_target = h1_emb_target.reshape(hshape[0], hshape[1], -1).detach()
        h1_emb_target = h1_emb_target.reshape(hshape[0], hshape[1], -1)
        h1_n = h1_emb_target.norm(dim=2).unsqueeze(2)
        h1_norm = h1_emb_target/torch.max(h1_n, eps*torch.ones_like(h1_n))

        hl_emb_target = hl_emb_target.reshape(hshape[0], hshape[1], -1)
        hl_n = hl_emb_target.norm(dim=2).unsqueeze(2)
        hl_norm = hl_emb_target/torch.max(hl_n, eps*torch.ones_like(hl_n))

        sim_matrix = torch.einsum('abc,adc->abd', h1_norm, hl_norm)
        sim_diag = torch.diagonal(sim_matrix, dim1=1, dim2=2)
        dim2 = sim_diag.shape[1]
        exp_sim_diag = torch.exp(sim_diag)
        temp_sim = torch.sum(sim_matrix, dim=2)
        temp_sim = torch.exp((temp_sim-sim_diag)/(dim2-1))
        nce = -torch.log(exp_sim_diag/(exp_sim_diag+temp_sim))
        return nce.mean()


    def compute_and_log_loss(self, predictions, labels, raw_edge_att, edge_att, node_embeddings, node_embeddings_graph_processing, patch_predictions, split):
        loss_label = "cross_entropy_loss"
        if self.task == "regression":
            loss_label = "mse_loss"
            predictions = predictions.flatten()
            
        loss_variance_edge_att = torch.tensor(0.0, device=self.device)
        loss_patch_predictions = torch.tensor(0.0, device=self.device)
        loss_graph_redundancy = torch.tensor(0.0, device=self.device)
        
        if node_embeddings is not None and node_embeddings_graph_processing is not None:
            loss_graph_redundancy = self.graph_redundancy_loss(node_embeddings, node_embeddings_graph_processing, split=split)
        if patch_predictions is not None:
            loss_patch_predictions = self.calc_patch_pred_loss(labels, patch_predictions)
            self.log("{}_patch_predictions_loss".format(split), loss_patch_predictions.item(), on_epoch=True, on_step=False)
        if edge_att is not None:
            edge_att_loss_fn = raw_edge_att.sigmoid()
            task_loss, loss_dict = gsat_loss(self.model, self.task_loss_fn, edge_att_loss_fn, predictions, labels, self.current_epoch, self.batch_size, self.lambda_gsat_loss)
            self.log("{}_{}".format(split, loss_label), loss_dict["pred"], on_epoch=True, on_step=False)
            self.log("{}_info_loss".format(split), loss_dict["info"], on_epoch=True, on_step=False)
            if self.lambda_gsat_weights_variance_loss > 0:
                gini_impurity = torch.mean(edge_att_loss_fn * (1 - edge_att_loss_fn))
                sparsity_loss = torch.tensor(0.0, device=self.device)
                if self.sparsity_budget is not None:
                    edge_sparsity = torch.mean(edge_att_loss_fn)
                    sparsity_loss = self.lambda_sparsity * (edge_sparsity - self.sparsity_budget)**2

                # print("edge_att.shape", edge_att.shape)
                # edge_att = edge_att.reshape(predictions.shape[0], -1)
                # variance_edge_att = edge_att.var(dim=1)
                # batch_variance = variance_edge_att.mean()
                self.log("{}_gini_impurity".format(split), gini_impurity.item(), on_epoch=True, on_step=False)
                loss_variance_edge_att = (self.lambda_gsat_weights_variance_loss * gini_impurity) + sparsity_loss
        else:
            task_loss = self.task_loss_fn(predictions, labels)
            self.log("{}_{}".format(split, loss_label), task_loss, on_epoch=True, on_step=False)
        
        total_loss = task_loss + loss_graph_redundancy + loss_variance_edge_att - loss_patch_predictions
        self.log("{}_total_loss".format(split), total_loss, on_epoch=True, on_step=False)
        return total_loss
    
    def graph_redundancy_loss(self, node_embeddings, node_embeddings_graph_processing, split="test"):
        # node_embeddings = node_embeddings.reshape((node_embeddings.shape[0], node_embeddings.shape[1], -1))
        # node_embeddings_1 = node_embeddings.unsqueeze(3)
        # node_embeddings_2 = node_embeddings.unsqueeze(2)
        # cosine_sim = torch.nn.functional.cosine_similarity(node_embeddings_1, node_embeddings_2, dim=1)
        # graph_redundancy_loss = torch.mean(cosine_sim.flatten())
        loss_cosine = self.Loss_cosine(node_embeddings_graph_processing)
        self.log("{}_within_emb_redundancy_loss".format(split), loss_cosine.item(), on_epoch=True, on_step=False)

        # cross_embedding_loss = self.Loss_contrastive(node_embeddings, node_embeddings_graph_processing)
        # self.log("{}_cross_emb_redundancy_loss".format(split), cross_embedding_loss.item(), on_epoch=True, on_step=False)

        return self.lambda_graph_redundancy_loss * loss_cosine


    def training_step(self, data, batch_idx) -> STEP_OUTPUT:
        with torch.autograd.set_detect_anomaly(True):

            optimizers = self.optimizers()
            schedulers = self.lr_schedulers()

            images = data["image"]
            train_labels = data["label"]
            train_labels_cut_mix = None
            if self.cutmix_or_mixup is not None:
                images, train_labels_cut_mix = self.cutmix_or_mixup(images, train_labels)

            if self.model.use_patch_predictions:
                adversarial_loss_optimizer = optimizers[1]
                adversarial_loss_optimizer.zero_grad()
                edge_att, raw_edge_att, node_embeddings, node_embeddings_graph_processing, patch_predictions, predictions = self._predict(images)
                loss_patch_predictions = self.calc_patch_pred_loss(train_labels_cut_mix if train_labels_cut_mix is not None else train_labels, patch_predictions)
                
                adversarial_loss_scheduler = schedulers[1]
                self.manual_backward(loss_patch_predictions)
                adversarial_loss_optimizer.step()
                adversarial_loss_scheduler.step()
                
                prediction_loss_optimizer = optimizers[0]
                prediction_loss_scheduler = schedulers[0]
            else:
                prediction_loss_optimizer = optimizers
                prediction_loss_scheduler = schedulers

            prediction_loss_optimizer.zero_grad()
            edge_att, raw_edge_att, node_embeddings, node_embeddings_graph_processing, patch_predictions, predictions = self._predict(images)

            total_loss = self.compute_and_log_loss(predictions,
                                            train_labels_cut_mix if train_labels_cut_mix is not None else train_labels,
                                            raw_edge_att,
                                            edge_att,
                                            node_embeddings,
                                            node_embeddings_graph_processing,
                                            patch_predictions,
                                            "train")

            self.manual_backward(total_loss)
            prediction_loss_optimizer.step()
            prediction_loss_scheduler.step()
                
            with torch.no_grad():
                if self.task == "regression":
                    r2_train = torch_metrics_r2_score(predictions.flatten(), train_labels)
                    self.log("train_r2", r2_train, on_epoch=True, on_step=False)
                else:
                    train_accuracy = accuracy(predictions, train_labels, task="multiclass", num_classes=self.num_classes)
                    self.log("train_accuracy", train_accuracy, on_epoch=True, on_step=False)


    def validation_step(self, data, batch_idx) -> Optional[STEP_OUTPUT]:
        edge_att, raw_edge_att,  node_embeddings, node_embeddings_graph_processing, val_patch_predictions, val_preds = self._predict(data["image"])

        val_labels = data["label"]
        loss = self.compute_and_log_loss(val_preds, val_labels, raw_edge_att, edge_att, node_embeddings, node_embeddings_graph_processing, val_patch_predictions, "val")

        if self.task != "regression":
            val_preds = val_preds.argmax(dim=-1)
        else:
            val_preds = val_preds.flatten()

        val_preds_np = val_preds.detach().cpu().numpy()
        labels_np = val_labels.detach().cpu().numpy()

        if "ids" in data:
            self.validation_ids.extend(data["ids"].detach().cpu().numpy().tolist())

        self.validation_step_preds.extend(val_preds_np.tolist())
        self.validation_labels.extend(labels_np.tolist())

        return loss

    def save_predictions(self, split="val"):

        if not os.path.exists(self.model_output_dir):
            os.makedirs(self.model_output_dir)

        prediction_data = {"labels": self.validation_labels, "preds": self.validation_step_preds}
        if len(self.validation_ids) > 0:
            prediction_data["ids"] = self.validation_ids
        prediction_data = pd.DataFrame(prediction_data)
        prediction_data.to_csv(os.path.join(self.model_output_dir, "predictions_{}_{}.csv".format(split, self.current_epoch)))

        if self.task == "regression":
            mse_val = mean_squared_error(self.validation_labels, self.validation_step_preds)
            r2_val = r2_score(self.validation_labels, self.validation_step_preds)
            tau_val = kendalltau(self.validation_step_preds, self.validation_labels).statistic

            self.log("{}_R2_entire_set".format(split), r2_val, on_epoch=True, on_step=False)
            self.log("{}_MSE_entire_set".format(split), mse_val, on_epoch=True, on_step=False)
            self.log("{}_tau_entire_set".format(split), tau_val, on_epoch=True, on_step=False)

            metrics_scores = {
                "mse": [mse_val],
                "r2": [r2_val],
                "tau": [tau_val]}
        else:
            accuracy_val = accuracy_score(self.validation_labels, self.validation_step_preds)
            f1_score_val = f1_score(self.validation_labels, self.validation_step_preds, average="macro")
            self.log("{}_accuracy".format(split), accuracy_val, on_epoch=True, on_step=False)
            self.log("{}_f1_score".format(split), f1_score_val, on_epoch=True, on_step=False)

            metrics_scores = {
                "accuracy": [accuracy_val],
                "f1_score": [f1_score_val]}

        metrics_scores_df = pd.DataFrame.from_dict(metrics_scores)
        metrics_scores_df.to_csv(os.path.join(self.model_output_dir, "metrics_{}_{}.csv".format(split, self.current_epoch)))

        self.validation_ids.clear()
        self.validation_step_preds.clear()
        self.validation_labels.clear()

    def on_validation_epoch_end(self) -> None:
        self.save_predictions()

    def test_step(self, data, batch_idx) -> Optional[STEP_OUTPUT]:
        edge_att, raw_edge_att, node_embeddings, node_embeddings_graph_processing, test_patch_predictions, test_preds = self._predict(data["image"])
        test_labels = data["label"]
        loss = self.compute_and_log_loss(test_preds, test_labels, raw_edge_att, edge_att, node_embeddings, node_embeddings_graph_processing, test_patch_predictions, "test")

        if self.task != "regression":
            test_preds = test_preds.argmax(dim=-1)
        else:
            test_preds = test_preds.flatten()

        test_preds_np = test_preds.detach().cpu().numpy()
        labels_np = test_labels.detach().cpu().numpy()

        if "ids" in data:
            self.validation_ids.extend(data["ids"].detach().cpu().numpy().tolist())
        self.validation_step_preds.extend(test_preds_np.tolist())
        self.validation_labels.extend(labels_np.tolist())

        return loss

    def on_test_epoch_end(self) -> None:
        self.save_predictions(split="test")

    def configure_optimizers(self):
        params_prediction_model = [p for name, p in self.model.named_parameters() if 'patch_prediction' not in name]
        params_adversarial_model = [p for name, p in self.model.named_parameters() if 'patch_prediction' in name]
        optimizer_prediction_model = torch.optim.AdamW(params_prediction_model, lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizers = [optimizer_prediction_model]
        scheduler_prediction_model = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_prediction_model, T_max=self.num_epochs, eta_min=0, last_epoch=-1)
        schedulers = [scheduler_prediction_model]
        if self.model.use_patch_predictions:
            optimizer_adversarial_model = torch.optim.AdamW(params_adversarial_model, lr=1e-4, weight_decay=self.weight_decay)
            scheduler_adversarial_model = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adversarial_model, T_max=self.num_epochs, eta_min=0, last_epoch=-1)
            optimizers.append(optimizer_adversarial_model)
            schedulers.append(scheduler_adversarial_model)

        return optimizers, schedulers