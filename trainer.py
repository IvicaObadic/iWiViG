import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay
import torch.cuda
import copy

class ModelTrainer():
    def __init__(self,
                 model,
                 model_output_dir,
                 train_loader,
                 test_loader,
                 optimizer,
                 loss_fn,
                 epochs):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.model_output_dir = model_output_dir
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        self.epoch_loss = []
        self.test_f1_score = []
        self.best_model = None

    def batch_predict(self, model, data):
        if torch.cuda.is_available():
            data = data.cuda()
        node_embeddings = data.x
        edges = data.edge_index
        batch = data.batch
        output = model(node_embeddings, edges, batch)
        return output

    def train_epoch(self, epoch):
        self.model.train()
        batch_losses = []
        for data in self.train_loader:
            self.optimizer.zero_grad()  # Clear gradients.
            predictions = self.batch_predict(self.model, data)
            loss = self.loss_fn(predictions, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            #print(self.model.conv_message_passing.lin.weight.grad)
            self.optimizer.step()  # Update parameters based on gradients.
            batch_losses.append(loss.detach().cpu().item())

        epoch_loss = np.array(batch_losses).mean()
        print("Epoch: {}, loss: {}".format(epoch, epoch_loss))
        self.epoch_loss.append(epoch_loss)

    def test(self, epoch, model):
        model.eval()
        ids = []
        gt = []
        predicted = []
        for data in self.test_loader:
            predictions = self.batch_predict(model, data)
            pred = predictions.argmax(dim=1)
            ids.extend(data.id)
            gt.extend(data.y.detach().cpu())
            predicted.extend(pred.detach().cpu())

        if epoch < self.epochs:
            test_f1_score = f1_score(gt, predicted)
            print("Test F1 score: {}".format(test_f1_score))
            self.test_f1_score.append(test_f1_score)

            if test_f1_score >= max(self.test_f1_score):
                self.best_model = copy.deepcopy(model)

        return ids, gt, predicted


    def fit(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            self.test(epoch, self.model)

        print("Evaluating the best model:")
        ids, gt, predicted = self.test(self.epochs, self.best_model)
        print("F1 score: {}".format(f1_score(gt, predicted)))
        self.save_results(ids, gt, predicted)


    def save_results(self, ids, gt, predicted):
        training_stats = {"Epoch": [i for i in range(1, self.epochs+1)],
                          "train_loss": self.epoch_loss,
                          "test_f1_score": self.test_f1_score}
        training_stats = pd.DataFrame(training_stats)
        training_stats.to_csv(os.path.join(self.model_output_dir, "training_stats.csv"))

        gt_vs_predicted = pd.DataFrame({
            "id": ids,
            "gt": gt,
            "predicted": predicted})
        gt_vs_predicted.to_csv(os.path.join(self.model_output_dir, "gt_vs_predicted.csv"))

        conf_matr = confusion_matrix(gt, predicted)
        pd.DataFrame(conf_matr).to_csv(os.path.join(self.model_output_dir, "conf_matr.csv"))
        torch.save({'epoch': self.epochs,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': self.epoch_loss[-1]},
                   os.path.join(self.model_output_dir, "model.pth"))

        torch.save(self.best_model.state_dict(), os.path.join(self.model_output_dir, "best_model.pth"))