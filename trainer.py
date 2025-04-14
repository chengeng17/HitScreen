import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from models import binary_cross_entropy, entropy_logits
from prettytable import PrettyTable
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, args, experiment=None):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = args.max_epoch
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = args 
        self.output_dir = args.result_output_dir
        self.current_epoch = 0

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]

        train_metric_header = ["# Epoch", "Train_loss"]

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)


    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1

            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            if self.experiment:
                self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc, epoch=self.current_epoch)
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))


            model_path = os.path.join(self.output_dir, f"model_epoch_{self.current_epoch-1}.pth")
            if os.path.exists(model_path):
                os.remove(model_path)
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
 
        self.save_result()
        return self.test_metrics

    def save_result(self):
        torch.save(self.best_model.state_dict(),
                    os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        # state = {
        #     "train_epoch_loss": self.train_loss_epoch,
        #     "val_epoch_loss": self.val_loss_epoch,
        #     "test_metrics": self.test_metrics,
        #     "config": self.config
        # }

        # torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        # Convert the Namespace object to a dictionary
        config_dict = vars(self.config)

        # Prepare the string to write to the file
        config_str = "\n".join(f"{key}: {value}" for key, value in config_dict.items())

        # Specify the file path for the configuration
        config_file_path = os.path.join(self.output_dir, "config.txt")

        # Write the configuration to the file
        with open(config_file_path, 'w') as file:
            file.write(config_str)


        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())


    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d,v_d_atomic_embedding, v_p_cnn, v_p_esm, labels) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            # v_d,v_d_atomic_embedding,v_p, v_p_esm, labels = v_d.to(self.device),v_d_atomic_embedding.to(self.device), v_p.to(self.device), v_p_esm.to(self.device), labels.float().to(self.device)
            v_d = v_d.to(self.device) if v_d is not None else None
            v_d_atomic_embedding = v_d_atomic_embedding.to(self.device) if v_d_atomic_embedding is not None else None
            v_p_cnn = v_p_cnn.to(self.device) if v_p_cnn is not None else None
            v_p_esm = v_p_esm.to(self.device) if v_p_esm is not None else None
            labels = labels.float().to(self.device) if labels is not None else None
            self.optim.zero_grad()
            f, score = self.model(v_d,v_d_atomic_embedding, v_p_cnn, v_p_esm)
            n, loss = binary_cross_entropy(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch
            
    def test(self, dataloader="val"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (v_d,v_d_atomic_embedding, v_p_cnn, v_p_esm, labels) in enumerate(tqdm(data_loader)):

                # v_d,v_d_atomic_embedding,v_d_cls_embedding, v_p, labels = v_d.to(self.device),v_d_atomic_embedding.to(self.device), v_d_cls_embedding.to(self.device), v_p.to(self.device), labels.float().to(self.device)

                v_d = v_d.to(self.device) if v_d is not None else None
                v_d_atomic_embedding = v_d_atomic_embedding.to(self.device) if v_d_atomic_embedding is not None else None
                v_p_cnn = v_p_cnn.to(self.device) if v_p_cnn is not None else None
                v_p_esm = v_p_esm.to(self.device) if v_p_esm is not None else None
                labels = labels.float().to(self.device) if labels is not None else None

                f, score = self.model(v_d,v_d_atomic_embedding, v_p_cnn, v_p_esm)

                n, loss = binary_cross_entropy(score, labels)

                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches

        return auroc, auprc, test_loss


