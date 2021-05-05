import time
import torch
import torch.nn as nn
import numpy as np


class GraphNetworkTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=self.config.learning_rate)

    # Define model evaluation function
    def evaluate(self, features, labels, mask):
        t_before = time.time()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            t_mask = torch.from_numpy(np.array(mask * 1., dtype=np.float32))
            tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1,
                                      0).repeat(1, labels.shape[1])
            loss = self.criterion(logits * tm_mask, torch.max(labels, 1)[1])
            pred = torch.max(logits, 1)[1]
            acc = ((pred == torch.max(labels, 1)[1]).float() *
                   t_mask).sum().item() / t_mask.sum().item()

        return (loss.numpy(), acc, pred.numpy(), labels.numpy(),
                (time.time() - t_before))

    def train(self, features, y_train, train_mask, t_y_val, val_mask):
        val_losses = []
        tm_train_mask = torch.transpose(torch.unsqueeze(train_mask, 0), 1,
                                        0).repeat(1, y_train.shape[1])

        # Train model
        for epoch in range(self.config.epochs):

            t = time.time()

            # Forward pass
            logits = self.model(features)
            loss = self.criterion(logits * tm_train_mask,
                                  torch.max(y_train, 1)[1])
            acc = (
                (torch.max(logits, 1)[1] == torch.max(y_train, 1)[1]).float() *
                train_mask).sum().item() / train_mask.sum().item()

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Validation
            val_loss, val_acc, pred, labels, duration = self.evaluate(
                features, t_y_val, val_mask)
            val_losses.append(val_loss)

            print("Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}"\
                        .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

            if epoch > self.config.early_stopping and val_losses[-1] > np.mean(
                    val_losses[-(self.config.early_stopping + 1):-1]):
                print("Early stopping...")
                return
        print("Optimization Finished!")
