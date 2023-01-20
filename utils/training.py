import abc
import os
import sys
import tqdm
import torch

from torch.utils.data import DataLoader
from typing import Callable, Any
from pathlib import Path

import sys
from utils.train_results import BatchResult, EpochResult, FitResult


class TorchTrainer():
    """
    A class for training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, loss_fn, optimizer, device='cpu'):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_val: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, post_epoch_fn=None, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_val: Dataloader for the validation set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        best_acc = -1
        epochs_without_improvement = 0

        checkpoint_filename = None
        if checkpoints is not None:
            checkpoint_filename = f'{checkpoints}.pt'
            Path(os.path.dirname(checkpoint_filename)).mkdir(exist_ok=True)
            if os.path.isfile(checkpoint_filename):
                print(f'*** Loading checkpoint file {checkpoint_filename}')
                saved_state = torch.load(checkpoint_filename,
                                         map_location=self.device)
                best_acc = saved_state.get('best_acc', best_acc)
                epochs_without_improvement =\
                    saved_state.get('ewi', epochs_without_improvement)
                self.model.load_state_dict(saved_state['model_state'])

        for epoch in range(num_epochs):
            save_checkpoint = False
            verbose = False  # pass this to train/val_epoch.
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            # TODO: Train & evaluate for one epoch
            # - Use the train/test_epoch methods (showing below)! .
            # - Save losses and accuracies in the lists above 
            # (use append for train_acc and test_acc and extend for train_loss and test_loss).
            # - Implement early stopping. This is a very useful and
            #   simple regularization technique that is highly recommended.
            # ====== YOUR CODE: ======
            train_epoch_results:EpochResult = self.train_epoch(dl_train)
            val_epoch_results:EpochResult = self.test_epoch(dl_val)
            
            batches_in_epoch = len(val_epoch_results.losses)

            avg_prev_loss = sum(val_loss[-batches_in_epoch:])/batches_in_epoch if epoch>0 else 1e8

            best_acc = max(val_acc+[best_acc])

            train_loss.extend(train_epoch_results.losses)
            train_acc.append(train_epoch_results.accuracy)
            val_loss.extend(val_epoch_results.losses)
            val_acc.append(val_epoch_results.accuracy)

            avg_current_loss = sum(val_loss[-batches_in_epoch:])/batches_in_epoch

            #self.sched.step(avg_current_loss)

            if epoch>0 and avg_prev_loss + kw["tol"] <= avg_current_loss:
              epochs_without_improvement = epochs_without_improvement+1
            else:
              epochs_without_improvement = 0 
            if epochs_without_improvement >= early_stopping:
              break

            if val_epoch_results.accuracy > best_acc:
                save_checkpoint=True

            print("epoches without improvement:", epochs_without_improvement)
            # ========================

            # Save model checkpoint if requested
            if save_checkpoint and checkpoint_filename is not None:
                saved_state = dict(best_acc=best_acc,
                                   ewi=epochs_without_improvement,
                                   model_state=self.model.state_dict())
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at epoch {epoch+1}')

            if post_epoch_fn:
                post_epoch_fn(epoch, train_epoch_results, val_epoch_results, verbose)

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, val_loss, val_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test/validation set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)
    
    def train_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        # TODO: Train the PyTorch model on one batch of data.
        # - zero grad the optimizer (self.optimizer)
        # - Forward pass of the model 
        # - Calculate the loss (use self.loss_fn)
        # - Apply loss.backward()
        # - Apply self.optimizer.step()
        # - Calculate number of correct predictions
        # - Return loss as a number (not a tensor) - see funciton .item()
        # ====== YOUR CODE: ======
        self.optimizer.zero_grad()
        pred = self.model(X)
        loss = self.loss_fn(pred,y)
        loss.backward()
        self.optimizer.step()
        num_correct = int(torch.sum((torch.argmax(pred,dim=1)==y).to(torch.int)))
        loss = loss.item()
        # ========================

        return BatchResult(loss, num_correct)

    def test_batch(self, batch) -> BatchResult:
        X, y = batch
        if self.device:
            X = X.to(self.device)
            y = y.to(self.device)

        with torch.no_grad():
            # TODO: Evaluate the PyTorch model on one batch of data.
            # - Forward pass of the model 
            # - Calculate the loss (use self.loss_fn)
            # - Calculate number of correct predictions
            # - Return loss as a number (not a tensor) - see funciton .item()
            # ====== YOUR CODE: ======
            pred = self.model(X)
            loss = self.loss_fn(pred,y)
            num_correct = int(torch.sum((torch.argmax(pred,dim=1)==y).to(torch.int)))
            loss = loss.item()
            # ========================
        return BatchResult(loss, num_correct)

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, accuracy=accuracy)

