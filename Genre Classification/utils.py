"""
Contains all the utility functions which are common among
the different models.
"""
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
import torch.utils.data as Data
from ray import train, tune
from ray.train import Checkpoint
import os
import tempfile


def dataset_split(dataset):
    """
    Splits the dataset into training/validation/test in ratio 80:10:10
    :param dataset: torch dataset being split
    """
    indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(indices)

    num_train = int(len(dataset) * 0.8)
    num_validation = int(len(dataset) * 0.1)
    train_indices = indices[:num_train]
    test_and_validation = indices[num_train:]
    validation_indices = test_and_validation[:num_validation]
    test_indices = test_and_validation[num_validation:]

    return test_indices, train_indices, validation_indices

def train_model(model, device, config, dataset):
    """
    Trains the model provided using the configuration file.
    Works with Ray Tune Hyperparameter tuning
    :param model: nn.Module
    :param device: device to run the model on
    :param config: Ray tune configuration
    :param dataset: torch dataset
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), config["lr"])

    _, train_indices, validation_indices = dataset_split(dataset)
    
    # Create test and train datasets
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    train_dataset = Data.DataLoader(dataset, batch_size=config["batch_size"], sampler=train_sampler)

    validation_dataset = Data.DataLoader(dataset, batch_size=config["batch_size"], sampler=validation_sampler)

    for epoch in range(config["num_epochs"]):
        for batch_id, curr_batch in enumerate(train_dataset):
            # Predict and get loss
            images, labels = curr_batch[0].to(device), curr_batch[1].to(device)
            pred = model(images)
            loss = loss_fn(pred, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"epoch: {epoch}, batch_id: {batch_id}, loss: {loss}")

        # Validation loss
        # Calculate avg.loss and accuracy for all datapoints in validation set.
        # Based on https://docs.ray.io/en/latest/tune/examples/tune-pytorch-cifar.html
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for data in validation_dataset:
            with torch.no_grad():
                images, labels = data
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = loss_fn(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Construct checkpoint for Ray Tuner
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )

        print(f"Validation Loss: {val_loss / val_steps}, Accuracy: {correct / total}")

def test_model(best_model, best_result, dataset, device):
    """
    Test the best configured model on the test dataset
    :param best_model: best faring nn.Module
    :param best_result: best configurations model state location
    :param dataset: torch dataset
    """
    # Get model state from best result
    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")
    model_state, _ = torch.load(checkpoint_path)
    best_model.load_state_dict(model_state)

    # Create test data loader
    test_indices, _, _ = dataset_split(dataset)
    test_sampler = SubsetRandomSampler(test_indices)
    test_dataset = Data.DataLoader(dataset, batch_size=5, sampler=test_sampler)

    total = 0
    correct = 0
    for data in test_dataset:
        with torch.no_grad():
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = best_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Best Model Accuracy {(correct * 100) / total}%")
