#!/usr/bin/env python

from argparse import ArgumentParser
from utils import get_mnist_data_loaders, DataLoaderProgress
from fastprogress.fastprogress import master_bar, progress_bar
import torch

"""
Assignment #6:

A convolutional neural network implemented with sequential:
    1. First convolutional layer
    2. ReLU function
    3. Second convolutional layer
    4. ReLU function
    5. MaxPool2D layer
    6. Flatten
    7. Linear layer
    8. ReLU
    9. Linear output layer

    Epoch  1/10: accuracy=97.55% and loss=0.073                            
    Epoch  2/10: accuracy=98.82% and loss=0.035                            
    Epoch  3/10: accuracy=98.88% and loss=0.036                            
    Epoch  4/10: accuracy=98.95% and loss=0.031                            
    Epoch  5/10: accuracy=99.10% and loss=0.029                            
    Epoch  6/10: accuracy=98.85% and loss=0.034                            
    Epoch  7/10: accuracy=99.06% and loss=0.030                            
    Epoch  8/10: accuracy=98.93% and loss=0.036                            
    Epoch  9/10: accuracy=99.06% and loss=0.032                            
    Epoch 10/10: accuracy=99.08% and loss=0.030
"""

# check


def train_one_epoch(dataloader, model, criterion, optimizer, device, mb):

    # Put the model into training mode
    model.train()

    # Loop over the data using the progress_bar utility
    for _, (X, Y) in progress_bar(DataLoaderProgress(dataloader), parent=mb):
        X, Y = X.to(device), Y.to(device)

        # Compute model output and then loss
        output = model(X)
        loss = criterion(output, Y)
        # - zero-out gradients
        optimizer.zero_grad()
        # - compute new gradients
        loss.backward()
        # - update paramaters
        optimizer.step()


def validate(dataloader, model, criterion, device, epoch, num_epochs, mb):

    # Put the model into validation/evaluation mode
    model.eval()

    N = len(dataloader.dataset)
    num_batches = len(dataloader)

    loss, num_correct = 0, 0

    # Tell pytorch to stop updating gradients when executing the following
    with torch.no_grad():

        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)

            # Compute the model output
            output = model(X)

            # TODO:
            # - compute loss
            loss += criterion(output, Y).item()

            # - compute the number of correctly classified examples
            num_correct += (output.argmax(1) ==
                            Y).type(torch.float).sum().item()

        loss /= num_batches
        accuracy = num_correct / N

    message = "Initial" if epoch == 0 else f"Epoch {epoch:>2}/{num_epochs}:"
    message += f" accuracy={100*accuracy:5.2f}%"
    message += f" and loss={loss:.3f}"
    mb.write(message)


def train(model, criterion, optimizer, train_loader, valid_loader, device, num_epochs):

    mb = master_bar(range(num_epochs))

    validate(valid_loader, model, criterion, device, 0, num_epochs, mb)

    for epoch in mb:
        train_one_epoch(train_loader, model, criterion, optimizer, device, mb)
        validate(valid_loader, model, criterion,
                 device, epoch + 1, num_epochs, mb)


def main():

    aparser = ArgumentParser("Train a neural network on the MNIST dataset.")
    aparser.add_argument(
        "mnist", type=str, help="Path to store/find the MNIST dataset")
    aparser.add_argument("--num_epochs", type=int, default=3)
    aparser.add_argument("--batch_size", type=int, default=128)
    aparser.add_argument("--learning_rate", type=float, default=0.1)
    aparser.add_argument("--seed", action="store_true")
    aparser.add_argument("--gpu", action="store_true")

    args = aparser.parse_args()

    # Set the random number generator seed if one is provided
    if args.seed:
        torch.manual_seed(args.seed)

    # Use GPU if requested and available
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    # Get data loaders
    train_loader, valid_loader = get_mnist_data_loaders(
        args.mnist, args.batch_size, 0)

    # TODO: create a new model
    # Your model can be as complex or simple as you'd like. It must work
    # with the other parts of this script.)

    model = torch.nn.Sequential(

        torch.nn.Conv2d(1, 28, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(28, 56, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),

        torch.nn.Flatten(),
        torch.nn.Linear(10976, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 10))

    # TODO:
    # - create a CrossEntropyLoss criterion
    # - create an optimizer of your choice
    criterion = torch.nn.CrossEntropyLoss()

    # Adam
    # optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # SGD
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)

    # Adagrad
    # optimizer = torch.optim.Adagrad(model.parameters(), args.learning_rate)

    # RMSprop
    # optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)

    train(
        model, criterion, optimizer, train_loader, valid_loader, device, args.num_epochs
    )


if __name__ == "__main__":
    main()
