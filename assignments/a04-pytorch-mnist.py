#!/usr/bin/env python

from argparse import ArgumentParser
from utils import get_mnist_data_loaders, DataLoaderProgress
from fastprogress.fastprogress import master_bar, progress_bar
import torch

"""
Original Accuracy (Original, SGD): 
Initial accuracy= 8.08% and loss=2.349
Epoch  1/10: accuracy=92.51% and loss=0.250                                                                               
Epoch  2/10: accuracy=94.84% and loss=0.171                                                                               
Epoch  3/10: accuracy=95.94% and loss=0.140                                                                               
Epoch  4/10: accuracy=96.11% and loss=0.132                                                                               
Epoch  5/10: accuracy=96.09% and loss=0.128                                                                               
Epoch  6/10: accuracy=96.40% and loss=0.116                                                                               
Epoch  7/10: accuracy=96.20% and loss=0.117                                                                               
Epoch  8/10: accuracy=96.57% and loss=0.112                                                                               
Epoch  9/10: accuracy=96.73% and loss=0.108                                                                               
Epoch 10/10: accuracy=96.95% and loss=0.102  

Epoch  1/3: accuracy=93.57% and loss=0.216                        
Epoch  2/3: accuracy=94.10% and loss=0.195                        
Epoch  3/3: accuracy=95.93% and loss=0.148

Second Accuracy (Adam);
Initial accuracy=13.29% and loss=2.343
Epoch  1/10: accuracy=50.97% and loss=1.419                      
Epoch  2/10: accuracy=40.83% and loss=1.803                        
Epoch  3/10: accuracy=45.58% and loss=1.756                        
Epoch  4/10: accuracy=36.84% and loss=1.714                        
Epoch  5/10: accuracy=36.44% and loss=1.917                        
Epoch  6/10: accuracy=27.26% and loss=1.943                        
Epoch  7/10: accuracy=26.12% and loss=2.019                        
Epoch  8/10: accuracy=19.60% and loss=2.152                        
Epoch  9/10: accuracy=18.56% and loss=2.561                        
Epoch 10/10: accuracy=14.62% and loss=2.265 

Initial accuracy= 6.27% and loss=2.358
Epoch  1/3: accuracy=29.49% and loss=1.860                        
Epoch  2/3: accuracy=24.22% and loss=1.983                        
Epoch  3/3: accuracy=24.38% and loss=1.982  

Third Accuracy (Adagrad)

Initial accuracy=15.38% and loss=2.337
Epoch  1/10: accuracy=92.46% and loss=0.258                        
Epoch  2/10: accuracy=94.26% and loss=0.195                        
Epoch  3/10: accuracy=95.25% and loss=0.164                        
Epoch  4/10: accuracy=95.65% and loss=0.152                        
Epoch  5/10: accuracy=95.59% and loss=0.147                        
Epoch  6/10: accuracy=95.80% and loss=0.142                        
Epoch  7/10: accuracy=96.01% and loss=0.138                        
Epoch  8/10: accuracy=96.04% and loss=0.137                        
Epoch  9/10: accuracy=95.89% and loss=0.137                        
Epoch 10/10: accuracy=96.21% and loss=0.132 

Initial accuracy=16.80% and loss=2.358
Epoch  1/3: accuracy=92.06% and loss=0.265                        
Epoch  2/3: accuracy=93.62% and loss=0.216                        
Epoch  3/3: accuracy=94.27% and loss=0.193

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
    aparser.add_argument("--num_epochs", type=int, default=10)
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
        torch.nn.Flatten(),
        torch.nn.Linear(784, 28),
        torch.nn.ReLU(),
        torch.nn.Linear(28, 10))

    # TODO:
    # - create a CrossEntropyLoss criterion
    # - create an optimizer of your choice
    criterion = torch.nn.CrossEntropyLoss()

    # Adam
    #optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    # SGD
    # optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)

    # Adagrad
    optimizer = torch.optim.Adagrad(model.parameters(), args.learning_rate)

    # AMSGrad
    optimizer = torch.optim.AMSGrad(model.parameters(), args.learning_rate)

    train(
        model, criterion, optimizer, train_loader, valid_loader, device, args.num_epochs
    )


if __name__ == "__main__":
    main()
