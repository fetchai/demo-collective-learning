from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as nn_func
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy

hook = sy.TorchHook(torch)

LOSSES = {}  # dictionary of loss for each worker


class Arguments:
    def __init__(self):
        self.batch_size = 64
        self.n_batches_for_vote = 5
        self.test_batch_size = 1000
        self.epochs = 5
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 30
        self.save_model = False
        self.n_hospitals = 5
        self.vote_threshold = (self.n_hospitals - 1) // 2


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = nn_func.relu(self.conv1(x))
        x = nn_func.max_pool2d(x, 2, 2)
        x = nn_func.relu(self.conv2(x))
        x = nn_func.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = nn_func.relu(self.fc1(x))
        x = self.fc2(x)
        return nn_func.log_softmax(x, dim=1)


def colearn_train(args, model: Net, device,
                  federated_train_loader: sy.FederatedDataLoader,
                  optimizer, epoch, workers):
    global LOSSES
    model.train()  # sets model to "training" mode. Does not perform training.
    # need to save the state dict of the old model
    state_dict = model.state_dict()
    # pick a random hospital
    proposer_index = randint(0, len(workers) - 1)
    proposer = workers[proposer_index]
    print("Proposer", proposer_index, proposer)
    model.send(proposer)

    # go through all the batches for hosp_n, perform training, get model back
    for batch_idx, data_dict in enumerate(federated_train_loader):  # a distributed dataset
        data, target = data_dict[proposer]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn_func.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            loss = loss.get()  # get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size,
                len(federated_train_loader) * args.batch_size,
                100. * batch_idx * len(workers) / len(federated_train_loader), loss.item()))

    # send to others and vote
    model.get()
    losses = test_on_training_set(args, model, device, federated_train_loader, workers)
    vote = False
    print("Doing voting")
    if not LOSSES:
        vote = True
        print("First round, no voting")
        LOSSES = losses
    else:
        votes = 0
        for worker, old_loss in LOSSES.items():
            if worker != proposer:
                if old_loss > losses[worker]:
                    votes += 1
                    print(worker, "votes yes")
                else:
                    print(worker, "votes no")
        if votes >= args.vote_threshold:
            print("Vote succeeded")
            vote = True
            LOSSES = losses
    if not vote:
        print("Vote failed")
        # then load the old weights into the model
        model.load_state_dict(state_dict)


def test_on_training_set(args: Arguments, model, device, train_loader, workers):
    model.eval()  # sets model to "eval" mode.
    losses = {w: 0 for w in workers}
    batch_count = 0
    with torch.no_grad():
        for data_dict in train_loader:
            for worker, (data, target) in data_dict.items():
                model.send(data.location)
                data, target = data.to(device), target.to(device)
                output = model(data)
                losses[worker] += nn_func.nll_loss(output, target, reduction='sum').get()  # sum up batch loss
                model.get()
            batch_count += 1
            if batch_count == args.n_batches_for_vote:
                break
    return losses


def test(args, model, device, test_loader):
    model.eval()  # sets model to "eval" mode.
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn_func.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    args = Arguments()

    hospitals = []
    for i in range(args.n_hospitals):
        hospitals.append(sy.VirtualWorker(hook, id="hospital " + str(i)))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    federated_train_loader = sy.FederatedDataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])).federate(hospitals),
        batch_size=args.batch_size, shuffle=True, iter_per_worker=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)  # TODO momentum is not supported at the moment

    for epoch in range(1, args.epochs + 1):
        colearn_train(args, model, device, federated_train_loader, optimizer, epoch, hospitals)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
