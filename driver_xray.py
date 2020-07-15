from pathlib import Path
from random import randint
import random as rand
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
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

        self.data_dir = "/home/emmasmith/Development/datasets/chest_xray"


class Net(nn.Module):
    """_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
Input (InputLayer)           [(None, 128, 128, 1)]     0
_________________________________________________________________
Conv1_1 (Conv2D)             (None, 128, 128, 32)      320
_________________________________________________________________
bn1 (BatchNormalization)     (None, 128, 128, 32)      128
_________________________________________________________________
pool1 (MaxPooling2D)         (None, 32, 32, 32)        0
_________________________________________________________________
Conv2_1 (Conv2D)             (None, 32, 32, 64)        18496
_________________________________________________________________
bn2 (BatchNormalization)     (None, 32, 32, 64)        256
_________________________________________________________________
global_max_pooling2d (Global (None, 64)                0
_________________________________________________________________
fc1 (Dense)                  (None, 1)                 65
=================================================================
Total params: 19,265
Trainable params: 19,073
Non-trainable params: 192
_________________________________________________________________"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(1, 64, (3, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = nn_func.relu(self.conv1(x))
        x = self.bn1(x)
        x = nn_func.max_pool2d(x, 4, 4)
        x = nn_func.relu(self.conv2(x))
        x = self.bn2(x)
        x = nn_func.max_pool2d(x, kernel_size=x.size()[2:])
        x = nn_func.sigmoid(self.fc1(x))
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


def to_rgb_normalize_and_resize(img, width, height):
    img = cv2.imread(str(img))
    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.
    img = np.reshape(img, (width, height, 1))

    return img


def normalize_image(filename, class_num, width, height):
    data = to_rgb_normalize_and_resize(filename, width, height)
    label = class_num

    return data, label


class XrayDataset(Dataset):
    """X-ray dataset."""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Path to the data directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.cases = list(Path(data_dir).rglob('*.jp*'))  # list of filenames
        if len(self.cases) == 0:
            raise Exception("No data foud in path: " + str(data_dir))
        shuffle_seed = 42
        rand.seed(shuffle_seed)
        rand.shuffle(self.cases)

        self.diagnosis = []  # list of filenames

        self.normal_data = []
        self.pneumonia_data = []
        for case in self.cases:
            if 'NORMAL' in str(case):
                self.diagnosis.append(0)
                self.normal_data.append(case)
            elif 'PNEUMONIA' in str(case):
                self.diagnosis.append(1)
                self.pneumonia_data.append(case)
            else:
                print(case, " - has invalid category")

        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            batch_size = len(idx)
        else:
            batch_size = 1

        width, height = 128, 128
        # Define two numpy arrays for containing batch data and labels
        batch_data = np.zeros((batch_size, width, height, 1), dtype=np.float32)
        batch_labels = np.zeros((batch_size, 1), dtype=np.float32)

        # Initialize a counter
        for i in range(batch_size):
            img_data, img_label = normalize_image(self.cases[i], 0, width,
                                                  height)

            batch_data[i] = img_data
            batch_labels[i] = img_label

        sample = (batch_data, batch_labels)
        if self.transform:
            sample = self.transform(sample)

        return sample


def main():
    args = Arguments()

    hospitals = []
    for i in range(args.n_hospitals):
        hospitals.append(sy.VirtualWorker(hook, id="hospital " + str(i)))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # need to make a federated dataset for training and an unfederated one for testing
    fed_dataset = XrayDataset(args.data_dir).federate(hospitals)

    fdataloader = sy.FederatedDataLoader(fed_dataset, batch_size=args.batch_size,
                                         shuffle=True, iter_per_worker=True, **kwargs)

    # todo: unfederated test loader
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        colearn_train(args, model, device, fdataloader, optimizer, epoch, hospitals)
        # test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
