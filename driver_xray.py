import sys
from pathlib import Path
from random import randint
import random as rand
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as nn_func
import torch.optim as optim
from sklearn.metrics import roc_curve
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np
import syft as sy
from torchsummary import summary

hook = sy.TorchHook(torch)

LOSSES = {}  # dictionary of loss for each worker


class Arguments:
    def __init__(self):
        self.batch_size = 8
        self.n_batches_for_vote = 12
        self.test_batch_size = 64
        self.epochs = 15
        self.steps_per_epoch = 10
        self.lr = 0.01
        # self.momentum = 0.5  # momentum is not supported by pysyft
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 1
        self.save_model = False
        self.n_hospitals = 5
        self.vote_threshold = (self.n_hospitals - 1) // 2
        self.train_ratio = 0.92
        self.data_dir = "/home/emmasmith/Development/datasets/chest_xray"
        self.pos_weight = torch.tensor([0.27])  # there is about 1 normal to every 3 pneu


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
        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 1)
        self.threshold = 0.5

    def forward(self, x):
        x = nn_func.relu(self.conv1(x))
        x = self.bn1(x)
        x = nn_func.max_pool2d(x, kernel_size=(4, 4))
        x = nn_func.relu(self.conv2(x))
        x = self.bn2(x)
        x = nn_func.max_pool2d(x, kernel_size=(32, 32))
        x = x.view(-1, 64)
        x = self.fc1(x)
        return x  # NB: output is in *logits*


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
    criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight, reduction='mean')

    # go through all the batches for hosp_n, perform training, get model back
    for batch_idx, data_dict in enumerate(federated_train_loader):  # a distributed dataset
        data, target = data_dict[proposer]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            loss = loss.get()  # get the loss back
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size,
                       len(federated_train_loader) * args.batch_size,
                       100. * batch_idx * len(workers) / len(federated_train_loader), loss.item()))
        if batch_idx == args.steps_per_epoch:
            break

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
                if old_loss >= losses[worker]:
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
    criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight, reduction='sum')
    best_thresholds = []
    with torch.no_grad():
        for data_dict in train_loader:
            for worker, (data, target) in data_dict.items():
                model.send(data.location)
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred_prob = torch.sigmoid(output)
                losses[worker] += criterion(output, target).get()  # sum up batch loss
                best_thresholds.append(get_best_threshold(target.get().numpy(),
                                                          pred_prob.get().numpy()))
                model.get()
            batch_count += 1
            if batch_count == args.n_batches_for_vote:
                break
    print(best_thresholds)
    best_thresholds.sort()
    model.threshold = best_thresholds[int(len(best_thresholds) / 2)]
    return losses


def test(args, model, device, test_loader):
    model.eval()  # sets model to "eval" mode.
    test_loss = 0
    correct = 0
    criterion = nn.BCEWithLogitsLoss(pos_weight=args.pos_weight, reduction='sum')
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            pred = torch.sigmoid(output) > model.threshold
            correct += int((pred == target).int().sum())

    test_loss /= len(test_loader.dataset)
    print(model.threshold)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_best_threshold(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # get the best threshold
    j = tpr - fpr
    ix = np.argmax(j)
    best_thresh = thresholds[ix]
    return best_thresh


class XrayDataset(Dataset):
    """X-ray dataset."""

    def __init__(self, data_dir, width=128, height=128, seed=42, transform=None, train=True, train_ratio=0.96):
        """
        Args:
            data_dir (string): Path to the data directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.width, self.height = width, height
        self.cases = list(Path(data_dir).rglob('*.jp*'))  # list of filenames
        if len(self.cases) == 0:
            raise Exception("No data foud in path: " + str(data_dir))

        rand.seed(seed)
        rand.shuffle(self.cases)

        n_cases = int(train_ratio * len(self.cases))
        assert (n_cases > 0), "There are no cases"
        if train:
            self.cases = self.cases[:n_cases]
        else:
            self.cases = self.cases[n_cases:]

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
        # todo: balanced classes
        if torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            idx = [idx]

        batch_size = len(idx)

        # Define two numpy arrays for containing batch data and labels
        batch_data = np.zeros((batch_size, self.width, self.height), dtype=np.float32)
        batch_labels = np.zeros(batch_size, dtype=np.float32)

        for j, index in enumerate(idx):
            batch_data[j] = self.to_rgb_normalize_and_resize(self.cases[index], self.width, self.height)
            batch_labels[j] = self.diagnosis[index]

        sample = (batch_data, batch_labels)
        if self.transform:
            sample = self.transform(sample)

        return sample

    @staticmethod
    def to_rgb_normalize_and_resize(filename, width, height):
        img = cv2.imread(str(filename))
        img = cv2.resize(img, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype(np.float32) / 255.

        return img


def main():
    args = Arguments()

    hospitals = []
    for i in range(args.n_hospitals):
        hospitals.append(sy.VirtualWorker(hook, id="hospital " + str(i)))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    model = Net().to(device)
    summary(model, input_size=(1, 128, 128))

    # need to make a federated dataset for training and an unfederated one for testing
    test_dataset = XrayDataset(args.data_dir, train=False, train_ratio=args.train_ratio)

    labels = test_dataset.diagnosis  # corresponding labels of samples
    class_counts = [len(test_dataset.normal_data), len(test_dataset.pneumonia_data)]
    num_samples = len(test_dataset)

    class_weights = [num_samples / i for i in class_counts]
    weights = [class_weights[labels[i]] for i in range(num_samples)]
    sampler = WeightedRandomSampler(torch.tensor(weights), int(num_samples))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, sampler=sampler, **kwargs)

    # initial accuracy
    test(args, model, device, test_loader)
    # sys.exit()

    # make a federated dataset for training
    train_dataset = XrayDataset(args.data_dir, train_ratio=args.train_ratio)

    # labels = train_dataset.diagnosis  # corresponding labels of samples
    # class_counts = [len(train_dataset.normal_data), len(train_dataset.pneumonia_data)]
    # num_samples = len(train_dataset)
    #
    # class_weights = [num_samples / i for i in class_counts]
    # weights = [class_weights[labels[i]] for i in range(num_samples)]
    # sampler = WeightedRandomSampler(torch.tensor(weights), int(num_samples))

    fed_dataset = train_dataset.federate(hospitals)
    # sampler is not supported
    fdataloader = sy.FederatedDataLoader(fed_dataset, batch_size=args.batch_size,
                                         iter_per_worker=True, shuffle=True, **kwargs)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        colearn_train(args, model, device, fdataloader, optimizer, epoch, hospitals)
        test(args, model, device, test_loader)
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
