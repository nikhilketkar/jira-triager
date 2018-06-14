import sys
import gzip
import json
import torch
import torch.nn
import torch.optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import copy
import numpy

import ipdb

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

class String2Position:
    def __init__(self):
        self.mapping = {}
        self.counter = 0

    def get(self, given):
        if given not in self.mapping:
            self.mapping[given] = self.counter
            self.counter += 1
        return self.mapping[given]

def data_reader(path):
    result = []
    max_length = 0
    with open(path, 'r') as input_file:
        for line in input_file:
            words = line.strip().split("\t")
            component = words[1]
            summary = words[2]
            result.append((component, summary))
            max_length = max(len(summary), max_length)
    return max_length, result

def build_dataset(data_path, split):
    max_length, data = data_reader(data_path)
    train = []
    test = []
    s2p = String2Position()
    for component, summary in data:
        s2p.get(component)
        if numpy.random.random() > split:
            train.append((component, summary))
        else:
            test.append((component, summary))
    return train, test, max_length, s2p

class JiraIssues(Dataset):
    def __init__(self, data, max_length, s2p):
        self.data = data
        self.max_length = max_length
        self.s2p = s2p

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        component, summary = self.data[index]
        actual_length = len(summary)
        actual_values = [ord(i) for i in summary]
        padding_length = self.max_length - actual_length
        i = torch.LongTensor([actual_values + [0] * padding_length])
        o = self.s2p.get(component)
        return {"component": component,
                "summary": summary,
                "input": i,
                "actual_length": actual_length,
                "output": o}

def custom_batch(examples):
    examples.sort(key = lambda x: x["actual_length"], reverse = True)
    result = {"input": torch.cat([i["input"] for i in examples]),
              "actual_length": [i["actual_length"] for i in examples],
              "output": torch.LongTensor([i["output"] for i in examples])}
    return result

def custom_batch_cuda(examples):
    examples.sort(key = lambda x: x["actual_length"], reverse = True)
    result = {"input": torch.cat([i["input"] for i in examples]).cuda(),
              "actual_length": [i["actual_length"] for i in examples],
              "output": torch.LongTensor([i["output"] for i in examples]).cuda()}
    return result

class Model(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, lstm_layers, classes):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = embedding_size
        self.lstm_layers = lstm_layers
        self.E = torch.nn.Embedding(vocab_size, embedding_size)
        self.LSTM = torch.nn.LSTM(input_size=embedding_size, hidden_size=embedding_size, num_layers=lstm_layers, batch_first=True)
        self.Linear = torch.nn.Linear(embedding_size, classes)
        self.Softmax = torch.nn.Softmax(dim=1)

    def gen_hidden(self, batch_size):
        if next(self.parameters()).is_cuda:
            h0 = torch.rand((self.lstm_layers, batch_size, self.hidden_size)).cuda()
            c0 = torch.rand((self.lstm_layers, batch_size, self.hidden_size)).cuda()
        else:
            h0 = torch.rand((self.lstm_layers, batch_size, self.hidden_size))
            c0 = torch.rand((self.lstm_layers, batch_size, self.hidden_size))
        return h0, c0

    def forward(self, data):
        i = data["input"]
        actual_length = data["actual_length"]
        t0 = self.E(i)
        t1 = torch.nn.utils.rnn.pack_padded_sequence(t0, actual_length, batch_first=True)
        h0, c0 = self.gen_hidden(data["input"].shape[0])
        output, (h_n, c_n) = self.LSTM(t1,(h0,c0))
        x, y = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        i = torch.stack([y - 1] * self.hidden_size, 1).view(y.shape[0], 1, -1)
        if next(self.parameters()).is_cuda:
            i = i.cuda()
        lstm_out = x.gather(1, i).view(y.shape[0], -1).transpose(0,1)
        linear_out = self.Linear(lstm_out.transpose(0,1))
        softmax_out = self.Softmax(linear_out)
        return softmax_out

def eval(model, data, criterion):
    loss = []
    actual = []
    predicted = []
    for batch in data:
        y_hat = model.forward(batch)
        _, y_hat_index = y_hat.max(1)
        actual.append(batch["output"].cpu())
        predicted.append(y_hat_index.cpu())
        loss.append(criterion(y_hat.cpu(), batch["output"].cpu()).data[0])
    actual = torch.cat(actual)
    predicted = torch.cat(predicted)
    return precision_recall_fscore_support(actual, predicted, average = 'weighted'), np.mean(loss)

def train(model, criterion, train_data_loader, optimizer):
    for batch in train_data_loader:
        optimizer.zero_grad()
        y_hat = model.forward(batch)
        loss = criterion(y_hat, batch["output"])
        loss.backward()
        optimizer.step()


def write_results(parameters, epoch, train_metric, test_metric, train_loss, test_loss, output_file):
    result = copy.copy(parameters)
    result["epoch"] = epoch
    result["train_metric"] = train_metric
    result["test_metric"] = test_metric
    result["train_loss"] = float(train_loss)
    result["test_loss"] = float(test_loss)
    output_file.write(json.dumps(result) + "\n")


def train_eval(train_examples, test_examples, max_length, s2p, epochs, parameters, output_file, cuda):
    train_dataset = JiraIssues(train_examples, max_length, s2p)
    test_dataset = JiraIssues(test_examples, max_length, s2p)

    if cuda:
        custom_batch_function = custom_batch_cuda
    else:
        custom_batch_function = custom_batch

    train_data_loader = DataLoader(train_dataset, batch_size=parameters["batch_size"], collate_fn=custom_batch_function)
    test_data_loader = DataLoader(test_dataset, batch_size=parameters["batch_size"], collate_fn=custom_batch_function)

    model = Model(parameters["embedding_in"], parameters["embedding_out"], parameters["lstm_layers"], s2p.counter)
    
    if cuda:
        model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=parameters["learning_rate"])

    train_metric, train_loss = eval(model, train_data_loader, criterion)
    test_metric, test_loss = eval(model, test_data_loader, criterion)
    write_results(parameters, -1, train_metric, test_metric, train_loss, test_loss, output_file)

    for epoch in range(epochs):
        train(model, criterion, train_data_loader, optimizer)
        train_metric, train_loss = eval(model, train_data_loader, criterion)
        test_metric, test_loss = eval(model, test_data_loader, criterion)
        write_results(parameters, epoch, train_metric, test_metric, train_loss, test_loss, output_file)


class RandomGrid:
    def __init__(self):
        self.parameters = {}
        self.static_values = {}
    def add_parameter(self, name, values):
        self.parameters[name] = values
    def add_static(self, name, value):
        self.static_values[name] = value

    def grid(self):
        counter = 0
        while(True):
            counter += 1
            result = {}
            for parameter, values in self.parameters.items():
                result[parameter] = random.choice(values)
            for static_key, static_value in self.static_values.items():
                result[static_key] = static_value
            yield counter, result

def main(data_path, split, epochs, grid_points, output_path, cuda=False):
    train_examples, test_examples, max_length, s2p = build_dataset(data_path, split)
    random_grid = RandomGrid()
    random_grid.add_parameter("learning_rate", [1e-2, 1e-3, 1e-4, 1e-5])
    random_grid.add_parameter("batch_size", [1024])
    random_grid.add_parameter("embedding_in", [256])
    random_grid.add_parameter("embedding_out", [32, 64, 128])
    random_grid.add_parameter("lstm_layers", [1,2,3])

    with open(output_path, 'w', 0) as output_file:
        for counter, parameters in random_grid.grid():
            if counter < grid_points:
                train_eval(train_examples, test_examples, max_length, s2p, epochs, parameters, output_file, cuda)

if __name__ == "__main__":
    payload_path = sys.argv[1]
    labels_path = sys.argv[2]
    retain = float(sys.argv[3])
    split = float(sys.argv[4])
    epochs = int(sys.argv[5])
    grid_points = float(sys.argv[6])
    output_path = sys.argv[7]
    if sys.argv[8] == "cuda":
        cuda = True
    elif sys.argv[8] == "no-cuda":
        cuda = False

    main(payload_path, labels_path, retain, split, epochs, grid_points, output_path, cuda=cuda)
