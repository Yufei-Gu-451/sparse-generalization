import torch
import csv
import os

from datetime import datetime
from tqdm import tqdm

import models
import data_src


# ------------------------------------------------------------------------------------------


# Return the train_dataloader and test_dataloader
def get_train_and_test_dataloader(args, dataset_path, noise_ratio):
    train_dataset = data_src.load_train_dataset_from_file(label_noise_ratio=noise_ratio,
                                                          dataset_path=dataset_path)

    train_dataloader = data_src.get_dataloader_from_dataset(train_dataset, args.batch_size, args.workers)

    test_dataset = data_src.get_test_dataset(dataset=args.dataset)

    test_dataloader = data_src.get_dataloader_from_dataset(test_dataset, args.batch_size, args.workers)

    print(f'Load {args.dataset} dataset success;')

    return train_dataloader, test_dataloader


# ------------------------------------------------------------------------------------------


def train_model_manual_bp(model, device, optimizer, criterion, train_dataloader):
    model.train()
    cumulative_loss, correct, total, idx = 0.0, 0, 0, 0

    norm_sigmoid = models.NormSigmoid()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # print("1:{}".format(torch.cuda.memory_allocated(0)))

        optimizer.zero_grad()
        _, _, act_2 = model.forward_full(inputs)
        loss = criterion(act_2, labels)
        loss.backward()

        # print("2:{}".format(torch.cuda.memory_allocated(0)))

        with torch.no_grad():
            for i, (name, param) in enumerate(model.named_parameters()):
                if 'features.1.weight' in name:
                    sparse_regu_term = (norm_sigmoid(model.features_act_mat) / 10000).to(device)
                elif 'classifier.1.weight' in name:
                    sparse_regu_term = (norm_sigmoid(model.act_mat_list[i // 2]) / 10000).to(device)
                else:
                    sparse_regu_term = torch.zeros(param.grad.shape).to(device)

                param -= optimizer.param_groups[0]['lr'] * (param.grad + sparse_regu_term)
                sparse_regu_term.detach().cpu()
                del sparse_regu_term

        cumulative_loss += loss.item()
        _, predicted = act_2.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.argmax(1)).sum().item()

        act_2.detach()
        inputs.detach()
        labels.detach()
        del inputs, labels, act_2
        torch.cuda.empty_cache()

        # print("3:{}".format(torch.cuda.memory_allocated(0)))

    train_loss = cumulative_loss / (idx + 1)
    train_acc = correct / total

    return model, train_loss, train_acc


def train_model(model, device, optimizer, criterion, train_dataloader):
    model.train()
    cumulative_loss, correct, total, idx = 0.0, 0, 0, 0

    for idx, (inputs, labels) in enumerate(train_dataloader):
        labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        cumulative_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.argmax(1)).sum().item()

    train_loss = cumulative_loss / (idx + 1)
    train_acc = correct / total

    return model, train_loss, train_acc


def test_model(model, device, criterion, test_dataloader):
    model.eval()
    cumulative_loss, correct, total, idx = 0.0, 0, 0, 0

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(1)).sum().item()

    test_loss = cumulative_loss / (idx + 1)
    test_acc = correct / total

    return test_loss, test_acc


# Train and Evaluate the model
def train_and_evaluate_model(model, device, args, train_dataloader, test_dataloader,
                             dictionary_path, checkpoint_path, manual_bp):
    start_time = datetime.now()

    # Set the optimizer and criterion
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    n_parameters = sum(p.numel() for p in model.parameters())
    n_hidden_units = model.n_hidden_units

    # Initialize the dictionary file for n_hidden_unit
    dict_n_path = os.path.join(dictionary_path, "dictionary_%d.csv" % n_hidden_units)
    init_dict(dict_n_path)

    for epoch in tqdm(range(1, args.epochs + 1)):
        # Train Model
        if manual_bp:
            model, train_loss, train_acc = train_model_manual_bp(model, device, optimizer, criterion, train_dataloader)
        else:
            model, train_loss, train_acc = train_model(model, device, optimizer, criterion, train_dataloader)

        print("Epoch : %d ; Train Loss : %f ; Train Acc : %.3f ; Learning Rate : %f" %
              (epoch, train_loss, train_acc, optimizer.param_groups[0]['lr']))

        if epoch % 50 == 0:
            if args.dataset == 'MNIST':
                optimizer.param_groups[0]['lr'] = args.lr / pow(1 + epoch // 50, 0.5)
            elif args.dataset == 'CIFAR-10':
                optimizer.param_groups[0]['lr'] = args.lr / pow(1 + epoch * 10, 0.5)
            else:
                raise NotImplementedError

            # Test Model
            test_loss, test_acc = test_model(model, device, criterion, test_dataloader)

            curr_time = datetime.now()
            time = (curr_time - start_time).seconds / 60

            save_dict(n_hidden_units, epoch, n_parameters, train_loss, train_acc, test_loss, test_acc,
                      optimizer.param_groups[0]['lr'], time, curr_time, dict_n_path)

    models.save_model(model, checkpoint_path, n_hidden_units)

    return

# ------------------------------------------------------------------------------------------


def init_dict(dictionary_n_path):
    dictionary = {'Hidden Neurons': 0, 'Epochs': 0, 'Parameters': 0, 'Train Loss': 0, 'Train Accuracy': 0,
                  'Test Loss': 0, 'Test Accuracy': 0, 'Learning Rate': 0, 'Time Cost': 0, 'Date-Time': 0}

    with open(dictionary_n_path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=dictionary.keys())
        writer.writeheader()


def save_dict(n_hidden_units, epoch, parameters, train_loss, train_acc, test_loss, test_acc, lr,
              time, curr_time, dict_path):
    print("Hidden Neurons : %d ; Parameters : %d ; Train Loss : %.3f ; Train Acc : %.3f ; Test Loss : %.3f ; "
          "Test Acc : %.3f\n" % (n_hidden_units, parameters, train_loss, train_acc, test_loss, test_acc))

    print('Writing to a csv file...')
    state_dictionary = {'Hidden Neurons': n_hidden_units, 'Epochs': epoch, 'Parameters': parameters,
                        'Train Loss': train_loss, 'Train Accuracy': train_acc,
                        'Test Loss': test_loss, 'Test Accuracy': test_acc,
                        'Learning Rate': lr, 'Time Cost': time, 'Date-Time': curr_time}

    with open(dict_path, "a", newline="") as file:
        # Create a writer object
        dict_writer = csv.DictWriter(file, fieldnames=state_dictionary.keys())

        # Write the data rows
        dict_writer.writerow(state_dictionary)
        print('Done writing to a csv file\n')


def read_dict(dict_path, epochs, hidden_units):
    parameters, train_accuracy, test_accuracy, train_losses, test_losses = [], [], [], [], []

    for hidden_unit in hidden_units:
        # Get Parameters and dataset Losses
        dictionary_path_n = os.path.join(dict_path, "dictionary_%d.csv" % hidden_unit)

        with open(dictionary_path_n, "r", newline="") as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                if int(row['Epochs']) == epochs:
                    parameters.append(int(row['Parameters']) // 1000)
                    train_accuracy.append(float(row['Train Accuracy']))
                    test_accuracy.append(float(row['Test Accuracy']))
                    train_losses.append(float(row['Train Loss']))
                    test_losses.append(float(row['Test Loss']))

    return parameters, train_accuracy, test_accuracy, train_losses, test_losses
