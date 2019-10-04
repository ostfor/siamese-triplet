import torch
import numpy as np, os
from rclone_data_loader.config import create_not_ex
from livelossplot import PlotLosses


def timestamp():
    from datetime import datetime
    return datetime.now().strftime("%d-%m-%y-%H%M%S")


def visualize(test_data, test_data_cpu, model):
    from visualization import plot_embeddings
    embs = model.get_embedding(test_data)
    embs = embs.detach().cpu().numpy()
    plot_embeddings(test_data_cpu, embs)


def fit(train_loader, val_loader, test_data, test_data_cpu, model, loss_fn, optimizer, scheduler, n_epochs, cuda,
        log_interval, metrics=[],
        start_epoch=0, project_root=None, path_to_entire_model=None):
    if project_root is None:
        project_root = create_not_ex("data/results/net_tex_run_{}".format(timestamp()))
    if path_to_entire_model is not None:
        model = torch.load(path_to_entire_model)
    for epoch in range(0, start_epoch):
        scheduler.step()

    states_path = create_not_ex(os.path.join(project_root, "states"))
    entire_path = create_not_ex(os.path.join(project_root, "entire"))
    training_name = "net_tex_model_{}_".format(model.__class__.__name__) + "_{}_{:.4f}.model"

    best_loss = None
    liveloss = PlotLosses(skip_first=1)
    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage

        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
        train_loader.dataset.reinitialize_train_dataset()

        print(message)
        liveloss.update({"val_loss": val_loss, "loss": train_loss})
        liveloss.draw()
        visualize(test_data, test_data_cpu, model)

        if best_loss is None or val_loss < best_loss:
            path_to_entire = os.path.join(entire_path, training_name.format(epoch, train_loss))
            path_to_eval = os.path.join(states_path, training_name.format(epoch, train_loss))
            torch.save(model.state_dict(), path_to_eval)
            torch.save({
                "epoch": epoch,
                "project_root": project_root,
                "val_loss": val_loss,
                "train_loss": train_loss,
                "scheduler": scheduler.state_dict(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, path_to_entire)
            best_loss = val_loss


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target
        # print("Loss Inputs: ", loss_inputs)
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
