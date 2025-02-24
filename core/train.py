import torch
import torch.nn.functional as F
import os
import logging
import numpy as np


def model_save(model, ckpts, step, k=None, freq=50):

    if (step + 1) % freq == 0:
        prefix = '' if k == None else 'client{}_'.format(k)
        file_name = os.path.join(ckpts, prefix + 'model_{}.pth'.format(step+1))
        torch.save(model.state_dict(), file_name)


def tensorboard_record(train_loss, val_records, overall_val_loss, overall_val_acc, writer, step, k): 

    prefix = 'client{}_'.format(k)
    writer.add_scalar(prefix + 'train_loss', train_loss, global_step=step)
    for k, (val_loss, val_acc) in enumerate(val_records):
        prefix = 'client{}_'.format(k)
        writer.add_scalar(prefix + 'val_loss', val_loss, global_step=step)
        writer.add_scalar(prefix + 'val_acc', val_acc, global_step=step)
    writer.add_scalar('overall_val_loss', overall_val_loss, global_step=step)
    writer.add_scalar('overall_val_acc', overall_val_acc, global_step=step)


def tensorboard_record_si(pred_train_loss, surr_train_loss, train_loss, val_records, overall_val_loss, overall_val_acc, writer, step, k): 

    prefix = 'client{}_'.format(k)
    writer.add_scalar(prefix + 'pred_train_loss', pred_train_loss, global_step=step)
    writer.add_scalar(prefix + 'surr_train_loss', surr_train_loss, global_step=step)
    writer.add_scalar(prefix + 'train_loss', train_loss, global_step=step)
    for k, (val_loss, val_acc) in enumerate(val_records):
        prefix = 'client{}_'.format(k)
        writer.add_scalar(prefix + 'val_loss', val_loss, global_step=step)
        writer.add_scalar(prefix + 'val_acc', val_acc, global_step=step)
    writer.add_scalar('overall_val_loss', overall_val_loss, global_step=step)
    writer.add_scalar('overall_val_acc', overall_val_acc, global_step=step)


def tensorboard_record_bacc(pred_train_loss, surr_train_loss, train_loss, val_records, overall_val_loss, overall_val_acc, overall_bacc, writer, step, k): 

    prefix = 'client{}_'.format(k)
    writer.add_scalar(prefix + 'pred_train_loss', pred_train_loss, global_step=step)
    writer.add_scalar(prefix + 'surr_train_loss', surr_train_loss, global_step=step)
    writer.add_scalar(prefix + 'train_loss', train_loss, global_step=step)
    for k, (val_loss, val_acc, bacc) in enumerate(val_records):
        prefix = 'client{}_'.format(k)
        writer.add_scalar(prefix + 'val_loss', val_loss, global_step=step)
        writer.add_scalar(prefix + 'val_acc', val_acc, global_step=step)
        writer.add_scalar(prefix + 'val_bacc', bacc, global_step=step)
    writer.add_scalar('overall_val_loss', overall_val_loss, global_step=step)
    writer.add_scalar('overall_val_acc', overall_val_acc, global_step=step)
    writer.add_scalar('overall_val_bacc', overall_bacc, global_step=step)


def tensorboard_record_baseline(train_loss, val_loss, val_acc, writer, step, k): 

    prefix = 'client{}_'.format(k)
    writer.add_scalar(prefix + 'train_loss', train_loss, global_step=step)
    writer.add_scalar(prefix + 'val_loss', val_loss, global_step=step)
    writer.add_scalar(prefix + 'val_acc', val_acc, global_step=step)


def tensorboard_record_fedavg(val_records, overall_val_loss, overall_val_acc, writer, step):

    for k, (val_loss, val_acc) in enumerate(val_records):
        prefix = 'global_client{}_'.format(k)
        writer.add_scalar(prefix + 'val_loss', val_loss, global_step=step)
        writer.add_scalar(prefix + 'val_acc', val_acc, global_step=step)
    writer.add_scalar('global_overall_val_loss', overall_val_loss, global_step=step)
    writer.add_scalar('global_overall_val_acc', overall_val_acc, global_step=step)

def tensorboard_record_fedavg_bacc(val_records, overall_val_loss, overall_val_acc, overall_bacc, writer, step):

    for k, (val_loss, val_acc, bacc) in enumerate(val_records):
        prefix = 'global_client{}_'.format(k)
        writer.add_scalar(prefix + 'val_loss', val_loss, global_step=step)
        writer.add_scalar(prefix + 'val_acc', val_acc, global_step=step)
        writer.add_scalar(prefix + 'val_bacc', bacc, global_step=step)
    writer.add_scalar('global_overall_val_loss', overall_val_loss, global_step=step)
    writer.add_scalar('global_overall_val_acc', overall_val_acc, global_step=step)
    writer.add_scalar('global_overall_val_bacc', overall_bacc, global_step=step)



def compute_bacc(predictions, targets, n_classes):

    predictions = torch.tensor(predictions, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    confusion_matrix = torch.zeros(n_classes, n_classes)
    for t, p in zip(targets.view(-1), predictions.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    print(confusion_matrix)
    per_class_accuracy = confusion_matrix.diag() / confusion_matrix.sum(1)
    balanced_accuracy = per_class_accuracy[~torch.isnan(per_class_accuracy)].mean().item()
    return balanced_accuracy



def train_loop(dataloader, model, loss_fn, optimizer):
    
    model.train()

    train_loss = 0

    for i, (X, y) in enumerate(dataloader): 

        # Data
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # Model
        pred = model(X)
        loss = loss_fn(pred, y) / len(X)

        # GD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        msg = 'Iter {}/{}, Loss {:.4f}, '.format((i+1), len(dataloader), loss.item())
        logging.info(msg)        
        
        train_loss += loss.item()

    train_loss /= len(dataloader)
    return train_loss


def val_loop(dataloader, model, loss_fn, bacc=False):
    
    model.eval()

    val_loss, correct = 0, 0

    if bacc:
        label_list = []
        pred_list = []

    for i, (X, y) in enumerate(dataloader):
        
        # Data
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # Model
        pred = model(X)
        val_loss += loss_fn(pred, y).item()

        # Acc
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # BAcc
        if bacc:
            label_list += y.tolist()
            _, pred_label = pred.max(1) 
            pred_list += pred_label.tolist()

    val_loss /= (len(dataloader) * len(X))
    correct /= (len(dataloader) * len(X))
    if bacc:
        bacc = compute_bacc(pred_list, label_list, n_classes=pred.size(1))
        return val_loss, correct, (bacc, pred_list, label_list)

    return val_loss, correct


def val_loop_binary(dataloader, model, loss_fn, mask):
    # mask: [True, True, False, False, ...] : ndarray
    model.eval()

    val_loss, correct = 0, 0

    for i, (X, y) in enumerate(dataloader):
        
        # Data
        X = X.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        # Model
        pred = model(X)
        val_loss += loss_fn(pred, y).item()

        # Acc
        mask = torch.Tensor(mask).bool().expand(pred.shape).cuda()
        pred = pred[mask].reshape(pred.shape[0], 2)
        y = torch.eye(10)[y][mask].reshape(y.shape[0], 2).cuda()
        correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    val_loss /= (len(dataloader) * len(X))
    correct /= (len(dataloader) * len(X))

    return val_loss, correct










    



