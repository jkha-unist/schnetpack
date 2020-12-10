import os

import schnetpack as spk
import torch
from torch.optim import Adam


__all__ = ["get_trainer", "simple_loss_fn", "tradeoff_loss_fn", "get_metrics"]


def get_trainer(args, model, train_loader, val_loader, metrics):
    # setup optimizer
    # filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=args.lr)

    # setup hook and logging
    hooks = [spk.train.MaxEpochHook(args.max_epochs)]
    if args.max_steps:
        hooks.append(spk.train.MaxStepHook(max_steps=args.max_steps))

    schedule = spk.train.ReduceLROnPlateauHook(
        optimizer=optimizer,
        patience=args.lr_patience,
        factor=args.lr_decay,
        min_lr=args.lr_min,
        window_length=1,
        stop_after_min=True,
    )
    hooks.append(schedule)

    if args.logger == "csv":
        logger = spk.train.CSVHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)
    elif args.logger == "tensorboard":
        logger = spk.train.TensorboardHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)

    # setup loss function
    loss_fn = get_loss_fn(args)

    # setup trainer
    trainer = spk.train.Trainer(
        args.modelpath,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        checkpoint_interval=args.checkpoint_interval,
        keep_n_checkpoints=args.keep_n_checkpoints,
        hooks=hooks,
    )
    return trainer


def get_loss_fn(args):
    derivative = spk.utils.get_derivative(args)
    if derivative is None:
        return simple_loss_fn(args)
    if args.loss == "l2":
        print("Loss function is set to L2 function.")
        return tradeoff_loss_fn(args, derivative)
    if args.loss == "phase_less_loss":
        print("Loss function is set to phase less loss.")
        return phase_less_loss(args, derivative)
    if args.loss == "w_phase_less_loss":
        print("Loss function is set to weighted phase less loss.")
        return w_phase_less_loss(args, derivative)


def simple_loss_fn(args):
    def loss(batch, result):
        diff = batch[args.property] - result[args.property]
        diff = diff ** 2
        err_sq = torch.mean(diff)
        return err_sq

    return loss


def tradeoff_loss_fn(args, derivative):
    def loss(batch, result):
        
        nat = torch.sum(batch["_atom_mask"],1)
        
        diff = batch[args.property] - result[args.property]
        diff = diff ** 2

        der_diff = batch[derivative] - result[derivative]
        der_diff = der_diff ** 2

#        err_sq = args.rho * torch.mean(diff.view(-1)) + (1 - args.rho) * torch.mean(
#            der_diff.view(-1)
#        )
        
        err_sq = args.rho * torch.mean(diff.view(-1)) + (1 - args.rho) * torch.mean((torch.sum(
            der_diff,(1,2)
        )/3.0/nat)).view(-1)
        
        return err_sq

    return loss

def phase_less_loss(arg, derivative):
    def loss(batch, result):
        
        nat = torch.sum(batch["_atom_mask"],1)

        diff1 =        batch[arg.property] - result[arg.property]
        diff2 = -1.0 * batch[arg.property] - result[arg.property]
        diff1 = diff1 ** 2
        diff2 = diff2 ** 2
        
        der_diff1 =        batch[derivative] - result[derivative]
        der_diff1 = der_diff1 ** 2
        der_diff2 = -1.0 * batch[derivative] - result[derivative]
        der_diff2 = der_diff2 ** 2
        
        l_1 = arg.rho*diff1.view(-1) + (1.0 - arg.rho)*torch.sum(der_diff1, (1,2))/3.0/nat
        l_2 = arg.rho*diff2.view(-1) + (1.0 - arg.rho)*torch.sum(der_diff2, (1,2))/3.0/nat
       
        err_sq = torch.mean(torch.min(l_1,l_2).view(-1))
        
        return err_sq
    
    return loss

def w_phase_less_loss(arg, derivative):
    def loss(batch, result):
        
        diff1 =        batch[arg.property] - result[arg.property]
        diff2 = -1.0 * batch[arg.property] - result[arg.property]
        diff1 = diff1 ** 2
        diff2 = diff2 ** 2
        
        der_diff1 =        batch[derivative] - result[derivative]
        der_diff1 = der_diff1 ** 2
        der_diff2 = -1.0 * batch[derivative] - result[derivative]
        der_diff2 = der_diff2 ** 2
        
        l_1 = (9.0 * torch.exp(-1.0*batch[arg.property]**2/0.01).view(-1) + 1.0) * (arg.rho*diff1.view(-1) + \
                (1.0 - arg.rho)*torch.mean(der_diff1, (1,2)))
        l_2 = (9.0 * torch.exp(-1.0*batch[arg.property]**2/0.01).view(-1) + 1.0)* (arg.rho*diff2.view(-1) + \
                (1.0 - arg.rho)*torch.mean(der_diff2, (1,2)))
       
        err_sq = torch.mean(torch.min(l_1,l_2).view(-1))
        
        return err_sq
    return loss

def get_metrics(args):
    # setup property metrics
    metrics = [
        spk.train.metrics.MeanAbsoluteError(args.property, args.property),
        spk.train.metrics.RootMeanSquaredError(args.property, args.property),
    ]

    # add metrics for derivative
    derivative = spk.utils.get_derivative(args)
    if derivative is not None:
        metrics += [
            spk.train.metrics.MeanAbsoluteError(
                derivative, derivative, element_wise=True
            ),
            spk.train.metrics.RootMeanSquaredError(
                derivative, derivative, element_wise=True
            ),
        ]

    return metrics
