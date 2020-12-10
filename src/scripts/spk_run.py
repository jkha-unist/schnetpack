#!/usr/bin/env python3
import os
import torch
import logging
import schnetpack as spk
from schnetpack.utils import (
    get_dataset,
    get_metrics,
    get_loaders,
    get_statistics,
    get_model,
    get_trainer,
    ScriptError,
    evaluate,
    setup_run,
    get_divide_by_atoms,
)
from schnetpack.utils.script_utils.settings import get_environment_provider
from schnetpack.utils.script_utils.parsing import build_parser


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def main(args):

    # setup
    '''
    Read input arguments, ex. # of features, cutoff, ....
    Set device.
    '''
    train_args = setup_run(args) # spk/utils/script_utils/setup.py
    device = torch.device("cuda" if args.cuda else "cpu") # pytorch
    
    # get dataset
    '''
    Set class (simple, ase, torch) that will be used to set the list of neighbor atoms and PBCs
    Get dataset from arguments, (spk.data.AtomsData object)
    '''
    environment_provider = get_environment_provider(train_args, device=device)   
    # spk/utils/script_utils/settings.py
    dataset = get_dataset(train_args, environment_provider=environment_provider) 
    # spk/utils/script_utils/data.py

    # get dataloaders
    '''
    Set dataset split path
    Split dataset into 3 dataset and returns the corresponding dataloaders. (spk.data.AtomsLoader)
    '''
    split_path = os.path.join(args.modelpath, "split.npz")
    train_loader, val_loader, test_loader = get_loaders(
        args, dataset=dataset, split_path=split_path, logging=logging
    ) # spk/utils/script_utils/data.py

    # define metrics
    '''
    metrics = 
      [spk.train.metrics.MeanAbsoluteError(property, property), spk.train.metrics.RootMeanSquaredError(property,property)]
     
    '''
    metrics = get_metrics(train_args) # spk/utils/script_utils/training.py
    print(metrics)
    # train or evaluate
    if args.mode == "train":

        # get statistics
        '''
        Return multiple single atom reference values as a dictionary.(?) (for delta case, atomref = {'delta':None})
        Calculate mean and standard deviation values of property (mean= {'delta':mean}, ...)
               "divide_by_atoms" is predefined in spk/utils/script_utils/settings.py for dataset and properties.
               For custum case, returns true if the "aggregation_mode" is "sum" (for delta case, it's true)
        '''
        atomref = dataset.get_atomref(args.property) # spk/data/atoms.py  
        print(f"atomref= {atomref}")
        print(f"divide_by_atoms= {get_divide_by_atoms(args)}")
        mean, stddev = get_statistics(
            args=args,
            split_path=split_path,
            train_loader=train_loader,
            atomref=atomref,
            divide_by_atoms=get_divide_by_atoms(args), 
            logging=logging,
        ) # spk/utils/script_utils/data.py
        print(f"mean= {mean}")
        print(f"stddev= {stddev}")

        # build model
        '''
        Build a model from selected parameters or load trained model for evaluation, spk.AtomisticModel
        Create trainer, spk.train.Trainer
        Train, spk.train.Trainer.train

        '''
        print(args.contributions)
        model = get_model(args, train_loader, mean, stddev, atomref, logging=logging) # spk/utils/script_utils/model.py

        # build trainer
        logging.info("training...")
        trainer = get_trainer(args, model, train_loader, val_loader, metrics) # spk/utils/script_utils/training.py

        # run training
        trainer.train(device, n_epochs=args.n_epochs) # spk/train/trainer.py
        logging.info("...training done!")

    elif args.mode == "eval":

        # remove old evaluation files
        evaluation_fp = os.path.join(args.modelpath, "evaluation.txt")
        if os.path.exists(evaluation_fp):
            if args.overwrite:
                os.remove(evaluation_fp)
            else:
                raise ScriptError(
                    "The evaluation file does already exist at {}! Add overwrite flag"
                    " to remove.".format(evaluation_fp)
                )

        # load model
        logging.info("loading trained model...")
        model = torch.load(os.path.join(args.modelpath, "best_model"))

        # run evaluation
        logging.info("evaluating...")
        if spk.utils.get_derivative(train_args) is None:
            with torch.no_grad():
                evaluate(
                    args,
                    model,
                    train_loader,
                    val_loader,
                    test_loader,
                    device,
                    metrics=metrics,
                )
        else:
            evaluate(
                args,
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                metrics=metrics,
            )
        logging.info("... evaluation done!")

    else:
        raise ScriptError("Unknown mode: {}".format(args.mode))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "from_json":
        args = spk.utils.read_from_json(args.json_path)

    import argparse

    main(args)
