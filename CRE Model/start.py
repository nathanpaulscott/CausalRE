import torch
import argparse, yaml, logging, os
from pathlib import Path
from types import SimpleNamespace

from modules.train import Trainer
from modules.config_manager import Config
from modules.logger import Logger




def create_parser():
    '''
    create the arg parser, allows specifying the config file and the log file location
    '''
    parser = argparse.ArgumentParser(description="Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser




def load_config(config_path):
    """
    Load a YAML configuration file to a config object containing a namespace
    """
    if not config_path.exists():
        raise FileNotFoundError(f"The configuration file {str(config_path)} does not exist.")
    
    try:
        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    
    if config_dict is None:
        raise ValueError('the config is blank, exiting....')
    
    return config_dict




if __name__ == "__main__":
    '''
    This is the main kick off code for training
    setup the arg parser and then read in the configs
    then instantiate the trainer which takes in the config and internally sets up the model, reads the dataset, and trains/evals/infers the model
    When we call the model, we call it in model.train() mode for train and model.eval() with no grad for eval (val and test)
    We have a 3rd mode model.eval(), and no grad with no labels which is used for the prediction class, see comments below
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'

    #for debugging only
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ['TORCH_USE_CUDA_DSA'] = "1"

    #get the configs
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        #load the config file to a dict
        app_path = Path(__file__).parent
        config_path = app_path / args.config
        config_dict = load_config(config_path)
        #make the config object
        main_configs = Config(config_dict)
        main_configs.update({'app_path': app_path, 
                             'device': device, 
                             'torch_precision': torch.float16 if config_dict['num_precision'] == 'half' else torch.float32})
    except Exception as e:
        raise Exception(f"Failed to load and parse the configuration: {e}")

    #make the config as namespace and set the cwd
    config = main_configs.as_namespace
    os.chdir(config.app_path)
    #set up the logger
    log_folder = str(config.app_path / config.log_folder)
    logger = Logger(log_folder, enable_console_output=config.print_log)
    logger.write('Start...')
    #add logger to main_configs
    main_configs.update({'logger': logger})

    #make the Trainer object to orchestrate the training/inference
    trainer = Trainer(main_configs)
    #run the trainer
    trainer.run()





'''
-------------------------------

TO DO
-----------------------
working on conll04 with similar params to spert

1) I have profiled it and fixed the worst parts, will leave the profile code in for now, take out later
make_rel_reps is a bit slow, but it is oe of the worst parts, I have already optimised it, coudl take another look, but it is the one thing that I think I have to live with

2) For a spert like test need to set the max_span_width to 10 only and mess aroudn with other params like rel gen method etc...
this rel percentile stuff is not tested at all

############################################################33
3) now start looking at the metrics, why do they look weird, is my code actually working?  I see the metrics look all the same (acc, P, R, F1)???
ok I think there are some concerns with neg sampling, alignment and various things

in spert we neg sampled spans and rels as we had all poss spans and rels up front.  We then used that mask of pos cases and seleted neg cases to filter the spans/rels to use for metrics
I do not think we can do that here due to the filtering layers and also for rels it is not knwon until runtime
Thus for spans, even if we neg sample, we sample another subset later for filtering, for rels we do not neg sample, we just filter a subset and potentially not even teacher force all pos cases....

neg sampling
--------------
we are only using neg sampling for spans, I think however it is not necessary as we are using the filtering layer to shortlist spans to cand_spans
Thus I want to put in a param to use span neg sampling or not (it will just affect the span_masks which in turn affect cand_span_masks)

span loss
-------------
when calculating span loss we determine the spans preds vs span labels, thus the mask indicates which spans to consider for the loss, so removing eg sampling will have an impact

span metrics
----------------
we only consider the cand_span_ids in cand_span_masks for the metrics
i.e. labels => the actual pos labels + and false positives from preds
     preds => TP and FN from pos labels + FP from preds

     
Rels it is even more complex....
Need to contemplate this, it is convoluted, need to really think about it

############################################################33
'''