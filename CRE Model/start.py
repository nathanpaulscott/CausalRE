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
TO DO
-----------------------
working on conll04 with similar params to spert

** Need to profile the model and find which parts are slowing it down
predictor is very slow, should be able to be sped up
The main issue is the horrible loop in the get_rels code, you must refactor this and vectorize
The spans loop is around 10x faster, probably cause it is 2 layer, while the rel loop is 3 layer

evaluator is fast
make_rel_reps is slow even though I already optimised it, that is really the main stumbling block



For a spert like test need to set the max_span_width to 10 only and mess aroudn with other params like rel gen method etc...
this rel percentile stuff is not tested at all

'''