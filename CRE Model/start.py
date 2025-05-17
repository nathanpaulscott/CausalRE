import torch
import argparse, yaml, logging, os, re
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

from modules.train import Trainer
from modules.config_manager import Config
from modules.logger import Logger
from modules.utils import join_paths, get_filename_no_extension



def create_parser():
    '''
    create the arg parser, allows specifying the config file and the log file location
    '''
    parser = argparse.ArgumentParser(description="Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="path to config file")
    return parser




def load_config(config_path):
    """
    Load a YAML configuration file to a config object containing a namespace
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"The configuration file {config_path} does not exist.")
    
    try:
        with Path(config_path).open("r") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    
    if config_dict is None:
        raise ValueError('the config is blank, exiting....')
    
    return config_dict



def save_config(data, config_path):
    with open(config_path, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)



def run_model(config_path):
    """
    Main function to set up and run the training process.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    #for debugging only
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #os.environ['TORCH_USE_CUDA_DSA'] = "1"

    # Load the configuration
    try:
        app_path = str(Path(__file__).parent)
        full_config_path = join_paths(app_path, config_path)
        config_dict = load_config(full_config_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        #set the full model file name and full log file name
        model_file_name = f"{config_dict['model_name']}.pt"    
        if 'log_name' in config_dict:
            log_file_name = f"{config_dict['log_name']}.log"
        else:
            log_file_name = f"log_{config_dict['model_name']}_{timestamp}.log"

        #set the model folders
        model_colab_folder = "/content"     #for the colab/drive scenario
        model_colab_path = join_paths(model_colab_folder, model_file_name)
        model_full_folder = join_paths(app_path, config_dict['model_folder'])

        # Create config object and update settings
        main_configs = Config(config_dict)
        main_configs.update({
            'app_path':           app_path, 
            'config_path':        config_path,
            'log_file_name':      log_file_name,
            'model_file_name':    model_file_name,
            'device':             device, 
            'torch_precision':    torch.float16 if config_dict['num_precision'] == 'half' else torch.float32,
            'num_limit':          1e4 if config_dict['num_precision'] == 'half' else 1e10,
            'model_colab_folder': model_colab_folder,     #for the colab/drive scenario
            'model_colab_path':   model_colab_path,
            'model_full_folder':  model_full_folder
        })
    except Exception as e:
        raise Exception(f"Failed to load and parse the configuration: {e}")

    # Set working directory and initialize logger
    config = main_configs.as_namespace
    os.chdir(config.app_path)
    #set up the logger
    logger = Logger(config, enable_console_output=config.print_log)
    logger.write('Start...')
    #add logger to main_configs
    main_configs.update({'logger': logger})

    # Initialize and run the trainer
    trainer = Trainer(main_configs)
    trainer.run()

    logger.write('Done...')
    
    #close the logger
    logger.close()
    #remove objects
    del trainer
    del logger
    del main_configs



if __name__ == "__main__":
    '''
    This is the main kick off code for training
    setup the arg parser and then read in the configs
    then instantiate the trainer which takes in the config and internally sets up the model, reads the dataset, and trains/evals/infers the model
    When we call the model, we call it in model.train() mode for train and model.eval() with no grad for eval (val and test)
    We have a 3rd mode model.eval(), and no grad with no labels which is used for the prediction class, see comments below
    '''
    #get the configs
    parser = create_parser()
    args = parser.parse_args()

    run_model(args.config)


