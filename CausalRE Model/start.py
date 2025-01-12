import torch
import argparse
import yaml
from pathlib import Path
from types import SimpleNamespace

from modules.train import Trainer
from modules.config_manager import Config



def create_parser():
    '''
    create the arg parser, allows specifying the config file and the log file location
    '''
    parser = argparse.ArgumentParser(description="Model")
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to config file")
    return parser




def load_config(config_path, app_path, device):
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
    
    #make the config object
    main_configs = Config(config_dict)
    main_configs.update(dict(
        app_path   = app_path,
        device     = device
        ))
    
    return main_configs




if __name__ == "__main__":
    '''
    This is the main kick off code for training
    setup the arg parser and then read in the configs
    then instantiate the trainer which takes in the config and internally sets up the model, reads the dataset, and trains/evals/infers the model
    When we call the model, we call it in model.train() mode for train and model.eval() with no grad for eval (val and test)
    We have a 3rd mode model.eval(), and no grad with no labels which is used for the prediction class, see comments below

    NOTE:
    I have not coded the prediction no labels code yet
    This will not use the train class, it will use the predict class, some key differences will be:
    - our input data only has one split, call it 'predict', not train, val, test
    - the predict split will only have tokens and potentiall schema no spans/relations
    - we thus create one dataloader with no shuffle
    - we also do not do negative sampling as we have only negative sampels, all possible spans/rels are negative sampels, we just pass them all
    - we can reuse the same preprocessing code with some adjustments to disable neg sampling
    '''
    #get the configs
    parser = create_parser()
    args = parser.parse_args()
    
    #load the config file to a config object
    app_path = Path(__file__).parent
    config_path = app_path / args.config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main_configs = load_config(config_path, app_path, device)
    
    #make the Trainer object to orchestrate the training/inference
    trainer = Trainer(main_configs)
    #run the trainer
    trainer.run()

