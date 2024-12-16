import torch
import argparse
import yaml
from pathlib import Path

from modules.train import Trainer
from modules.validation import config_validator




def create_parser():
    '''
    create the arg parser, allows specifying the config file and the log file location
    '''
    parser = argparse.ArgumentParser(description="Model")
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to config file")
    return parser




def load_config_as_namespace(config_path):
    """
    Load a YAML configuration file and convert it into a namespace for easy attribute access.
    Args:
        config_path: Path object: Path to the YAML configuration file.
    Returns:
        argparse.Namespace: A namespace populated with configurations loaded from the file.
    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
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
        return argparse.Namespace()  # Return an empty namespace if config is empty
    
    #validate configs
    err = config_validator(config_dict)
    if err: raise Exception(f'Error: the config file has errors {err}')

    #add the base path to the configs
    return argparse.Namespace(**config_dict)



if __name__ == "__main__":
    '''
    This is the main kick off code
    setup the arg parser and then read in the configs
    then instantiate the trainer which takes in the config and internally sets up the model, reads the dataset, and trains/evals/infers the model
    '''
    #get the configs
    parser = create_parser()
    args = parser.parse_args()
    #load the config file
    app_path = Path(__file__).parent
    config_path = app_path / args.config
    config = load_config_as_namespace(config_path)
    #add key properties to config
    config.app_path = app_path
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #make a Trainer object to orchestrate everything
    trainer = Trainer(config)
    #run it
    trainer.run()




#I am at the dataloaders part, you need to mod all the code in the collate_fn