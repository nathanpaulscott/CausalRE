import torch
import argparse
import yaml
from pathlib import Path

from modules.train import Trainer




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
    
    config = argparse.Namespace(**config_dict)

    #add the base path to the configs
    return config



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
    #load the config file
    app_path = Path(__file__).parent
    config_path = app_path / args.config
    config = load_config_as_namespace(config_path)
    #add key properties to config
    config.app_path = app_path
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #make the Trainer object to orchestrate the training/inference
    trainer = Trainer(config)
    #run the trainer
    trainer.run()

