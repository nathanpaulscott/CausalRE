from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_inverse_sqrt_schedule,
)
import torch
import os, re
from pathlib import Path
from types import SimpleNamespace
import copy


###############################################
#custom imports
from .utils import import_data, load_from_json, save_to_json
from .model import Model





class ModelManager:
    '''
    Description
    '''
    def __init__(self, config):
        self.config = config


    def save_pretrained(self, model, save_directory: str):
        """Save the model parameters and config to the specified directory"""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_directory / "pytorch_model.bin")
        #Optionally save the configuration file
        save_to_json(save_directory / 'config.json')


    def load_pretrained(self, model, model_path):
        """Load model weights from the specified path"""
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")


    def save_top_k_checkpoints(self, model, save_path, checkpoint, top_k = 1):
        """
        Save the most recent top_k models, I have top_k set to 1 by default, so it just saves the most recent model

        Parameters:
            model (Model): The model to save.
            save_path (str): The directory path to save the checkpoints.
            top_k (int): The number of top checkpoints to keep. Defaults to 1.
        """
        # Save the current model and tokenizer
        self.save_pretrained(model, os.path.join(save_path, str(checkpoint)))
        # List all files in the directory
        files = os.listdir(save_path)
        # Filter files to keep only the model checkpoints
        checkpoint_folders = [file for file in files if re.search(r'model_\d+', file)]
        # Sort checkpoint files by modification time (latest first)
        checkpoint_folders.sort(key=lambda x: os.path.getmtime(os.path.join(save_path, x)), reverse=True)
        # Keep only the top-k checkpoints
        for checkpoint_folder in checkpoint_folders[top_k:]:
            checkpoint_folder = os.path.join(save_path, checkpoint_folder)
            checkpoint_files = [os.path.join(checkpoint_folder, f) for f in os.listdir(checkpoint_folder)]
            for file in checkpoint_files:
                os.remove(file)
            os.rmdir(os.path.join(checkpoint_folder))



    def get_model(self, device=None):
        '''
        This sets up the model
        loads the model if one is given or it creates a new one
        '''
        if device is None:
            device = self.config.device
        
        #make the model
        model = Model(self.config).to(device)
        current_config_dict = copy.deepcopy(vars(model.config))

        #load the pretrained weights is required
        if self.config.model_path and self.config.model_path.strip() not in ["none", ""]:
            #this loads a pretrained model
            model = self.load_pretrained(model, self.config.model_path).to(device)

            #only keep these params in teh loaded model params, all others should come from main_configs
            loaded_config_dict = vars(model.config)
            #keep the loaded values of these parameters only, all other prams use the mina_configs settings
            keep_params = ['model_name', 
                           'name', 
                           'max_span_width', 
                           'hidden_size', 
                           'dropout', 
                           'subtoken_pooling', 
                           'span_mode',
                           'fine_tune', 
                           'max_types', 
                           'max_seq_len', 
                           'num_heads', 
                           'num_transformer_layers', 
                           'ffn_ratio']

            #Update the main_configs with the keep params
            update_dict = {param: loaded_config_dict[param] for param in keep_params if param in loaded_config_dict}
            self.config.update(update_dict)

            #revert all model params to the original except the keep params
            for param in keep_params:
                if param in loaded_config_dict:
                    current_config_dict[param] = loaded_config_dict[param]
            #Write the updated config back to the model
            model.config = SimpleNamespace(**current_config_dict)        

        #return the model
        return model






class Optimizer:
    def __init__(self, config, model):
        lr_encoder = float(config.lr_encoder)
        lr_others = float(config.lr_others)
        freeze_encoder = config.freeze_encoder

        param_groups = []
        # Handling the transformer encoder parameters: either freeze or assign learning rate
        trans_encoder_params = list(model.transformer_encoder_w_prompt.parameters())
        if freeze_encoder:
            for param in trans_encoder_params:
                param.requires_grad = False
        else:
            if trans_encoder_params:  # Ensure there are parameters to add
                param_groups.append({"params": trans_encoder_params, "lr": lr_encoder})

        # Dynamically assign lr_others to all other parameters
        for name, layer in model.named_children():
            if name != "transformer_encoder_w_prompt" and any(p.requires_grad for p in layer.parameters(recurse=False)):
                param_groups.append({"params": layer.parameters(), "lr": lr_others})

        #make the optimizer
        optimizer = torch.optim.AdamW(param_groups)

        return optimizer





class Scheduler:
    def __init__(self, config, optimizer, num_warmup_steps, num_steps):
        '''
        Setup the learning rate scheduler
        '''
        scheduler_type = config.scheduler_type, 
        if scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_steps
            )
        elif scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_steps
            )
        elif scheduler_type == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        elif scheduler_type == "polynomial":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_steps
            )
        elif scheduler_type == "inverse_sqrt":
            scheduler = get_inverse_sqrt_schedule(
                optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            raise ValueError(
                f"Invalid scheduler_type value: '{scheduler_type}' \n Supported scheduler types: 'cosine', 'linear', 'constant', 'polynomial', 'inverse_sqrt'"
            )
        return scheduler
    
