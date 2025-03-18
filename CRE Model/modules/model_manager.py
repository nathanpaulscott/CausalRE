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
from .utils import load_from_json, save_to_json
from .model import Model
#from .model_spert import Model as model_spert
#from .model_span_marker import Model as model_span_marker



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
        #save_to_json(save_directory / 'config.json')


    def load_pretrained(self, model, model_path):
        """Load model weights from the specified path"""
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        self.config.logger.write(f"Model loaded from {model_path}")


    def save_top_k_checkpoints(self, model, save_path, checkpoint, top_k = 1):
        """
        Save the most recent top_k models, I have top_k set to 1 by default, so it just saves the most recent model

        Parameters:
            model (Model): The model to save.
            save_path (str): The directory path to save the checkpoints.
            top_k (int): The number of top checkpoints to keep. Defaults to 1.
        """
        self.config.logger.write('Saving Checkpoints...')

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
        self.config.logger.write('Making the model')

        if device is None:
            device = self.config.device
        
        #make the model
        if self.config.model_module_name == 'normal':
            model = Model(self.config).to(device)
        
        #elif self.config.model_module_name == 'token_tagger_span_marker':
        #    model = model_token_tagger_span_marker(self.config).to(device)
        
        #elif self.config.model_module_name == 'spert':
        #    model = model_spert(self.config).to(device)
        
        else:
            raise Exception("mode name not supported")
        
        current_config_dict = copy.deepcopy(vars(model.config))

        #load the pretrained weights if required
        if self.config.model_path and self.config.model_path is not None and self.config.model_path.strip() not in ["none", ""]:
            raise Exception("loading the pretrained model his not supported yet")
            #this loads a pretrained model
            model = self.load_pretrained(model, self.config.model_path).to(device)

            #only keep these params in teh loaded model params, all others should come from main_configs
            loaded_config_dict = vars(model.config)
            #keep the loaded values of these parameters only, all other prams use the main_configs settings
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


##########################################################
##########################################################
##########################################################
##########################################################

class Optimizer:
    def __init__(self, config, model):
        lr_encoder = float(config.lr_encoder)
        lr_others = float(config.lr_others)
        freeze_encoder = config.freeze_encoder
        weight_decay = float(config.opt_weight_decay)

        if config.opt_type == 1:
            self.optimizer = torch.optim.AdamW(model.parameters(), 
                                               lr=lr_others, 
                                               weight_decay=weight_decay)
        else:
            opt_params = self.get_optimizer_params_alt(model, config)

            self.optimizer = torch.optim.AdamW(opt_params, 
                                               lr=lr_others, 
                                               weight_decay=weight_decay)


    def get_optimizer_params_alt(self, model, config):
        """
        Prepare optimizer parameters with specific settings for different model components.
        - `lr_encoder` for transformer encoder parameters.
        - `lr_others` for all other parameters.
        - Optionally freeze the transformer encoder.
        """
        lr_encoder = float(config.lr_encoder)
        lr_others = float(config.lr_others)
        param_groups = []

        # Freeze or set specific learning rate for the transformer encoder
        encoder_params = list(model.transformer_encoder.parameters())
        if config.freeze_encoder:
            for param in encoder_params:
                param.requires_grad = False
        else:
            param_groups.append({"params": encoder_params, "lr": lr_encoder})

        # Set specific learning rate for other parameters
        # Exclude encoder parameters by checking if they belong to the encoder module
        processed_params = set(encoder_params) if not config.freeze_encoder else set()
        for module in model.modules():
            # Skip the root model module to prevent double processing
            if module != model:
                module_params = list(module.parameters())
                # Filter out parameters that are already processed
                unique_params = [p for p in module_params if p not in processed_params and p.requires_grad]
                if unique_params:
                    param_groups.append({"params": unique_params, "lr": lr_others})
                    processed_params.update(unique_params)

        return param_groups


    def get_optimizer_params(self, model, config):
        '''
        Sets no weight decay on certain params and specific learning rates on specific params.
        If encoder is frozen, sets requires_grad to False for its parameters.
        '''
        lr_encoder = float(config.lr_encoder)
        lr_others = float(config.lr_others)
        weight_decay = float(config.opt_weight_decay)
        no_decay = {'bias', 'LayerNorm.bias', 'LayerNorm.weight'}
        param_groups = []
        used_params = set()

        # Handle transformer encoder parameters
        if config.freeze_encoder:
            # Freeze all parameters in the encoder
            for param in model.transformer_encoder.parameters():
                param.requires_grad = False
        else:
            encoder_params = list(model.transformer_encoder.named_parameters())
            for name, param in encoder_params:
                if param not in used_params:
                    if any(nd in name for nd in no_decay):
                        param_groups.append({'params': [param], 'lr': lr_encoder, 'weight_decay': 0.0})
                    else:
                        param_groups.append({'params': [param], 'lr': lr_encoder, 'weight_decay': weight_decay})
                    used_params.add(param)

        # Handle all other parameters
        for name, layer in model.named_children():
            if name != "transformer_encoder":
                for param_name, param in layer.named_parameters():
                    if param not in used_params:
                        if any(nd in param_name for nd in no_decay):
                            param_groups.append({'params': [param], 'lr': lr_others, 'weight_decay': 0.0})
                        else:
                            param_groups.append({'params': [param], 'lr': lr_others, 'weight_decay': weight_decay})
                        used_params.add(param)

        return param_groups


    @property
    def return_object(self):
        return self.optimizer
    

##########################################################
##########################################################
##########################################################
##########################################################


class Scheduler:
    def __init__(self, config, optimizer, num_warmup_steps, num_steps):
        '''
        Setup the learning rate scheduler
        '''
        scheduler_type = config.scheduler_type
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
        self.scheduler = scheduler
    
    @property
    def return_object(self):
        return self.scheduler
