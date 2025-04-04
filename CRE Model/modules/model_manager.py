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
import copy, shutil


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
    def __init__(self, main_configs):
        self.main_configs = main_configs
        self.config = main_configs.as_namespace

    def copy_model_to_drive(self, local, drive, name):
        """
        Copies a saved model (.pt) from a local Colab path to Google Drive.

        Args:
            local (str or Path): location of the local model (e.g., '/content')
            drive (str or Path): Destination folder in Google Drive (e.g., '/content/drive/MyDrive/models')
            name (str): Base filename (without .pt extension)
        """
        Path(drive).mkdir(parents=True, exist_ok=True)
        local_path = str(Path(f'{local}/{name}.pt'))
        dest_path = str(Path(f'{drive}/{name}.pt'))
        shutil.copy(local_path, dest_path)

        self.config.logger.write(f"Model copied to Google Drive at: {dest_path}")

        
    def save_pretrained(self, model, save_folder, name):
        """Save the model parameters and config to the specified directory"""
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        model_filename = name  + ".pt"
        model_path = str(Path(f'{save_folder}/{model_filename}'))
        #torch.save(model.state_dict(), model_path)
        torch.save(model, model_path)
        self.config.logger.write(f"Model saved to {model_path}")


    def load_pretrained(self, model_path):
        """Load model weights from the specified path"""
        #state_dict = torch.load(model_path)
        #model.load_state_dict(state_dict)
        model = torch.load(model_path, weights_only=False)
        self.config.logger.write(f"Model loaded from {model_path}")
        return model


    def get_model(self, device=None):
        '''
        This sets up the model
        loads the model if one is given or it creates a new one
        '''
        self.config.logger.write('Making the model')

        if device is None:
            device = self.config.device
        
        #make a new model only if we are in train mode and we specifcally say do not load
        if not self.config.model_load and self.config.run_type == 'train':
            model = Model(self.config).to(device)
            
        #load the pretrained model if required (predict => always load a new model. train => only if we ask for it)
        elif ((self.config.run_type == 'train' and self.config.model_load) or (self.config.run_type == 'predict')) and (self.config.model_name and self.config.model_name.strip().lower() not in ["none", ""]):
            #get the safe params to overwrite to the loaded model namespace
            safe_params = [
                            'run_type', 
                            'predict_split',
                            'random_seed',
                            'eval_step_display',
                            'log_step_data',
                            'model_folder',
                            'model_name',
                            'model_save',
                            'model_load',
                            'model_min_save_steps',
                            'mode_early_stopping',
                            'log_folder',
                            'print_log',
                            'data_path',
                            'data_format',
                            'num_steps',
                            'train_batch_size',
                            'accumulation_steps',
                            'eval_batch_size',
                            'eval_every',
                            'opt_type',
                            'opt_weight_decay',
                            'lr_encoder_span',
                            'lr_encoder_rel',
                            'lr_encoder_marker',
                            'lr_others',
                            'warmup_ratio',
                            'scheduler_type',
                            'loss_reduction',
                            'clear_tensor_steps',
                            'grad_clip',
                            'collect_grads',
                            'span_loss_mf',
                            'rel_loss_mf',
                            'span_force_pos: always',
                            'force_pos_step_limit',
                            'rel_force_pos',
                            'span_neg_sampling_limit',
                            'span_neg_sampling',
                            'rel_neg_sampling_limit',
                            'rel_neg_sampling'
                        ]
            #get teh safe param values, read from teh main_configs
            safe_update_dict = {k: v for k, v in self.main_configs.to_dict.items() if k in safe_params}
            #load the pre-trained model and read the non-safe params from it to update the current config namespace
            model_path = str(Path(f'{self.config.app_path}/{self.config.model_folder}/{self.config.model_name}.pt'))
            try:
                model = self.load_pretrained(model_path).to(device)
                #remember self.config is not connected to model.config as the model init creates a copy of it
            
            except Exception as e:
                raise RuntimeError(f'Error loading the model: {e}') from e
            
            #update the loaded model config namespace with the safe params
            try:
                for key, value in safe_update_dict.items():
                    setattr(model.config, key, value)
            except Exception as e:
                raise RuntimeError(f'Error applying external safe params to loaded model config: {e}') from e

            #Reflect non-safe params from model.config to self.config and main_configs
            loaded_config_dict = vars(model.config)  # or model.config.__dict__ if needed
            non_safe_update_dict = {k: v for k, v in loaded_config_dict.items() if k not in safe_params}
            try:
                self.main_configs.update(non_safe_update_dict)
                self.config = self.main_configs.as_namespace
            except Exception as e:
                raise RuntimeError(f'Error updating external config with non-safe params from the loaded model: {e}') from e

        return model

##########################################################
##########################################################
##########################################################
##########################################################

class Optimizer:
    def __init__(self, config, model):
        self.config = config
        lr_others = float(config.lr_others)
        weight_decay = float(config.opt_weight_decay)

        if config.opt_type == 1:
            self.optimizer = torch.optim.AdamW(model.parameters(), 
                                               lr=lr_others, 
                                               weight_decay=weight_decay)
        else:
            opt_params = self.get_optimizer_params(model, config)

            self.optimizer = torch.optim.AdamW(opt_params, 
                                               lr=lr_others, 
                                               weight_decay=weight_decay)


    def get_optimizer_params(self, model, config):
        """
        Prepare optimizer parameters with specific settings for different model components.
        - `lr_encoder` for transformer encoder parameters.
        - `lr_others` for all other parameters.
        - Optionally freeze the transformer encoder.
        """
        lr_encoder_span = float(config.lr_encoder_span)
        lr_encoder_rel = float(config.lr_encoder_rel)
        lr_encoder_marker = float(config.lr_encoder_marker)
        lr_others = float(config.lr_others)
        param_groups = []

        encoder_params_span = list(model.transformer_encoder_span.parameters())
        encoder_params_marker = list(model.transformer_encoder_marker.parameters())
        if not self.config.bert_shared_unmarked_span_rel:
            encoder_params_rel = list(model.transformer_encoder_rel.parameters())
        # Handle freezing and learning rate assignment separately for each encoder
        if config.freeze_encoder:
            for param in encoder_params_span:
                param.requires_grad = False
            for param in encoder_params_marker:
                param.requires_grad = False
            if not self.config.bert_shared_unmarked_span_rel:
                for param in encoder_params_rel:
                    param.requires_grad = False
            processed_params = set()  # No encoder parameters will be trained

        else:
            # Assign different learning rates if needed
            param_groups.append({"params": encoder_params_span, "lr": lr_encoder_span})
            param_groups.append({"params": encoder_params_marker, "lr": lr_encoder_marker})
            # Track which parameters are processed
            processed_params = set(encoder_params_span + encoder_params_marker)
            #deal with bert rel params if not shared
            if not self.config.bert_shared_unmarked_span_rel:
                param_groups.append({"params": encoder_params_rel, "lr": lr_encoder_rel})
                processed_params.update(encoder_params_rel)

        # Set specific learning rate for other parameters
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


    def get_optimizer_params_old(self, model, config):
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
