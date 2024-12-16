from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_inverse_sqrt_schedule,
)
import torch
import os, re
from typing import Tuple, List, Dict, Union
from types import SimpleNamespace
from tqdm import tqdm

###############################################
#custom imports
from .model import Model
from .data_processor import DataProcessor
from .utils import import_data








'''
NATHAN
--------------------
This sets up the model, data_processor, scheduler, scaler etc..

This sets up the train and eval loops
the pimary info is the batch size and the number of batch steps (num_steps) which will be as many as you want, i.e. it just recycles through the batches if it runs out, as opposed to defining epochs

'''




class Trainer:
    '''
    This is the Trainer class that brings in the config and basically orchestrates everything    
    '''
    def __init__(self, config):
        #put the config instance into a property as a backup
        self.config = config

    def get_model(self, device=None):
        '''
        This sets up the model
        loads the model if one is given or it creates a new one
        '''
        if device is None:
            device = self.config.device
        
        #make the model
        model = Model(self.config).to(device)
        
        #load the pretrained weights is required
        if self.config.model_path and self.config.model_path.strip() not in ["none", ""]:
            #this loads a pretrained model
            model = Model.load_pretrained(self.config.model_path).to(device)

            #NEED TO VERIFY THIS STILL WORKS
            #NEED TO VERIFY THIS STILL WORKS
            #NEED TO VERIFY THIS STILL WORKS
            #update the params of the trained model with the current config except for these keep params which need to follow the pretrained model
            current_config = vars(self.config).copy()
            loaded_config = vars(model.config)
            #overwrite all loaded params with the self.config properties except for these ones
            keep_params = ['model_name', 'name', 'max_span_width', 'hidden_size', 'dropout', 'subtoken_pooling', 'span_mode',
                           'fine_tune', 'max_types', 'max_seq_len', 'num_heads', 'num_transformer_layers', 'ffn_ratio',
                            'scorer']
            #Overwrite with the keep params
            for param in keep_params:
                if param in loaded_config:
                    current_config[param] = loaded_config[param]
            # Write the updated config back to the model
            model.config = SimpleNamespace(**current_config)        

        #return the model and the optimizer objects
        return model




    def get_optimizer(self, model, lr_encoder, lr_others, freeze_token_rep=False):
        """
        Sets learning rates for the encoder and all other layers, with an option to freeze the encoder.

        Parameters:
        - lr_encoder: Learning rate for the transformer encoder layer.
        - lr_others: Learning rate for all other layers.
        - freeze_token_rep: Whether to freeze the transformer encoder layer.
        """
        # Ensure learning rates are float values
        lr_encoder = float(lr_encoder)
        lr_others = float(lr_others)

        param_groups = []
        # Handling the transformer encoder parameters: either freeze or assign learning rate
        transformer_encoder_params = list(model.transformer_encoder_w_prompt.parameters())
        if freeze_token_rep:
            for param in transformer_encoder_params:
                param.requires_grad = False
        else:
            if transformer_encoder_params:  # Ensure there are parameters to add
                param_groups.append({
                    "params": transformer_encoder_params, 
                    "lr": lr_encoder
                })

        # Dynamically assign lr_others to all other parameters
        for name, layer in model.named_children():
            if name != "transformer_encoder_w_prompt" and any(p.requires_grad for p in layer.parameters(recurse=False)):
                param_groups.append({
                    "params": layer.parameters(), 
                    "lr": lr_others
                })

        #make the optimizer
        optimizer = torch.optim.AdamW(param_groups)

        return optimizer



    def init_scheduler(self, scheduler_type, optimizer, num_warmup_steps, num_steps):
        '''
        Setup the learning rate scheduler
        '''
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



    #function def uses type hints (eg. : int)
    #I have not reviewed this code yet
    def save_top_k_checkpoints(self, model: Model, save_path: str, checkpoint: int, top_k: int = 5):
        """
        Save the top-k checkpoints (latest k checkpoints) of a model and tokenizer.

        Parameters:
            model (Model): The model to save.
            save_path (str): The directory path to save the checkpoints.
            top_k (int): The number of top checkpoints to keep. Defaults to 5.
        """
        # Save the current model and tokenizer
        model.save_pretrained(os.path.join(save_path, str(checkpoint)))
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


    ##########################################################################
    #TRAIN
    #getting issues with the scheduler
    #getting issues with the scheduler
    #getting issues with the scheduler
    #getting issues with the scheduler
    #getting issues with the scheduler

    ##########################################################################
    def train(self, model, optimizer, loaders):
        '''
        This is the training function
        '''
        #read some params from config
        device = self.config.device
        train_loader = loaders['train']
        val_loader = loaders['val']
        test_loader = loaders['test']
        num_steps = self.config.num_steps
        pbar = tqdm(range(num_steps))
        warmup_ratio = self.config.warmup_ratio
        num_warmup_steps = int(num_steps * warmup_ratio) if warmup_ratio < 1 else int(warmup_ratio)
        eval_every = self.config.eval_every
        save_total_limit = self.config.save_total_limit
        log_dir = self.config.log_dir

        model.train()
        scheduler = self.init_scheduler(self.config.scheduler_type, optimizer, num_warmup_steps, num_steps)
        iter_train_loader = iter(train_loader)
        scaler = torch.cuda.amp.GradScaler()

        for step in pbar:
            optimizer.zero_grad()
            #NATHAN: fetches a batch and moves it to the GPU
            try:
                x = next(iter_train_loader)
            except StopIteration:
                iter_train_loader = iter(train_loader)
                x = next(iter_train_loader)

            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)

            #NATHAN: this is the core of it, running the data x through the model
            with torch.cuda.amp.autocast(dtype=torch.float16):    #forcing data to half precision
                loss = model(x)

            if torch.isnan(loss).any():
                print("Warning: NaN loss detected")
                continue

            #NATHAN: this backpropagates and runs the optimiser and scheduler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            #NATHAN: writes status to the screen
            description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
            pbar.set_description(description)

            #NATHAN: runs eval, need to put the code in here for eval
            if (step + 1) % eval_every == 0:
                #I think you need to do eval here
                #I think you need to do eval here
                #I think you need to do eval here

                checkpoint = f'model_{step + 1}'
                self.save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)
                #if val_data_dir != "none":
                    #get_for_all_path(model, step, log_dir, val_data_dir)
                model.train()


    def make_all_possible_spans(self):
        '''
        This makes all possible spans indices given:
        - max allowable input word token sequence length (seq_len)
        - max allowable span width in word tokens (self.config.max_span_width)
        
        The output is a list of tuples where each tuple is the start and end token idx for that given span.  
        NOTE: to get the valid spans from a particular seq_len, you can generate a mask with: valid_span_mask = torch.tensor(span_indices, dtype=torch.long)[:, 1] > seq_len
        
        NOTE: the start and end are python list style indices, such that start is the actual word token idx and end is the actual word token idx + 1
        NOTE: remember this calc includes spans that end outside of the sequence length, they are just masked later, this actually makes things easier
              as the num spans here = max_seq_len * max_span_wdith, i.e. each idx in the sequence has max_span_width spans that start on it, a nice useful idea
        '''
        span_indices = []
        for i in range(self.config.max_seq_len):
            span_indices.extend([(i, i + j) for j in range(1, self.config.max_span_width + 1)])
        return span_indices



    def create_type_mappings(self, types: List[str], none_type: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Generic function to get the mapping from type to id for spans or relations
        input are the list of types as well as the none_type which will be placed at id 0
        output are the type string to type id (t_to_id) dict and type id to type string (id_to_t) dict
        """
        #validate types
        if not types or types is None:
            raise Exception('there are no types, can not make mappings')
        #remove none_type from types if it is there
        if none_type in types:
            types.pop(none_type)
        #make the mappings without the none_type
        type_to_id = {t:i for i,t in enumerate(types, start=1)}
        id_to_type = {i:t for t,i in type_to_id.items()}
        #add the none_type at id 0
        type_to_id[none_type] = 0
        id_to_type[0] = none_type
        
        return type_to_id, id_to_type


    def make_type_mappings_and_span_ids(self):
        '''
        add to the config => the span and rel type mapping dicts and the all_possible_spans data
        '''
        self.config.none_span = 'none_span'
        self.config.none_rel = 'none_rel'
        self.config.s_to_id, self.config.id_to_s = self.create_type_mappings(self.config.span_types, self.config.none_span)
        self.config.r_to_id, self.config.id_to_r = self.create_type_mappings(self.config.rel_types, self.config.none_rel)
        self.config.all_span_ids = self.make_all_possible_spans()


    def check_loader(self, loader):
        '''
        temp function for dataset checking        
        '''
        x = iter(loader)
        batch = next(x)
        print(batch.keys())
        print(batch)



    def run(self):
        #load the training, val, test data
        data_path = self.config.app_path / self.config.data_path
        dataset, self.config.span_types, self.config.rel_types = import_data(str(data_path))
        #add to the config => the span and rel type mapping dicts and the all_possible_spans data
        self.make_type_mappings_and_span_ids()
        #make the data loaders
        self.data_processor = DataProcessor(self.config)
        loaders = self.data_processor.create_dataloaders(dataset)
        #testing
        #self.check_loader(loaders['train'])
        #get the model
        model = self.get_model()
        #get the optimiser
        optimizer = self.get_optimizer(model, 
                                       self.config.lr_encoder, 
                                       self.config.lr_others, 
                                       freeze_token_rep=self.config.freeze_token_rep)
        #kick off the training
        self.train(model, 
                   optimizer, 
                   loaders)




#I think I am ready to test this