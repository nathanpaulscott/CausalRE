from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_inverse_sqrt_schedule,
)
import torch
import numpy as np
import os, re
from pathlib import Path
from typing import Tuple, List, Dict, Union
from types import SimpleNamespace
from tqdm import tqdm

###############################################
#custom imports
from .model import Model
from .data_processor import DataProcessor
from .utils import import_data, load_from_json, save_to_json
from .evaluator import Evaluator
from .predictor import Predictor
from .validation import config_validator



class Trainer:
    '''
    This is the Trainer class that brings in the config and basically orchestrates everything 
    It is used for prediction and training   
    '''
    def __init__(self, config):
        self.config = config


    ################################################
    #get the model
    ################################################
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
            model = self.load_pretrained(model, self.config.model_path).to(device)

            #update the params of the trained model with the current config except for these keep params which need to follow the pretrained model
            current_config = vars(self.config).copy()
            loaded_config = vars(model.config)
            #overwrite all loaded params with the self.config properties except for these ones
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
                           'ffn_ratio',
                           'scorer']
            #Overwrite with the keep params
            for param in keep_params:
                if param in loaded_config:
                    current_config[param] = loaded_config[param]
            # Write the updated config back to the model
            model.config = SimpleNamespace(**current_config)        

        #return the model
        return model



    ################################################
    #get the optimiser
    ################################################
    def get_optimizer(self, model):
        """
        Sets learning rates for the encoder and all other layers, with an option to freeze the encoder.
        """
        lr_encoder = float(self.config.lr_encoder)
        lr_others = float(self.config.lr_others)
        freeze_encoder = self.config.freeze_encoder

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



    ################################################
    #get the scheduler
    ################################################
    def get_scheduler(self, optimizer, num_warmup_steps, num_steps):
        '''
        Setup the learning rate scheduler
        '''
        scheduler_type = self.config.scheduler_type, 
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



    ##########################################################################
    #TRAIN/EVAL
    ##########################################################################
    def train_loop(self, model, optimizer, loaders):
        '''
        This is the training and eval loop
        '''
        #read some params from config
        device = self.config.device
        num_steps = self.config.num_steps
        pbar = tqdm(range(num_steps))
        warmup_ratio = self.config.warmup_ratio
        num_warmup_steps = int(num_steps * warmup_ratio) if warmup_ratio < 1 else int(warmup_ratio)
        eval_every = self.config.eval_every
        save_total_limit = self.config.save_total_limit
        log_dir = self.config.log_dir

        #get the optimiser and scheduler
        optimizer = self.get_optimizer(model)
        scheduler = self.get_scheduler(optimizer, num_warmup_steps, num_steps)
        scaler = torch.cuda.amp.GradScaler()

        #get the loaders
        train_loader = loaders['train']
        val_loader = loaders['val']
        test_loader = loaders['test']
        #make an infinitely iterable from train loader
        iter_loader_inf = self.load_loader(train_loader, device, infinite=True)

        for step in pbar:
            optimizer.zero_grad()
            #get next batch and run through model
            x = next(iter_loader_inf)
            with torch.cuda.amp.autocast(dtype=torch.float16):    #forcing data to half precision
                model.train()
                result = model(x, step=step)
                loss = result['loss']
            #skip if loss is NaN
            if torch.isnan(loss).any():
                print(f"Warning: NaN loss detected, not processing step {step}")
                continue

            #backpropagates and runs the optimiser and scheduler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            #writes status to the screen
            description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
            pbar.set_description(description)

            #runs eval and saves the model
            if (step + 1) % eval_every == 0:
                #run the eval loop
                result = self.eval_loop(model, val_loader, device, step=step, msg='interim_val')
                #save the model
                checkpoint = f'model_{step + 1}'
                self.save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)

        #run the final eval and test loop
        result_eval = self.eval_loop(model, val_loader, device, step=None, msg='final_val')
        result_test = self.eval_loop(model, test_loader, device, step=None, msg='final_test')
        #do some shit with the results here
        #save the model
        checkpoint = f'model_{step + 1}'
        self.save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)


    def print_metrics_summary(self, result, msg):
        print(f'Eval_type: {msg}\n-------------------------------')
        print(f"Span Metrics:    {result['span_metrics']['msg']}")
        print(f"Rel Metrics:     {result['rel_metrics']['msg']}")
        print(f"Rel_mod Metrics: {result['rel_mod_metrics']['msg']}")
        print('-------------------------------')


    def eval_loop(self, model, data_loader, device, step=None, msg=None):
        '''
        The run_type param here will be 'train' so we have labels and calculate loss
        the only difference is that the loss is disconnected from the graph as the model is run in .eval mode
        '''
        #make the predictor and evaluators
        predictor = Predictor(self.config)
        evaluator = Evaluator(self.config)
        
        iter_loader = self.load_loader(data_loader, device, infinite=False)
        model.eval()
        with torch.no_grad():
            for x in iter_loader:
                result = model(x, step=step)
                #get preds => the predicted positive cases
                #NOTE: the rel_preds are the full rels with the span start,end,type info
                span_preds, rel_preds = predictor.predict(result)
                #prepare and add the batch of preds and labels to the evaluator
                evaluator.prep_and_add_batch(span_preds, rel_preds, x['spans'], x['relations'])
        
        #run the evaluator on whole dataset results and return a dict with the metrics and preds
        result = evaluator.evaluate()
        self.print_metrics_summary(result, msg)
        return result
            


    def make_full_relations_from_raw_annotations(self, x):
        '''
        #THIS IS BALL PARK FOR EXTRACTING THE RAW ANNOTATED DATA INTO FULL RELATIONS
        #THIS IS BALL PARK FOR EXTRACTING THE RAW ANNOTATED DATA INTO FULL RELATIONS
        #THIS IS BALL PARK FOR EXTRACTING THE RAW ANNOTATED DATA INTO FULL RELATIONS
        This converts the span and relation raw annotations (both being list of list of tuples)
        to full relation labels which is a list of list tuples of the format:
        [[(head start, head end, head type, tail start, tail end, tail type, rel_type), (...), (...)...]]
        '''
        batch_spans =  x['spans']
        batch_rels = x['relations']
        batch = len(spans)
        batch_rels_full = [[] for i in range(batch)]
        for i in range(batch):
            rels = batch_rels[i]
            spans = batch_spans[i]
            for rel in rels:
                full_rel = (spans[rel[0]], spans[rel[1]], rel[2])
                batch_rels_full[i].append(full_rel)
        return batch_rels_full



    ##########################################################################
    #PREDICT
    ##########################################################################
    def predict_loop(self, model, loaders):
        '''
        This is the predict loop, the run_type param is 'predict' so the model will run without labels and not calculate loss
        The loss returned from the model will be None
        '''
        #read some params from config
        device = self.config.device
        predict_loader = loaders['predict']
        log_dir = self.config.log_dir
        pred_thd = self.config.predict_threshold
        pred_conf = self.config.predict_confidence

        iter_loader = self.load_loader(predict_loader, device, infinite=False)
        preds = []
        pos_labels = []
        model.eval()
        with torch.no_grad():
            for x in iter_loader:
                result = model(x)
                
                '''
                Here, implement your logic to process outputs, which contain logits + associated data only
                so we would report:
                - predictions:
                    - spans (start, end, type, potentially show the actual textual span)
                    - rels (head_idx, tail_idx, type)
    
                I am not sure what this returns right now if anything

                '''

                #Get span and rel preds
                spans, rels = self.predictor(x, result)
                preds.extend(batch_predictions)
                pos_labels.extend(self.get_full_relations(x))
        
        return spans, rels




    predictor = 



    ################################################
    #helper functions
    ################################################
    ################################################
    def load_loader(self, data_loader, device, infinite=True):
        """
        Generator function to endlessly yield batches from the data loader, optionally moving them to a specified device,
        restarting from the beginning once all batches have been yielded.

        Args:
            data_loader (DataLoader): The DataLoader from which to fetch data.
            device (str or torch.device): The device to which tensors in the batches should be moved.
            infinite (bool): If True, yields batches indefinitely. If False, yields batches once through the data loader.

        Yields:
            batch (dict): A batch from the data_loader, with all tensors moved to the specified device.
        """
        while True:
            for batch in data_loader:
                # Move each tensor in the batch to the specified device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                yield batch
            if not infinite:
                break



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


    def create_type_mappings(self, types):
        """
        Creates mappings from type names to IDs and vice versa for labeling tasks.

        This function generates dictionaries to map type names to unique integer IDs and vice versa.  
        for unilabels => idx 0 is for the none type (the negative case)
        for multilabels => there is no none type in the types as it is not explicitly given, so idx 0 is for the first pos type

        Parameters:
        - types (list of str): The list of type names for which to create mappings.

        Returns:
        - tuple of (dict, dict): 
            - The first dictionary maps type names to IDs.
            - The second dictionary maps IDs back to type names.
        """
        # Create type to ID mapping
        type_to_id = {t: i for i, t in enumerate(types)}
        # Create ID to type mapping
        id_to_type = {i: t for t, i in type_to_id.items()}

        return type_to_id, id_to_type



    def make_type_mappings_and_span_ids(self):
        """
        Initializes and configures the type mappings and identifier settings for span and relationship types
        within the model's configuration. This method sets several configuration properties related to span
        and relationship types, including creating mapping dictionaries and a list of all possible spans.

        Updates the following in self.config:
        - span_types => adds the none type to the start of the list for unilabels case
        - rel_types => adds the none type to the start of the list for the unilabels case
        - num_span_types: Number of span types   (will be pos and neg types for unilabel and just pos types for multilabel)
        - num_rel_types: Number of relationship types   (will be pos and neg types for unilabel and just pos types for multilabel)
        - s_to_id: Dictionary mapping from span types to their indices.
        - id_to_s: Dictionary mapping from indices to span types.
        - r_to_id: Dictionary mapping from relationship types to their indices.
        - id_to_r: Dictionary mapping from indices to relationship types.
        - all_span_ids: A list of all possible spans generated from the span types.

        No parameters are required as the method operates directly on the class's config attribute.

        NOTE:
        for span/rel_labels == 'unilabels' => span/rel_types will include the none type at idx 0, num_span/rel_types includes the none type
        for span/rel_labels == 'multilabels' => there is no none type in the types as it does not need to be explicitly given, so idx 0 is for the first pos type, num_span/rel_types DOES NOT include the none type
        """
        #add the negative type (none type) for the unilabel case, will take up idx 0
        #add the none type to start of the types lists, it will not be in the list as validation checks would have rejected it
        #NOTE: for the multilabel case the none_span/rel is not required explicitly in types
        if self.config.span_labels == 'unilabel':
            self.config.span_types = ['none'] + self.config.span_types
        if self.config.rel_labels == 'unilabel':
           self.config.rel_types = ['none'] + self.config.rel_types

        #get the num span and rel total types
        #unilabels => will be pos and neg types
        #multilabels => just pos types
        self.config.num_span_types = len(self.config.span_types)
        self.config.num_rel_types  = len(self.config.rel_types)

        self.config.s_to_id, self.config.id_to_s = self.create_type_mappings(self.config.span_types)
        self.config.r_to_id, self.config.id_to_r = self.create_type_mappings(self.config.rel_types)
        
        #get all span_ids in seq_len
        self.config.all_span_ids = self.make_all_possible_spans()



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


    def load_and_prep_data(self):
        '''
        Description
        '''
        #load the predict data
        result = import_data(self.config)
        #read in the data
        dataset                     = result['dataset']
        self.config.span_types      = result['span_types']
        self.config.rel_types       = result['rel_types']

        #validate configs
        config_validator(self.config)

        #add to the config => the span and rel type mapping dicts and the all_possible_spans data
        #adds the none_span/rel types to the possible types for the unilabel case (see function for details)
        self.make_type_mappings_and_span_ids()

        #make the data loaders
        self.data_processor = DataProcessor(self.config)
        return self.data_processor.create_dataloaders(dataset)

    ################################################
    ################################################
    ################################################


    ################################################
    #main orchestration function for training/eval
    ################################################
    def run(self):
        #load and prepare data
        loaders = self.load_and_prep_data()
        #testing
        #self.testing_check_loader(loaders['train'])

        #get the model
        model = self.get_model()

        #kick off the train_loop or predict_loop
        if self.config.run_type == 'train': 
            self.train_loop(model, loaders)

        elif self.config.run_type == 'predict': 
            self.predict_loop(model, loaders)

        else:
            raise Exception("Error - run_type must be either 'train' or 'predict'")




    def testing_check_loader(self, loader):
        '''
        Keep this function, it is for testing
        '''
        x = iter(loader)
        batch = next(x)
        print(batch.keys())
        print(batch)

