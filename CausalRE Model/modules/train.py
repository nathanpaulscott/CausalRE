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
from typing import Tuple, List, Dict, Union
from types import SimpleNamespace
from tqdm import tqdm

###############################################
#custom imports
from .model import Model
from .data_processor import DataProcessor
from .utils import import_data, er_decoder, get_relation_with_span, load_from_json, save_to_json
from .evaluator import Evaluator



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
        transformer_encoder_params = list(model.transformer_encoder_w_prompt.parameters())
        if freeze_encoder:
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



    ################################################
    #save the model code
    ################################################
    def save_top_k_checkpoints(self, model: Model, save_path: str, checkpoint: int, top_k: int = 1):
        """
        Save the most recent top_k models, I have top_k set to 1 by default, so it just saves the most recent model

        Parameters:
            model (Model): The model to save.
            save_path (str): The directory path to save the checkpoints.
            top_k (int): The number of top checkpoints to keep. Defaults to 1.
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

        #set model to train mode
        model.train()
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
                result = model(x, step, mode='train')
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
                result = self.eval_loop(model, val_loader, device, step)
                #save the model
                checkpoint = f'model_{step + 1}'
                self.save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)
                #if val_data_dir != "none":
                    #get_for_all_path(model, step, log_dir, val_data_dir)
                model.train()


        #run the final eval and test loop
        result_eval = self.eval_loop(model, val_loader, device, step)
        result_test = self.eval_loop(model, test_loader, device, step)
        #do some shit with the results here
        #save the model
        checkpoint = f'model_{step + 1}'
        self.save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)




    def eval_loop(self, model, data_loader, device, step):
        model.eval()
        iter_loader = self.load_loader(data_loader, device, infinite=False)
        with torch.no_grad():
            for x in iter_loader:
                result = model(x, step=step, mode='pred_w_labels')
                '''
                Here, implement your logic to process outputs, which contain loss and logits + associated data
                so we would report:
                - loss (potentially need to breakdown loss also)
                - predictions:
                    - spans (start, end, type, potentially show the actual textual span)
                    - rels (head_idx, tail_idx, type)
                - metrics 
    
                I am not sure what this returns right now if anything
                '''
        return 'something'



    ##########################################################################
    #PREDICT
    ##########################################################################
    def predict_loop(self, model, loaders):
        '''
        This is the training function
        '''
        #read some params from config
        device = self.config.device
        predict_loader = loaders['predict']
        log_dir = self.config.log_dir

        model.eval()
        iter_loader = self.load_loader(predict_loader, device, infinite=False)
        with torch.no_grad():
            for x in iter_loader:
                result = model(x, mode='pred_no_labels')
                '''
                Here, implement your logic to process outputs, which contain logits + associated data only
                so we would report:
                - predictions:
                    - spans (start, end, type, potentially show the actual textual span)
                    - rels (head_idx, tail_idx, type)
    
                I am not sure what this returns right now if anything

                '''
        
        return 'something'
    ########################################################################







    #################################################################
    #INTEGRATE THIS!!!!!!
    #INTEGRATE THIS!!!!!!
    #INTEGRATE THIS!!!!!!
    #INTEGRATE THIS!!!!!!
    #INTEGRATE THIS!!!!!!
    ###########################################################################
    #Base Functions, these were in basemodel before, moving to the train class
    ###########################################################################
    def save_pretrained(self, save_directory: str):
        """Save the model parameters and config to the specified directory"""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        # Optionally save the configuration file
        save_to_json(save_directory / 'config.json')


    def load_pretrained(self, model_path):
        """Load model weights from the specified path"""
        state_dict = torch.load(model_path)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")


    def adjust_logits(self, logits, keep):
        """Adjust logits based on the keep tensor."""
        keep = torch.sigmoid(keep)
        keep = (keep > 0.5).unsqueeze(-1).float()
        adjusted_logits = logits + (1 - keep) * -1e9
        return adjusted_logits


    def predict(self, x, threshold=0.5, output_confidence=False):
        """Predict entities and relations."""
        out = self.forward(x, prediction_mode=True)

        # Adjust relation and entity logits
        out["span_logits"] = self.adjust_logits(out["span_logits"], out["keep_span"])
        out["rel_logits"] = self.adjust_logits(out["rel_logits"], out["keep_rel"])

        # Get entities and relations
        spans, rels = er_decoder(x, 
                                 out["span_logits"], 
                                 out["rel_logits"], 
                                 out["topK_rel_idx"], 
                                 out["max_top_k"], 
                                 out["candidate_spans_idx"], 
                                 threshold=threshold, 
                                 output_confidence=output_confidence)
        return spans, rels


    def evaluate(self, eval_loader, threshold=0.5, batch_size=12, rel_types=None):
        self.eval()
        device = next(self.parameters()).device
        all_preds = []
        all_trues = []
        for x in eval_loader:
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            batch_predictions = self.predict(x, threshold)
            all_preds.extend(batch_predictions)
            all_trues.extend(get_relation_with_span(x))
        evaluator = Evaluator(all_trues, all_preds)
        out, f1 = evaluator.evaluate()
        return out, f1
    ###########################################################################
    ###########################################################################
    ###########################################################################




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
        """
        Initializes and configures the type mappings and identifier settings for span and relationship types
        within the model's configuration. This method sets several configuration properties related to span
        and relationship types, including creating mapping dictionaries and a list of all possible spans.

        Updates the following in self.config:
        - num_span_types: Number of span types, not including the 'none_span'.
        - num_rel_types: Number of relationship types, not including the 'none_rel'.
        - none_span: A placeholder name for non-existent span types.
        - none_rel: A placeholder name for non-existent relationship types.
        - s_to_id: Dictionary mapping from span types to their indices.
        - id_to_s: Dictionary mapping from indices to span types.
        - r_to_id: Dictionary mapping from relationship types to their indices.
        - id_to_r: Dictionary mapping from indices to relationship types.
        - all_span_ids: A list of all possible spans generated from the span types.

        No parameters are required as the method operates directly on the class's config attribute.
        """
        self.config.none_span = 'none_span'
        self.config.none_rel  = 'none_rel'
        
        self.config.num_span_types = len(self.config.span_types)   #scalar, does not include the none_span (idx 0)
        self.config.num_rel_types  = len(self.config.rel_types)    #scalar, does not include the none_rel (idx 0)
        
        self.config.s_to_id, self.config.id_to_s = self.create_type_mappings(self.config.span_types, self.config.none_span)
        self.config.r_to_id, self.config.id_to_r = self.create_type_mappings(self.config.rel_types, self.config.none_rel)
        
        #get all span_ids in seq_len
        self.config.all_span_ids = self.make_all_possible_spans()


    def testing_check_loader(self, loader):
        '''
        Keep this function, it is for testing
        '''
        x = iter(loader)
        batch = next(x)
        print(batch.keys())
        print(batch)


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

        #add to the config => the span and rel type mapping dicts and the all_possible_spans data
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




