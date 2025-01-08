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
                result = self.eval_loop(model, val_loader, device, step)
                #save the model
                checkpoint = f'model_{step + 1}'
                self.save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)


        #run the final eval and test loop
        result_eval = self.eval_loop(model, val_loader, device, step)
        result_test = self.eval_loop(model, test_loader, device, step)
        #do some shit with the results here
        #save the model
        checkpoint = f'model_{step + 1}'
        self.save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)




    def eval_loop(self, model, data_loader, device, step):
        '''
        The run_type param here will be 'train' so we have labels and calculate loss
        the only difference is that the loss is disconnected from the graph as the model is run in .eval mode
        '''
        #gets the configs
        pred_thd = self.config.predict_thd
        pred_conf = self.config.predict_confidence

        iter_loader = self.load_loader(data_loader, device, infinite=False)
        model.eval()
        with torch.no_grad():
            preds = []
            pos_labels = []
            for x in iter_loader:
                result = model(x, step=step)
                '''
                output = dict(
                    loss                 = total_loss,
                    ######################################
                    logits_span          = logits_span,
                    cand_span_masks      = cand_span_masks,
                    cand_span_ids        = cand_span_ids,
                    cand_span_labels     = cand_span_labels,
                    top_k_spans          = top_k_spans,
                    ######################################
                    logits_rel           = logits_rel,
                    cand_rel_masks       = cand_rel_masks,
                    cand_rel_ids         = cand_rel_ids,
                    cand_rel_labels      = cand_rel_labels,
                    top_k_rels           = top_k_rels,
                    lost_rel_counts      = lost_rel_counts  #how many positive rels did not get into rel_reps, i.e. not missing due to filtering or misclassification
                )
                '''
                #get preds => the predicted positive cases
                span_preds, rel_preds = self.predictor(x, result)
                #get the labels => the actual positive cases
                #align them and fill out the missing data in each one
                '''
                so for the full rel F1 metrics we need the full rels for pred and label (head_start, head_end, head_type, tail_start, tail_end, tail_type, rel_type)
                I would first get the preds for spans and rels using the appropriate calc for the label type
                I would then convert the positive preds to actual human readable data, i.e. for spans a list of tuples (start, end, type_string)
                I would then re-order the preds to the same order as raw labels (x['spans']) and just put in a filler for missed labels with no preds and add the preds with no labels after the labels, add nul vlaues to the raw labels to match the aligned preds length
                eg. preds = ('4_6_dog', 8_10_cat'), raw_labels = ('8_10_cat', 12_15_d0g') => pred_align = ('8_10_cat','x_x_x', '4_6_dog')
                    then add nul vals to the raw_labels => raw_labels_align = ('8_10_cat', 12_15_d0g', 'x_x_x')
                You can return the pres unaligned to the user
                You can used the pred_align and labels_align for the metrics calc, i.e. you only make these if you have labels

                Then for rels, do the preds, if multilabel, use sigmoid and thd
                then split out to human readable data for the pos rels
                i.e. each 1 in each binary vector becomes one rel, (head, tail, type)
                Then we have to lookup the span start, end, span_type from the head and tail idx (these are idx in the cand_span_ids)
                If the span was predicted as a neg, we read that (none)

                Anyway, so we form the pred rels which is a list of full rel tuples including span data
                We can output to the user or just output the spans list for each obs and the rels list for each obs with the head and tail idx aligned.
                Think about this
                for the metrics, we have to form the full rels for the labels and preds and then align them and then run metrics on them
                

                #in summary
                1) do preds for spans and rels
                2) convert to python obs => make the span and full rel tuples from the preds, then make the final rel tuples and decide which ones to return to the user
                NOTE: this is involved and processing requires a lot of mapping back to the raw indices as well as dealing with multilabel and unilabel cases etc...
                3) if have labels => make the span and full rel tuples from the raw annotations (spans use as is, rels convert to full)
                4) align the span labels/preds to the labels order adding extra preds to the end filling unmathced labels/preds with some kind of filler for labels not in preds and vice versa
                5) flatten and run metrics on this
                6) you can return extra data to the user from labels, such as which spans and rels were lost, which labels were incorrectly predicted, which we correctly predicted
                
                
                I would throw his code for this stuff
                

                '''        


                #I have the loss already here, so do not need to do anythign there...
                #I need the preds
                #then format the preds and labels for metrics, now remember we will have lost rels so 
                #the rel_labels from the model while being aligned with the preds will miss some of the rels
                #so you need to acount for these in the metrics potentially you can do the alignment again from the raw labels
                #just form the labels and then align the preds with this somehow for the metrics
                #I think you need to form the span and rel labels into what ever format you need in the preprocessing step
                #This makes it easy to do it now for metrics the preds and labels just need to be 2 flattened lists for the batch of pred ints (unilabels) bool (multilabels).
                #Actually it doesn't make much sense to premake these, just make them as you need them
                #work out the code the long way first and then see if you can speed it up later
                #inputs => preds (tensor), mask or perhaps preds as a list of lists....think

                spans, rels = self.predictor(x, result)
                preds.extend(self.merge(spans, rels))
                pos_labels.extend(self.get_full_relations(x))

        evaluator = Evaluator(pos_labels, preds)
        out, f1 = evaluator.evaluate()
        return out, f1



    def get_full_relations(self, x):
        '''
        This converts the span and relation labels (both being list of tuples)
        to full relation labels which is a list of tuples of the format:
        [(head start, head end, head type), (tail start, tail end, tail type), rel_type), (...), (...)...]
        '''
        batch_spans =  x['relations']
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



    def predict_spans_unilabel(self, span_logits, span_ids):
        '''
        Extracts the span predictions for unilabel classification in a list of lists of tuples form,
        where each list corresponds to a batch item.
        '''
        preds, probs = self.predict_unilabel(span_logits)  # Get predictions and probabilities
        #Initialize the list to hold batch-wise span information
        spans = [[] for _ in range(span_logits.shape[0])]  # batch_size is the first dimension
        #Find indices where predictions are positive (ignoring class 0, the "none" class)
        #will return 2 tensors of same length with that dims idx for each pos case 
        batch_indices, span_indices = torch.where(preds > 0)
        for batch_idx, span_idx in zip(batch_indices, span_indices):
            # Extract span details
            span_start = span_ids[batch_idx, span_idx, 0].item()
            span_end = span_ids[batch_idx, span_idx, 1].item()
            label = self.config.id_to_s[preds[batch_idx, span_idx].item()]  # Map class index to label string
            if self.config.predict_conf:
                # Include confidence score if required
                conf = probs[batch_idx, span_idx].item()
                spans[batch_idx.item()].append(((span_start, span_end, label), conf))
            else:
                spans[batch_idx.item()].append((span_start, span_end, label))

        return spans


    def predict_spans_multilabel(self, span_logits, span_ids):
        '''
        Extracts the span predictions for multilabel classification in a list of lists of tuples form,
        where each list corresponds to a batch item.
        '''
        preds, probs = self.predict_multilabel(span_logits, self.config.predict_thd)  # Get predictions and probabilities
        #Initialize the list to hold batch-wise span information
        spans = [[] for _ in range(span_logits.shape[0])]  # batch_size is the first dimension
        #Find indices where predictions are positive
        #will return 3 tensors of same length with that dims idx for each pos case 
        batch_indices, span_indices, label_indices = torch.where(preds == 1)
        for batch_idx, span_idx, label_idx in zip(batch_indices, span_indices, label_indices):
            #Extract span details
            span_start = span_ids[batch_idx, span_idx, 0].item()
            span_end = span_ids[batch_idx, span_idx, 1].item()
            label = self.config.id_to_s[label_idx.item()]  # Map label index to label string
            if self.config.predict_conf:
                #Include confidence score if required
                conf = probs[batch_idx, span_idx, label_idx].item()
                spans[batch_idx.item()].append(((span_start, span_end, label), conf))
            else:
                spans[batch_idx.item()].append((span_start, span_end, label))

        return spans





    def predict_rels(self, id_to_r, logits, span_ids, rel_ids, top_k_rels):
        #Apply sigmoid function to logits
        probs = torch.sigmoid(logits)
        #Initialize list of relations
        rels = [[] for _ in range(len(logits))]
        #Get indices where probability is greater than threshold
        above_thd_ids = (probs > self.config.predict_thd).nonzero(as_tuple=True)

        #Iterate over indices where probability is greater than threshold
        for batch_idx, pos, class_idx in zip(*above_thd_ids):
            #Get relation label
            label = id_to_r[class_idx.item() + 1]
            #Get predicted pair index
            pred_rel_id = rel_ids[batch_idx, pos].item()
            #Unravel predicted rel index into head and tail
            head_id, tail_id = np.unravel_index(pred_rel_id, (top_k_rels, top_k_rels))
            #Convert head and tail indices to tuples
            head = tuple(span_ids[batch_idx, head_id].tolist())
            tail = tuple(span_ids[batch_idx, tail_id].tolist())
            if self.config.predict_conf:
                #Get confidence
                conf = probs[batch_idx, pos, class_idx].item()
                #Append relation to list
                rels[batch_idx.item()].append((head, tail, label, conf))
            else:
                rels[batch_idx.item()].append((head, tail, label))

        return rels


    def predict_unilabel(self, logits):
        """
        Convert logits to single class predictions by applying softmax
        and then taking the argmax, along with the maximum probability
        for each predicted class.
        
        Args:
            logits (torch.Tensor): Logits tensor of shape (batch_size, num_classes).
        
        Returns:
            torch.Tensor: Predicted class indices tensor of shape (batch_size,).
            torch.Tensor: Maximum class probabilities tensor of shape (batch_size,).
        """
        # Apply softmax to convert logits to probabilities
        probs = torch.softmax(logits, dim=1)
        # Get the predicted class index
        preds = torch.argmax(probs, dim=1)
        # Gather the max probabilities for the predicted classes
        max_probs = torch.max(probs, dim=1)[0]  # [0] to select values only, [1] would give indices which are `preds`
        return preds, max_probs


    def predict_multilabel(self, logits, thd=0.5):
        """
        Convert logits to multilabel predictions by applying sigmoid
        and then using a threshold to determine label assignment.

        Args:
            logits (torch.Tensor): Logits tensor of shape (batch_size, num_labels).
            thd (float): Threshold for determining label assignment.

        Returns:
            torch.Tensor: Predicted labels tensor of shape (batch_size, num_labels),
                        where each element is 0 or 1.
            torch.Tensor: Probabilities tensor of shape (batch_size, num_labels),
                        representing the probability of each label.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)
        # Apply threshold to determine label assignments
        preds = (probs >= thd).int()
        return preds, probs


    def predictor(self, out):
        '''
        describe this function
        '''
        if self.config.span_labels == 'unilabel':
            spans = self.predict_spans_unilabel(out['logits_span'], out['cand_span_ids'])
        elif self.config.span_labels == 'multilabel':
            spans = self.predict_spans_multilabel(out['logits_span'], out['cand_span_ids'])


        #to do this part
        #to do this part
        #to do this part
        #to do this part
        if self.config.rel_labels == 'unilabel':
            rels = self.predict_rels_unilabel(out['logits_rel'], out['cand_rel_ids'])
        elif self.config.rel_labels == 'multilabel':
            rels = self.predict_rels_multilabel(out['logits_rel'], out['cand_rel_ids'])

        return spans, rels
    ########################################################################



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


    def testing_check_loader(self, loader):
        '''
        Keep this function, it is for testing
        '''
        x = iter(loader)
        batch = next(x)
        print(batch.keys())
        print(batch)

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




