import torch
import numpy as np
import os, re
from pathlib import Path
from typing import Tuple, List, Dict, Union
from tqdm import tqdm

###############################################
#custom imports
from .data_processor import DataProcessor
from .utils import import_data, load_from_json, save_to_json
from .evaluator import Evaluator
from .predictor import Predictor
from .validator import Validator
from .model_manager import ModelManager, Optimizer, Scheduler
from .data_preparation import DataPreparation






class Trainer:
    '''
    This is the Trainer class that brings in the config and basically orchestrates everything 
    It is used for prediction and training   
    '''
    def __init__(self, main_configs):
        self.main_configs = main_configs
        self.config = main_configs.as_namespace


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



    ##############################################################
    ##############################################################
    ##############################################################
    #TRAIN/EVAL
    ##############################################################

    def train_step(self, model, optimizer, scheduler, scaler, iter_loader_inf, step):
        optimizer.zero_grad()
        x = next(iter_loader_inf)
        with torch.cuda.amp.autocast(dtype=torch.float16):
            model.train()
            result = model(x, step=step)
            loss = result['loss']

        if torch.isnan(loss).any():
            print(f"Warning: NaN loss detected at step {step}. Skipping...")
            return None

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        return loss



    def train_loop(self, model, loaders):
        '''
        This is the training and eval loop
        '''
        config = self.config

        #read some params from config
        device = config.device
        num_steps = config.num_steps
        pbar = tqdm(range(num_steps))
        warmup_ratio = config.warmup_ratio
        num_warmup_steps = int(num_steps * warmup_ratio) if warmup_ratio < 1 else int(warmup_ratio)
        eval_every = config.eval_every
        save_total_limit = config.save_total_limit
        log_dir = config.log_dir

        #get the optimiser and scheduler
        optimizer = Optimizer(config).get_optimizer(model)
        scheduler = Scheduler(config).get_scheduler(optimizer, num_warmup_steps, num_steps)
        scaler = torch.cuda.amp.GradScaler()

        #get the loaders
        train_loader = loaders['train']
        val_loader = loaders['val']
        test_loader = loaders['test']
        #make an infinitely iterable from train loader
        iter_loader_inf = self.load_loader(train_loader, device, infinite=True)

        for step in pbar:
            loss = self.train_step(model, optimizer, scheduler, scaler, iter_loader_inf)
            if loss is None: continue

            description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
            pbar.set_description(description)
            
            #runs eval and saves the model
            if (step + 1) % eval_every == 0:
                result = self.eval_loop(model, val_loader, device, step=step, msg='interim_val')
                checkpoint = f'model_{step + 1}'
                self.model_manager.save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)

        #run the final eval and test loop
        result_eval = self.eval_loop(model, val_loader, device, step=None, msg='final_val')
        result_test = self.eval_loop(model, test_loader, device, step=None, msg='final_test')
        checkpoint = f'model_{step + 1}'
        self.model_manager.save_top_k_checkpoints(model, log_dir, checkpoint, save_total_limit)

        return dict(
            result_eval = result_eval,
            result_test = result_test
        )


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
        config = self.config

        #make the predictor and evaluators
        predictor = Predictor(config)
        evaluator = Evaluator(config)
        
        iter_loader = self.load_loader(data_loader, device, infinite=False)
        model.eval()
        with torch.no_grad():
            for x in iter_loader:
                with torch.cuda.amp.autocast(dtype=torch.float16):    #forcing data to half precision
                    result = model(x, step=step)
                #get preds => the predicted positive cases
                #NOTE: the rel_preds are the full rels with the span start,end,type info
                preds = predictor.predict(result, return_and_reset_results=True)
                #prepare and add the batch of preds and labels to the evaluator
                evaluator.prep_and_add_batch(preds, x['spans'], x['relations'])
        
        #run the evaluator on whole dataset results (stored in the evaluator object) and return a dict with the metrics and preds
        result = evaluator.evaluate()
        self.print_metrics_summary(result, msg)
        return result
    ##############################################################
    ##############################################################
    ##############################################################



    ##############################################################
    ##############################################################
    ##############################################################
    #PREDICT
    ##############################################################
    def predict_loop(self, model, loaders):
        '''
        This is the predict loop, the run_type param is 'predict' so the model will run without labels and not calculate loss
        The loss returned from the model will be None
        '''
        config = self.config

        #read some params from config
        device = config.device
        predict_loader = loaders['predict']

        #make the predictor, no evaluator as have no labels
        predictor = Predictor(config)

        iter_loader = self.load_loader(predict_loader, device, infinite=False)
        model.eval()
        with torch.no_grad():
            for x in iter_loader:
                with torch.cuda.amp.autocast(dtype=torch.float16):    #forcing data to half precision
                    result = model(x)
                #make the predictions for the batch and add to the predictor object    
                predictor.predict(result)

        #return the final predict results from the predictor object
        return predictor.all_preds_out
    ##############################################################
    ##############################################################
    ##############################################################



    ################################################
    #main orchestration function for training/eval
    ################################################
    def run(self):
        #load and prepare data
        #this also updates the main_configs with some key parameters
        data_preparation = DataPreparation(self.main_configs)             #needs to modify the main_configs
        dataset = data_preparation.load_and_prep_data()

        #make the data loaders
        #needs to be done here as the previous funtcion updated the main_configs
        data_processor = DataProcessor(self.config) 
        loaders = data_processor.create_dataloaders(dataset)

        #get the model
        model_manager = ModelManager(self.config) 
        model = model_manager.get_model()

        #kick off the train_loop or predict_loop
        if self.config.run_type == 'train': 
            result = self.train_loop(model, loaders)
            #this has the results_eval and result_test in a dict

        elif self.config.run_type == 'predict': 
            result = self.predict_loop(model, loaders)
            #you may want to reformat the preds here to a format that is importable by my js tool for viewing
            #i.e. the standard spans/relations format, where the rels just have the head/tail index in the spans list etc...

        else:
            raise Exception("Error - run_type must be either 'train' or 'predict'")


