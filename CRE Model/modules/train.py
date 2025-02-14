import torch
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from tqdm import tqdm
import numpy as np
import pandas as pd
import os, re
from pathlib import Path
from typing import Tuple, List, Dict, Union

###############################################
#custom imports
from .data_processor import DataProcessor
from .evaluator import Evaluator
from .predictor import Predictor
from .model_manager import ModelManager, Optimizer, Scheduler
from .data_preparation import DataPreparation
from .utils import clear_gpu_tensors





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
    def train_step(self, model, scaler, iter_loader_inf, step):
        with record_function("step_0.1: get next batch for loader"):
            x = next(iter_loader_inf)

        with record_function("step_0.2: run model"):
            model.train()
            if self.config.num_precision == 'half':
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    result = model(x, step=step)
            else:
                result = model(x, step=step)
            loss = result['loss']

            if torch.isnan(loss).any():
                self.config.logger.write(f"Warning: NaN loss detected at step {step}. Skipping...", level='warning')
                return None

        with record_function("step_0.3: scaler"):
            if self.config.num_precision == 'half':
                scaler.scale(loss).backward()  # Accumulate gradients
            else:
                loss.backward()
                
        #clear tensors to free up GPU memory
        if (step + 1) % self.config.clear_tensor_steps == 0:
            with record_function("step_0.4: clear_tensors external"):
                #self.config.logger.write(' Clearing tensors')
                clear_gpu_tensors([v for k,v in result.items() if k != 'loss']),
                                  #gc_collect=False,    #slows it down if true
                                  #clear_cache=False)   #slows it down if true

        return loss


    def train_loop(self, model, loaders):
        '''
        This is the training and eval loop
        '''
        self.config.logger.write('Starting the Train Loop')
        config = self.config

        #read some params from config
        device = config.device
        acc_steps = config.accumulation_steps
        num_steps = config.num_steps
        pbar = tqdm(range(num_steps))
        warmup_ratio = config.warmup_ratio
        num_warmup_steps = int(num_steps * warmup_ratio) if warmup_ratio < 1 else int(warmup_ratio)
        eval_every = config.eval_every
        save_top_k = config.save_top_k
        log_folder = config.log_folder

        #get the optimiser and scheduler
        optimizer = Optimizer(config, model).return_object
        scheduler = Scheduler(config, optimizer, num_warmup_steps, num_steps).return_object
        scaler = torch.cuda.amp.GradScaler()

        #get the loaders
        train_loader = loaders['train']
        val_loader = loaders['val']
        test_loader = loaders['test']


        profile_dir = './logs/profiler'
        if not os.path.exists(profile_dir):  
            os.makedirs(profile_dir)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                    #schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                    #on_trace_ready=tensorboard_trace_handler('./logs/profiler'),
                    record_shapes=False,
                    profile_memory=True,
                    with_stack=False) as prof:
            ###############################################################
            #this code is profiled
            ###############################################################
            #make an infinitely iterable from train loader
            iter_loader_inf = self.load_loader(train_loader, device, infinite=True)
            optimizer.zero_grad()    # Zero out gradients for next accumulation
            for step in pbar:
                loss = self.train_step(model, scaler, iter_loader_inf, step)
                if loss is None: continue  # Skip step if NaN loss

                if (step + 1) % acc_steps == 0 or step == num_steps - 1:
                    self.config.logger.write(' Applying accumulated gradients ...')
                    if self.config.num_precision == 'half':
                        scaler.step(optimizer)   # Update model parameters
                        scaler.update()          # Update the scale for next iteration
                    else:
                        optimizer.step
                    optimizer.zero_grad()    # Zero out gradients for next accumulation
                    scheduler.step()         # Scheduler step, if using a learning rate scheduler
                    
                description = f"step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f}"
                self.config.logger.write(description, 'info', output_to_console=False)   #no output to console for this one
                pbar.set_description(description)
                
                #runs eval and saves the model
                if (step + 1) % eval_every == 0:
                    result = self.eval_loop(model, val_loader, device, step=step)
                    self.config.logger.write(self.make_metrics_summary(result, msg='interim_val '))
                    #save the model
                    #checkpoint = f'model_{step + 1}'
                    #self.model_manager.save_top_k_checkpoints(model, log_folder, checkpoint, save_top_k)

                ###############################################################
                prof.step()  # Inform profiler that one step (iteration) is complete
                if step == 25:
                    break
                ###############################################################


        ###########################################
        # Directly print the results at the end of the profiling
        #NOTE: you can not see the averages, you have to calc them yourself
        # Extract events and relevant metrics
        filtered_events = [evt for evt in prof.key_averages() if evt.key.startswith("step_")]
        data = []
        for evt in filtered_events:
            data.append({
                "Name": evt.key,
                "Calls": evt.count,
                "CPU.tot.ms": round(evt.cpu_time_total / 1e3, 5),
                "SelfCPU.ms": round(evt.self_cpu_time_total / 1e3, 5),
                "CUDA.tot.ms": round(getattr(evt, "device_time", getattr(evt, "cuda_time_total", 0)/evt.count) * evt.count / 1e3, 5),  #New PyTorch
                "Self.CUDA.ms": round(getattr(evt, "self_device_time", getattr(evt, "self_cuda_time_total", 0)/evt.count) *evt.count/ 1e3, 5),  #New PyTorch
                "CPU.ave.ms": round(evt.cpu_time_total / evt.count / 1e3, 5) if evt.count > 0 else 0,
                "CUDA.ave.ms": round(getattr(evt, "device_time", getattr(evt, "cuda_time_total", 0)/evt.count) / 1e3, 5) if evt.count > 0 else 0,
                "CPU.GB": round(evt.cpu_memory_usage / (1024**3), 5),
                "Self.CPU.GB": round(evt.self_cpu_memory_usage / (1024**3), 5),
                "CUDA.GB": round(getattr(evt, "device_memory_usage", getattr(evt, "cuda_memory_usage", 0)) / (1024**3), 5),  # ✅ New PyTorch
                "Self.CUDA.GB": round(getattr(evt, "self_device_memory_usage", getattr(evt, "self_cuda_memory_usage", 0)) / (1024**3), 5)  # ✅ New PyTorch
            })
        # Create DataFrame
        df = pd.DataFrame(data)
        df_sorted = df.sort_values(by="CPU.tot.ms", ascending=False)
        print(df_sorted.head(30))
        # Sort by total CPU time if needed
        df_sorted = df.sort_values(by="CUDA.tot.ms", ascending=False)
        print(df_sorted.head(30))
        # Save the profiling results to a file
        prof.export_chrome_trace("./logs/profiler/output_trace.json")
        
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        exit()
        ###########################################


        #run the final eval and test loop
        result_val = self.eval_loop(model, val_loader, device)
        result_test = self.eval_loop(model, test_loader, device)
        #write summary to log
        self.config.logger.write(self.make_metrics_summary(result_val, msg='final_val '))
        self.config.logger.write(self.make_metrics_summary(result_test, msg='final_test '))
        #save the model
        checkpoint = f'model_{step + 1}'
        self.model_manager.save_top_k_checkpoints(model, log_folder, checkpoint, save_top_k)

        return dict(
            result_val = result_val,
            result_test = result_test
        )


    def make_metrics_summary(self, result, msg):
        msg = f'Eval_type: {msg}\n-------------------------------\n'
        msg += f"Span Metrics:    {result['span_metrics']['msg']}"
        msg += f"Rel Metrics:     {result['rel_metrics']['msg']}"
        msg += f"Rel_mod Metrics: {result['rel_mod_metrics']['msg']}"
        msg += '-------------------------------'
        return msg


    def eval_loop(self, model, data_loader, device, step=None):
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
        eval_step = 0
        total_eval_steps = len(data_loader)
        with torch.no_grad():
            for x in tqdm(iter_loader, desc=f"Eval({total_eval_steps} batches)", leave=True):
                with record_function("step_eval_1: run model"):
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        result = model(x, step=step)
                #get preds => the predicted positive cases
                #NOTE: the rel_preds are the full rels with the span start,end,type info
                with record_function("step_eval_2: predictor"):
                    preds = predictor.predict(result, return_and_reset_results=True)
                #prepare and add the batch of preds and labels to the evaluator
                with record_function("step_eval_3: evaluator"):
                    evaluator.prep_and_add_batch(preds, x['spans'], x['relations'])

                #clear tensors to free up GPU memory
                eval_step += 1
                if (eval_step + 1) % self.config.clear_tensor_steps == 0:
                    #self.config.logger.write(f' Eval step {eval_step}, clearing tensors')
                    clear_gpu_tensors([v for k,v in result.items()])
                
                #TEMP
                #TEMP
                #TEMP
                if eval_step > 20:
                    break
                #TEMP
                #TEMP
                #TEMP
        #run the evaluator on whole dataset results (stored in the evaluator object) and return a dict with the metrics and preds
        result = evaluator.evaluate()
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
        self.config.logger.write('Starting the Predict Loop')
        config = self.config

        #read some params from config
        device = config.device
        predict_loader = loaders['predict']

        #make the predictor, no evaluator as have no labels
        predictor = Predictor(config)

        pred_step = 0
        total_pred_steps = len(predict_loader)
        iter_loader = self.load_loader(predict_loader, device, infinite=False)
        model.eval()
        with torch.no_grad():
            for x in tqdm(iter_loader, desc=f"Pred({total_pred_steps} batches)", leave=True):
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    result = model(x)
                #make the predictions for the batch and add to the predictor object    
                predictor.predict(result)

                #clear tensors to free up GPU memory
                pred_step += 1
                if (pred_step + 1) % self.config.clear_tensor_steps == 0:
                    #self.config.logger.write(f' Eval step {eval_step}, clearing tensors')
                    clear_gpu_tensors([v for k,v in result.items()])

        #return the final predict results from the predictor object
        return predictor.all_preds_out
    ##############################################################
    ##############################################################
    ##############################################################

    #testing
    def check_loaders(self, loaders):
        # Check the loaders here
        # Fetch the first batch from the training loader to inspect
        N = 20
        train_iterator = iter(loaders['train'])
        for i in range(N):
            batch = next(train_iterator)
        print(f'batch {N} from training data loader:')
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: Tensor of shape {value.shape}")
            else:
                print(f"{key}: {value}")



    def run(self):
        #load and prepare data
        #this also updates the main_configs with some key parameters
        data_preparation = DataPreparation(self.main_configs)             #needs to modify the main_configs
        dataset = data_preparation.load_and_prep_data()
        
        #make the data loaders
        #needs to be done here as the previous funtcion updated the main_configs
        data_processor = DataProcessor(self.config) 
        loaders = data_processor.create_dataloaders(dataset)

        #check the loaders here
        testing = True
        if testing: self.check_loaders(loaders)

        #send config to log
        self.config.logger.write('Configs Snapshot:\n' + self.main_configs.dump_as_json_str)

        #get the model
        self.model_manager = ModelManager(self.config) 
        model = self.model_manager.get_model()

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


