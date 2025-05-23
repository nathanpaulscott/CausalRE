import torch
import torch.utils.checkpoint as checkpoint

#from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pathlib import Path
from typing import Tuple, List, Dict, Union
import os, re, math

###############################################
#custom imports
from .data_processor import DataProcessor
from .evaluator import make_evaluator
from .predictor import Predictor
from .model_manager import ModelManager, Optimizer, Scheduler
from .data_preparation import DataPreparation
from .utils import clear_gpu_tensors, set_all_seeds, save_to_json, join_paths
from .process_preds import process_preds




class Trainer:
    '''
    This is the Trainer class that brings in the config and basically orchestrates everything 
    It is used for prediction and training   
    '''
    def __init__(self, main_configs):
        self.main_configs = main_configs
        self.config = self.main_configs.as_namespace


    def load_loader(self, data_loader, infinite=True):
        """
        Generator function to endlessly yield batches from the data loader, optionally moving them to a specified device,
        restarting from the beginning once all batches have been yielded.

        Args:
            data_loader (DataLoader): The DataLoader from which to fetch data.
            infinite (bool): If True, yields batches indefinitely. If False, yields batches once through the data loader.

        Yields:
            batch (dict): A batch from the data_loader, with all tensors moved to the specified device.
        """
        while True:
            for batch in data_loader:
                # Move each tensor in the batch to the specified device
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                yield batch
            if not infinite:
                break



    ##############################################################
    ##############################################################
    ##############################################################
    #TRAIN/EVAL
    ##############################################################
    def train_step(self, model, scaler, iter_loader_inf, step):
        x = next(iter_loader_inf)

        model.train()
        if self.config.num_precision == 'half':
            with torch.amp.autocast('cuda', dtype=torch.float16):
                result = model(x, step=step)
        else:
            result = model(x, step=step)

        #abort if no results were returned
        if result is None:
            self.config.logger.write(f"Warning: forward pass aborted at step {step}. Skipping...", level='warning')
            return None, None, None, None
        elif result == 1:
            self.config.logger.write(f"Warning: forward pass aborted (top_k_spans was 0) at step {step}. Skipping...", level='warning')
            return None, None, None, None
        elif result == 2:
            self.config.logger.write(f"Warning: forward pass aborted (top_k_rels was 0) at step {step}. Skipping...", level='warning')
            return None, None, None, None
        elif torch.isnan(result['loss']).any():
            self.config.logger.write(f"Warning: NaN loss detected at step {step}. Skipping...", level='warning')
            return None, None, None, None
        
        #get the loss
        loss = result['loss']
        #check if we got NaN loss
        if self.config.loss_reduction == 'mean':
            loss = loss / self.config.accumulation_steps

        #Accumulate gradients
        if self.config.num_precision == 'half':
            scaler.scale(loss).backward()  
        else:
            loss.backward()
                
        #Clip gradients => NOTE: this needs adjustment for use with mixed precision
        raw_norm = None
        if self.config.grad_clip:
            clip_val = max(1.0, self.config.grad_clip * (1 - step / (self.config.num_steps/2)))
            raw_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
        
        #get lost rel count sum for info
        lost_rel_cnt = result['lost_rel_counts'].sum().item()
        #get teh loss_breakdown
        loss_breakdown = result['loss_breakdown']

        #clear tensors to free up GPU memory
        if (step + 1) % self.config.clear_tensor_steps == 0:
            #self.config.logger.write(' Clearing tensors')
            clear_gpu_tensors([v for k,v in result.items() if k != 'loss'], gc_collect=True, clear_cache=True)   #slows it down if true

        return loss, lost_rel_cnt, loss_breakdown, raw_norm


    def train_loop(self, model, loaders):
        '''
        This is the training and eval loop
        '''
        self.config.logger.write('Starting the Train Loop')

        #read some params from config
        acc_steps = self.config.accumulation_steps
        num_steps = self.config.num_steps
        warmup_ratio = self.config.warmup_ratio
        num_warmup_steps = int(num_steps * warmup_ratio) if warmup_ratio < 1 else int(warmup_ratio)
        eval_every = self.config.eval_every
        #Initialize structure to store gradient data
        grad_stats = dict(
            high_thd   = 1.0,  # Define thresholds for high gradients
            low_thd    = 0.01,  # Define thresholds for low gradients
            high_cnt   = 0, 
            low_cnt    = 0, 
            total_norm = 0,
            total_norm_sd = 0
        )
        grad_data = {name: [] for name, _ in model.named_parameters()}

        #get the optimiser and scheduler
        optimizer = Optimizer(self.config, model).return_object
        scheduler = Scheduler(self.config, optimizer, num_warmup_steps, num_steps).return_object
        scaler = torch.amp.GradScaler('cuda')

        #get the loaders
        train_loader = loaders['train']
        val_loader = loaders['val']
        test_loader = loaders['test']


        #profile_dir = './logs/profiler'
        #if not os.path.exists(profile_dir):  
        #    os.makedirs(profile_dir)
        ###############################################################
        '''
        #this code is profiled
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                    #schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                    #on_trace_ready=tensorboard_trace_handler('./logs/profiler'),
                    record_shapes=False,
                    profile_memory=True,
                    with_stack=False) as prof:
        '''
        ###############################################################
        #make an infinitely iterable from train loader
        iter_loader_inf = self.load_loader(train_loader, infinite=True)
        optimizer.zero_grad()    # Zero out gradients for next accumulation
        pbar_train = tqdm(range(num_steps), position=0, leave=True)
        train_loss = []
        train_loss_breakdown = []
        model_save_score = -self.config.num_limit
        model_early_stop_cnt = 0
        for step in pbar_train:
            loss, lost_rel_cnt, loss_breakdown, raw_norm = self.train_step(model, scaler, iter_loader_inf, step)
            if loss is None: continue  # Skip step if NaN loss or aborted forward pass

            if (step + 1) % acc_steps == 0 or step == num_steps - 1:
                #accumulate gradients
                if self.config.num_precision == 'half':
                    scaler.step(optimizer)   # Update model parameters
                    scaler.update()          # Update the scale for next iteration
                else:
                    optimizer.step()

                #Collect gradients for debugging
                if self.config.collect_grads: 
                    self.collect_grads(model, grad_data, grad_stats)
                    grad_stats['raw_total_norm'] = raw_norm

                #step the lr scheduler and zero grads
                scheduler.step()      
                optimizer.zero_grad() 

                #write description to log and stdout
                description = f"train step: {step} | epoch: {step // len(train_loader)} | loss: {loss.item():.2f} | lost rel cnt: {lost_rel_cnt} |"
                if self.config.collect_grads: 
                    description += f" | RawTotNorm: {grad_stats['raw_total_norm']:.4f} | TotNorm: {grad_stats['total_norm']:.4f} | TotNorm.SD: {grad_stats['total_norm_sd']:.4f} | HighGrads: {grad_stats['high_cnt']} | LowGrads: {grad_stats['low_cnt']}"
                if self.config.log_step_data:
                    self.config.logger.write(description, 'info', output_to_console=False)   
                pbar_train.set_description(description)
                
                #save the loss off the gpu
                train_loss.append(loss.detach().cpu().item())
                train_loss_breakdown.append(loss_breakdown)

            #runs eval and saves the model          
            #if step > self.config.model_min_eval_steps and (step + 1) % eval_every == 0:
            if step > self.config.model_min_eval_steps and (step + 1) % int(random.gauss(eval_every, eval_every*0.15)) == 0:    #randomise the eval points a bit
                '''
                #run the eval loop on the test set
                result = self.eval_loop(model, test_loader, step=step)
                #show the metrics on screen
                if 'visual_results' in result:
                    print(result['visual results - TEST'])
                print(self.make_metrics_summary(result, msg='test results ', type='format'))
                test_loss_msg = f"step: {step}, test loss mean: {round(result['eval loss'],4)}"
                self.config.logger.write(f"{test_loss_msg}, {self.make_metrics_summary(result, type='log')}", output_to_console=False)
                #get the test loss breakdown, its called the eval_loss breadown in the result!!!
                if self.config.loss_breakdown:
                    self.dump_loss_breakdown(result['eval_loss_breakdown'], prefix_msg='test loss breakdown')
                '''
                #run the eval loop
                result = self.eval_loop(model, val_loader, step=step)
                #get the train loss mean
                train_loss_mean = sum(train_loss) / len(train_loss) if len(train_loss) > 0 else -1
                model_save_score_current = 0 if result['eval loss'] <= 0 else self.make_model_save_score(result)
                loss_msg = f"step: {step}, train loss mean: {round(train_loss_mean,4)}, eval loss mean: {round(result['eval loss'],4)}, save_score: {round(model_save_score_current, 4)}"
                #reset the train loss list
                train_loss = []
                #show the metrics on screen
                if 'visual_results' in result:
                    print(result['visual results'])
                print(self.make_metrics_summary(result, msg='val results ', type='format'))
                print(loss_msg)
                #write the data to the log
                self.config.logger.write(f"{loss_msg}, {self.make_metrics_summary(result, type='log')}", output_to_console=False)
                #collect the loss breakdowns
                if self.config.loss_breakdown:
                    self.dump_loss_breakdown(train_loss_breakdown)
                    self.dump_loss_breakdown(result['eval_loss_breakdown'], prefix_msg='eval loss breakdown')

                #save the model if the metrics meet the criteria
                if self.config.model_save and step > self.config.model_min_save_steps:
                    eval_loss = result['eval loss']
                    if model_save_score_current == 0:   #eval_loss is None or eval_loss <= 0:
                        print(f'bad eval loss skipping save evaluation: {eval_loss}')
                    else:
                        ########################################################
                        if not hasattr(self, 'save_score_history'):
                            self.save_score_history = []
                            self.ema_score_history = []
                            self.ema_alpha = self.config.save_score_ema_alpha
                        # Append current raw score
                        self.save_score_history.append(model_save_score_current)
                        # Compute EMA
                        if len(self.save_score_history) == 1:
                            smoothed_score = model_save_score_current
                        else:
                            prev_ema = self.ema_score_history[-1]
                            smoothed_score = self.ema_alpha * model_save_score_current + (1 - self.ema_alpha) * prev_ema
                        self.ema_score_history.append(smoothed_score)
                        ########################################################

                        if smoothed_score > model_save_score:
                            model_early_stop_cnt = 0
                            model_save_score = smoothed_score
                            self.model_manager.save_pretrained(model, self.config.model_colab_folder, self.config.model_file_name)
                            self.config.logger.write(f"Model saved — step={step}, raw_score={model_save_score_current:.4f}, smoothed={smoothed_score:.4f}")
                        else:
                            #increment the early stopping counter and exit the train loop if it passes the thd
                            model_early_stop_cnt += 1
                            self.config.logger.write(f"Model not saved — step={step}, raw_score={model_save_score_current:.4f}, smoothed={smoothed_score:.4f} < {model_save_score:.4f}")
                            if model_early_stop_cnt >= self.config.model_early_stopping: 
                                self.config.logger.write("Early Stopping Triggered")
                                break



            '''
            ###############################################################
            prof.step()  # Inform profiler that one step (iteration) is complete
            
            #TEMP, break for quick analysis of the profiler
            if step > 25:
                break
            ###############################################################
            ''' 
        '''
        ###########################################
        # Directly print the results at the end of the profiling
        #NOTE: you can not see the averages, you have to calc them yourself
        # Extract events and relevant metrics
        #print_profile_results(prof)
        #exit()
        ###########################################
        '''
        pbar_train.close()

        #load the best model
        if not os.path.exists(self.config.model_colab_path):
            return None
        model = self.model_manager.load_pretrained(self.config.model_colab_path, self.config.device)

        #run the final eval and test loop
        result_val = self.eval_loop(model, val_loader)
        result_test = self.eval_loop(model, test_loader)
        #write summary to log
        self.config.logger.write(self.make_metrics_summary(result_val, msg='final_val '))
        self.config.logger.write(self.make_metrics_summary(result_test, msg='final_test '))
        #copy models to drive
        self.model_manager.copy_model_to_drive(self.config.model_colab_folder, self.config.model_full_folder, self.config.model_file_name)

        #run predict on the best model
        self.config.logger.write('running predict with best model')
        #force model to predict mode
        model.config.run_type = 'predict'
        result = self.predict_loop(model, test_loader)
        pred_result_file = join_paths(self.config.predict_folder, f'pred_{self.config.model_name}.json')
        pred_analysis_file = join_paths(self.config.predict_folder, f'pred_analysis_{self.config.model_name}.txt')
        save_to_json(result, pred_result_file)
        process_preds(pred_result_file, pred_analysis_file)

        return dict(
            result_val = result_val,
            result_test = result_test
        )



    def make_model_save_score(self, result):
        '''
        makes the f1 based score for saving the model, favours balanced precision and recall over unbalanced ones
        #the balance calc is reduced in severity by quad root so that balance requirements do not dominate
        '''
        def make_score(p, r, f1, k):
            balance = min(p, r) / max(p, r) if max(p, r) > 0 else 0
            balance = balance**self.config.balance_reduction_factor
            return f1 * balance

        # Span score
        p = result['span_metrics']['precision']
        r = result['span_metrics']['recall']
        f1 = result['span_metrics']['f1']
        span_score = make_score(p,r,f1, self.config.balance_reduction_factor) 
        # Rel score
        p = result['rel_metrics']['precision']
        r = result['rel_metrics']['recall']
        f1 = result['rel_metrics']['f1']
        rel_score = make_score(p,r,f1, self.config.balance_reduction_factor) 

        save_score  = (span_score + rel_score) / 2

        # Loss penalty
        #loss = result.get('eval loss')
        #loss_penalty = 1/(1 + loss)

        return save_score   # * loss_penalty



    def make_metrics_summary(self, result, msg='', type='format'):
        if type == 'format':
            #do formated version
            msg = f'Eval_type: {msg}\n-------------------------------\n'
            msg += f"Span:      {result['span_metrics']['msg_screen']}\n"
            msg += f"Rel:       {result['rel_metrics']['msg_screen']}\n"
            msg += f"Rel_mod:   {result['rel_mod_metrics']['msg_screen']}\n"
            msg += '-------------------------------\n'
            if self.config.matching_loose:
                msg += f"Span_l:    {result['span_metrics_l']['msg_screen']}\n"
                msg += f"Rel_l:     {result['rel_metrics_l']['msg_screen']}\n"
                msg += f"Rel_mod_l: {result['rel_mod_metrics_l']['msg_screen']}\n"
                msg += '-------------------------------\n'
        else:
            #do single line version
            msg = f"Span: {result['span_metrics']['msg_log']}, "
            msg += f"Rel: {result['rel_metrics']['msg_log']}, "
            msg += f"Rel_mod: {result['rel_mod_metrics']['msg_log']}"
            if self.config.matching_loose:
                msg += f", "
                msg += f"Span_l: {result['span_metrics_l']['msg_log']}, "
                msg += f"Rel_l: {result['rel_metrics_l']['msg_log']}, "
                msg += f"Rel_mod_l: {result['rel_mod_metrics_l']['msg_log']}"
        
        return msg


    def dump_loss_breakdown(self, loss_breakdown, prefix_msg='loss breakdown'):
        # Compute the average for each key
        n = len(loss_breakdown)
        avg = {}
        for k in loss_breakdown[0].keys():
            avg[k] = sum(item[k] for item in loss_breakdown) / n
        # Format and emit one line
        msg = ', '.join(f"{k}: {v:.2f}" for k, v in avg.items())
        self.config.logger.write(f"{prefix_msg}: {msg}", output_to_console=False)
        # Reset for next window
        loss_breakdown.clear()


    def check_inf_gradients(self, model):
        inf_gradients = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isinf(param.grad).any():  # Check if any element in gradients is inf
                    inf_gradients.append(name)
        return inf_gradients




    def collect_grads(self, model, grad_data, grad_stats):
        '''
        collect gradient data for debugging
        '''
        grad_stats['total_norm'] = 0
        grad_stats['high_cnt'] = 0
        grad_stats['low_cnt'] = 0
        all_norms = []
        inf_gradients = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isinf(param.grad).any():  # Check if any element in gradients is inf
                    inf_gradients.append(name)
                norm = torch.norm(param.grad).item()
                all_norms.append(norm)
                grad_data[name].append(norm)
                grad_stats['total_norm'] += norm ** 2  # Sum of squares for L2 norm

                # Counting high and low gradients
                if norm > grad_stats['high_thd']:
                    grad_stats['high_cnt'] += 1
                if norm < grad_stats['low_thd']:
                    grad_stats['low_cnt'] += 1

        grad_stats['total_norm'] = np.sqrt(grad_stats['total_norm'])  # Take square root for final L2 norm
        grad_stats['total_norm_sd'] = np.std(all_norms) if all_norms else 0

        #diagnostics
        diagnostics = False
        if diagnostics:
            print(all_norms)
            data_series = pd.Series(all_norms)
            # Converting all None to NaN for consistent handling in numpy
            data_series = data_series.apply(pd.to_numeric, errors='coerce')
            # Calculating the 7-number summary plus counts of NaN/None and inf values
            summary = (
                data_series.min(skipna=True),  # Minimum, ignoring NaN and inf
                data_series.quantile(0.25, interpolation='midpoint'),  # First Quartile, ignoring NaN and inf
                data_series.median(skipna=True),  # Median, ignoring NaN and inf
                data_series.quantile(0.75, interpolation='midpoint'),  # Third Quartile, ignoring NaN and inf
                data_series.max(skipna=True),  # Maximum, ignoring NaN and inf
                data_series.mean(skipna=True),  # Mean, ignoring NaN and inf
                data_series.std(skipna=True),  # Standard Deviation, ignoring NaN and inf
                data_series.isna().sum(),  # Count of NaN/None values
                np.isinf(data_series.replace(np.nan, 0)).sum()  # Count of inf values (replace NaN with 0 temporarily to avoid counting them as inf)
            )
            #Print each part of the summary with a description
            print("Extended 7-Number Summary including NaN, None, and inf counts:")
            print(f"Minimum: {summary[0]} (The smallest value, ignoring NaN and inf)")
            print(f"First Quartile (Q1): {summary[1]} (The 25th percentile, middle value of the first half, ignoring NaN and inf)")
            print(f"Median (Q2): {summary[2]} (The middle value of the dataset, ignoring NaN and inf)")
            print(f"Third Quartile (Q3): {summary[3]} (The 75th percentile, middle value of the second half, ignoring NaN and inf)")
            print(f"Maximum: {summary[4]} (The largest value, ignoring NaN and inf)")
            print(f"Mean: {summary[5]} (The average value, ignoring NaN and inf)")
            print(f"Standard Deviation (SD): {summary[6]} (A measure of the spread of the data, ignoring NaN and inf)")
            print(f"NaN/None Count: {summary[7]} (The total number of missing or invalid entries)")
            print(f"Infinite Count: {summary[8]} (The total number of infinite values)") 

            print(inf_gradients)




    def show_visual_results(self, evaluator):
        #here get the eval obs that you want to display from evaluator.all_preds['spans'] and evaluator.all_labels['spans']
        #the raw word tokenized tokens are in x['tokens']
        msg = '\n#######################\nVisual Results\n#####START###############\n'
        #catch cases where the eval_idx is too long for the available preds/labels and pass
        def numpy_int_to_int(data):
            return [tuple(int(item) for item in obs) for obs in data]

        try:
            for offset in range(self.config.eval_batch_size):
                #get preds and labels
                eval_idx = self.config.eval_step_display*self.config.eval_batch_size + offset
                #tokens
                show_tokens = evaluator.all_labels['tokens'][eval_idx]
                #spans
                show_span_preds = numpy_int_to_int(evaluator.all_preds['spans'][eval_idx])
                show_span_labels = evaluator.all_labels['spans'][eval_idx]
                #rels
                show_rel_preds = numpy_int_to_int(evaluator.all_preds['rels'][eval_idx])
                show_rel_labels = evaluator.all_labels['rels'][eval_idx]

                #sort
                show_span_preds = sorted(show_span_preds, key=lambda x: (x[0], x[1]))
                show_span_labels = sorted(show_span_labels, key=lambda x: (x[0], x[1]))
                show_rel_preds = sorted(show_rel_preds, key=lambda x: (x[0], x[1], x[3], x[4]))
                show_rel_labels = sorted(show_rel_labels, key=lambda x: (x[0], x[1], x[3], x[4]))
                msg += f'eval idx: {eval_idx}\n'
                msg += f'tokens: {show_tokens}\n'
                msg += f'span labels: {show_span_labels}\n'    
                msg += f'span_preds:  {show_span_preds}\n'
                msg += f'rel labels: {show_rel_labels}\n'    
                msg += f'rel_preds:  {show_rel_preds}\n'
                msg += '---------------------------\n'
        except Exception as e:
            pass
        msg += '######END################'

        return msg





    def eval_loop(self, model, data_loader, step=None):
        '''
        The run_type param here will be 'train' so we have labels and calculate loss
        the only difference is that the loss is disconnected from the graph as the model is run in .eval mode
        '''
        #make the predictor and evaluators
        predictor = Predictor(self.config)
        evaluator = make_evaluator(self.config)
        eval_loader = self.load_loader(data_loader, infinite=False)
        total_eval_steps = len(data_loader)
        visual_results = ''
        eval_loss = []
        eval_loss_breakdown = []
        model.eval()
        with torch.no_grad():
            #for x in tqdm(eval_loader, desc=f"Eval({total_eval_steps} batches)", leave=True):
            pbar_eval = tqdm(range(total_eval_steps), position=1, leave=True, desc='Eval Loop')
            for eval_step in pbar_eval:
                x = next(eval_loader)
                if self.config.num_precision == 'half':
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        result = model(x, step=step)    #step should be the train step here
                else:
                    result = model(x, step=step)        #step should be the train step here

                if result is None:
                    self.config.logger.write(f"Warning: Aborted forward pass at {eval_step}. Skipping...", level='warning')
                    continue  # Skip step if NaN loss or aborted forward pass
                elif result == 1:
                    self.config.logger.write(f"Warning: Aborted forward pass (top_k_spans was 0) at {eval_step}. Skipping...", level='warning')
                    continue  # Skip step if NaN loss or aborted forward pass
                elif result == 2:
                    self.config.logger.write(f"Warning: Aborted forward pass (top_k_rels was 0) at {eval_step}. Skipping...", level='warning')
                    continue  # Skip step if NaN loss or aborted forward pass
                elif torch.isnan(result['loss']).any():
                    self.config.logger.write(f"Warning: NaN loss detected at step {eval_step}. Skipping...", level='warning')
                    continue

                #get preds => the predicted positive cases only
                #NOTE: predicted neg cases are not included here as they are not required
                #NOTE: the rel_preds are the full rels with the span start,end,type info
                preds_and_labels = predictor.predict(x, result, return_and_reset_results=True)
                #prepare and add the batch of preds and labels to the evaluator
                evaluator.prep_and_add_batch(preds_and_labels)

                #write description to log and stdout
                description = f"eval step: {eval_step} | loss: {result['loss'].item():.2f} | lost rel cnt: {result['lost_rel_counts'].sum().item()} |"
                if self.config.log_step_data:
                    self.config.logger.write(description, 'info', output_to_console=False)   #no output to console for this one
                #disable for now
                #pbar_eval.set_description(description)

                #clear tensors to free up GPU memory
                if (eval_step + 1) % self.config.clear_tensor_steps == 0:
                    #self.config.logger.write(f' Eval step {eval_step}, clearing tensors')
                    clear_gpu_tensors([v for k,v in result.items() if k != 'loss'], gc_collect=True, clear_cache=True)   #slows it down if true
                
                eval_loss.append(result['loss'].detach().cpu().item())
                if self.config.loss_breakdown:
                    eval_loss_breakdown.append(result['loss_breakdown'])
            pbar_eval.close()


        #post loop processing
        #run the evaluator on whole dataset results (stored in the evaluator object) and return a dict with the metrics and preds
        result = evaluator.evaluate()
        #add the visual results and the eval mean loss
        if self.config.eval_step_display:
            result['visual results'] = self.show_visual_results(evaluator)
        result['eval loss'] = sum(eval_loss) / len(eval_loss) if len(eval_loss) > 0 else -1
        if self.config.loss_breakdown:
            result['eval_loss_breakdown'] = eval_loss_breakdown

        return result
    ##############################################################
    ##############################################################
    ##############################################################


    ##############################################################
    ##############################################################
    ##############################################################
    #PREDICT
    ##############################################################
    def predict_loop(self, model, data_loader):
        '''
        This is the predict loop, the run_type param is 'predict' so the model will run without labels and not calculate loss
        The loss returned from the model will be None
        '''
        self.config.logger.write('Starting the Predict Loop')

        #read some params from config
        #make the predictor and evaluators
        predictor = Predictor(self.config)
        evaluator = make_evaluator(self.config)
        pred_loader = self.load_loader(data_loader, infinite=False)
        total_pred_steps = len(data_loader)
        model.eval()
        with torch.no_grad():
            #for x in tqdm(pred_loader, desc=f"Pred({total_pred_steps} batches)", leave=True):
            pbar_eval = tqdm(range(total_pred_steps), position=1, leave=True, desc='Pred Loop')
            for pred_step in pbar_eval:
                x = next(pred_loader)
                if self.config.num_precision == 'half':
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        result = model(x)
                else:
                    result = model(x)

                if result is None:
                    self.config.logger.write(f"Warning: aborted forward pass at {pred_step}. Skipping...", level='warning')
                    continue  # Skip step if NaN loss or aborted forward pass
                elif result == 1:
                    self.config.logger.write(f"Warning: aborted forward pass (top_k_spans is 0) at {pred_step}. Skipping...", level='warning')
                    continue  # Skip step if NaN loss or aborted forward pass
                elif result == 2:
                    self.config.logger.write(f"Warning: aborted forward pass (top_k_rels is 0) at {pred_step}. Skipping...", level='warning')
                    continue  # Skip step if NaN loss or aborted forward pass

                #make the predictions for the batch and add to the predictor object    
                predictor.predict(x, result)

                #clear tensors to free up GPU memory
                pred_step += 1
                if (pred_step + 1) % self.config.clear_tensor_steps == 0:
                    #self.config.logger.write(f' Eval step {eval_step}, clearing tensors')
                    clear_gpu_tensors([v for k,v in result.items() if k != 'loss'], gc_collect=True, clear_cache=True)   #slows it down if true

        #convert preds from dict of list to list of dicts
        return predictor.gather_preds_and_convert_preds_to_list_of_dicts()
        

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
        
        set_all_seeds(self.config.random_seed)

        #load and prepare data
        #this also updates the main_configs with some key parameters
        data_preparation = DataPreparation(self.main_configs)             #needs to modify the main_configs
        dataset = data_preparation.load_and_prep_data()
        #update the self.configs from the updated main configs
        self.config = self.main_configs.as_namespace
        
        #make the data loaders
        #needs to be done here as the previous funtcion updated the main_configs
        data_processor = DataProcessor(self.config) 
        loaders = data_processor.create_dataloaders(dataset)

        #temp - for debugging
        #check the loaders here
        testing = False
        if testing: self.check_loaders(loaders)
        #temp - for debugging

        #get the model
        self.model_manager = ModelManager(self.main_configs) 
        model = self.model_manager.get_model()
        if model is None:
            print(f'no model was found to load with {self.config.model_file_name}')
            return None

        #send config to log
        self.config.logger.write('Configs Snapshot:\n' + self.main_configs.dump_as_json_str, 'info', output_to_console=True)

        #kick off the train_loop or predict_loop
        if self.config.run_type == 'train': 
            result = self.train_loop(model, loaders)
            #this has the results_eval and result_test in a dict

        elif self.config.run_type == 'predict': 
            result = self.predict_loop(model, loaders['predict'])
            pred_result_file = join_paths(self.config.predict_folder, f'pred_{self.config.model_name}.json')
            pred_analysis_file = join_paths(self.config.predict_folder, f'pred_analysis_{self.config.model_name}.txt')
            save_to_json(result, pred_result_file)
            process_preds(pred_result_file, pred_analysis_file)

        else:
            raise Exception("Error - run_type must be either 'train' or 'predict'")
        







###########################################################################
###########################################################################
###########################################################################
def print_profile_results(prof):
    '''
    temp function to show the profiler output
    '''
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
