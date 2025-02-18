import torch
import json, random, os, gc, time, types, logging
import numpy as np
from pathlib import Path
from datetime import datetime



##########################################
#general utilities
##########################################
def get_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_time


class measure_time():
    '''
    usage:
    meas= measure_time()

    #to start the meas...
    meas.start()

    #to end the meas...
    meas.stop()
    '''
    def __init__(self):
        pass

    def start(self):
        self.t_start = time.time()

    def stop(self):
        if self.t_start is None:
            print("Can't do time measurement as you did not call .start() first")
        else:
            print(f'delay is {round((time.time()-self.t_start)*1000,3)} ms')
            self.t_start = None



def set_all_seeds(seed=42):
    #Python RNG
    random.seed(seed)
    #Numpy RNG
    np.random.seed(seed)
    #PyTorch RNGs
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  # Sets Python hash seed





def clear_gpu_tensors(tensors, gc_collect=True, clear_cache=True):
    '''
    usage:
    if step % selfconfig.clear_tensor_period == 0:
        tensors_to_clear = a list of the tensor names to clear
        clear_gpu_tensors(tensors_to_clear)
    '''
    for tensor in tensors:
        if tensor is not None:
            del tensor
    
    if gc_collect:    #slows it down
        gc.collect()
    
    if torch.cuda.is_available() and clear_cache:    #slows it down
        torch.cuda.empty_cache()  # Clears the GPU cache




def not_class(x):
    #determines if an object is not a class
    if isinstance(x, (int, float, str, dict, list, tuple, set, frozenset, types.FunctionType, types.LambdaType)):
        return True
    return False



def print_overwrite(msg):
    '''
    prints a line but overwrites the last written line
    '''
    print(f'\r{msg}', end='')




##################################################################################
#import/export json
##################################################################################
##################################################################################
def check_utf_encoding(file_path):
    with open(file_path, 'rb') as file:
        first_three_bytes = file.read(3)
        if first_three_bytes == b'\xef\xbb\xbf':
            print(f"The file {file_path} is encoded with UTF-8-SIG.")
            return 'utf-8-sig'
        else:
            print(f"The file {file_path} is encoded with UTF-8 (no BOM).")
            return 'utf-8'




def load_from_json(filename, encoding=None):
    """
    Load data from a JSON file.
    Args:
        filename (str): The path to the JSON file to be loaded.
    Returns:
        dict: The data loaded from the JSON file, or None if an error occurs.
    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    #check if it is 'utf-8-sig'
    encoding = check_utf_encoding(filename)

    try:
        with open(filename, 'r', encoding=encoding) as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")
        raise
    except UnicodeDecodeError:
        print(f"Error: The file {filename} cannot be decoded with {encoding} encoding.")
        raise
    except json.JSONDecodeError:
        print(f"Error: The file {filename} contains invalid JSON.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise




def save_to_json(data, filename):
    """
    Save data to a JSON file.
    Args:
        data (dict): The data to save.
        filename (str): The path where the JSON file will be saved.
    Raises:
        TypeError: If the data provided is not serializable.
        IOError: If there are issues writing to the file.
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    except TypeError as e:
        print(f"Error: The data provided contains non-serializable types.")
        raise
    except IOError as e:
        print(f"Error: Unable to write to the file {filename}.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

###################################################################



#############################################################################
#gradient ecking
#############################################################################
def check_grads(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                print(f"NaN gradient at {name}")



def check_grads_and_show_stats(model):
    print("Gradient Statistics:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_min = grad.min().item()
            grad_max = grad.max().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            has_nan = torch.any(torch.isnan(grad)).item()

            # Printing gradient stats
            msg = 'contains NaNs' if has_nan else ''
            print(f"{name}: Min:{grad_min}, Max:{grad_max}, Mean:{grad_mean}, Std: {grad_std}, {msg}")



# Function to calculate the gradient norms
def calculate_gradient_metrics(model):
    grad_norms = {}
    grad_stdevs = {}
    max_grads = {}
    has_nans = {}
    total_norm = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            grad_data = parameter.grad.data.view(-1)  # Flatten the gradient data for easier computation
            param_norm = grad_data.norm(2)  # L2 norm
            param_stdev = grad_data.std()  # Standard deviation
            max_grad = grad_data.abs().max()  # Maximum absolute value
            has_nan = torch.any(torch.isnan(grad_data))

            grad_norms[name] = param_norm.item()
            grad_stdevs[name] = param_stdev.item()
            max_grads[name] = max_grad.item()
            has_nans[name] = has_nan.item()
            total_norm += param_norm.item() ** 2  # Sum of squares for the overall norm

    total_norm = total_norm ** 0.5  # Compute the square root of the total norm
    return grad_norms, grad_stdevs, max_grads, has_nans, total_norm

###################################################################


###################################################################
#load and save model weights code (from my spert code, maybe will itegrate this, I know it works)
###################################################################

def save_weights(model, model_dest, model_name, full=False):
    #save it
    if full:
        #save the full model
        model_path_full = model_dest + '/' + model_name + '_full.pth'
        os.makedirs(model_dest, exist_ok=True)
        torch.save(model, model_path_full)
    else:
        #save the state dict only
        model_path_full = model_dest + '/' + model_name + '_wt.pth'

        print(model_path_full)
        os.makedirs(model_dest, exist_ok=True)
        torch.save(model.state_dict(), model_path_full)



def load_weights(model_name, model_source, full=False, new_model=None, device=None):
    if full:
        #load the full model
        model_path_full = model_source + '/' + model_name + '_full.pth'
        if os.path.exists(model_path_full):
            new_model = torch.load(model_path_full, map_location=device)
        else:
            return None, 'no model file'
    else:
        if new_model is None:
            raise ValueError("new_model must be provided if full=False")
        #load the state dict only
        model_path_full = model_source + '/' + model_name + '_wt.pth'
        if os.path.exists(model_path_full):
            new_model.load_state_dict(torch.load(model_path_full, map_location=device))
        else:
            return None, 'no model file'

    return new_model, None
###################################################################




def remove_span_types_from_full_rels(full_rels):
    '''
    This removes the span types from the full rels as this is required for some analysis
    NOTE: as there could be doubles, we set and list each batch obj list
    '''
    return [list(set([(rel[0],rel[1],rel[3],rel[4],rel[6]) for rel in obs])) for obs in full_rels]
