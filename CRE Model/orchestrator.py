import os
from pathlib import Path
from datetime import datetime


import start
from modules.utils import join_paths, get_filename_no_extension, get_filename, ensure_clean_directory, copy_file, move_file, load_from_json, confirm_action

# Call the main function explicitly
#start.main()

class Orchestrator():
    def __init__(self, experiments_folder, experiments_file, base_config_path, train=True, predict=True):
        #get the app path
        self.base_path = str(Path(__file__).parent)
        os.chdir(self.base_path)
        #set the experiments path
        self.experiments_path = join_paths(experiments_folder, experiments_file)
        self.experiments_folder = experiments_folder
        self.experiments_file = experiments_file
        #set the base config file 
        self.base_config_path = base_config_path
        self.train = train
        self.predict = predict



    def make_new_config(self, experiment, seed=None, first_seed=True, train=False):
        #load the base config dict
        config_dict = start.load_config(self.base_config_path)
        #get the experiment unique name
        experiment_name = experiment['name']
        experiment_name_w_seed = f"{experiment['name']}_{seed}"

        #edit the configs
        config_dict['model_name'] = f"model_{experiment_name_w_seed}"    
        config_dict['random_seed']    = seed
        config_dict['predict_folder'] = experiment['folder']
        config_dict['log_folder']     = experiment['folder']         #new log file will be saved here  
        config_dict['model_folder']   = experiment['folder']       #new model will be saved here  
        for param in experiment['params']:
            config_dict[param] = experiment['params'][param]

        #fix the dataset if it is to use a seed specific one (i.e. if it has a .json extension, just use that)
        if config_dict['data_path'][-5:] != ".json":
            config_dict['data_path'] = f"{config_dict['data_path']}{seed}.json"       
        
        #do the train/predict specific params
        if train:
            #train case
            config_dict['run_type']   = 'train'
            config_dict['log_name']   = f"log_{experiment_name_w_seed}_train"    
            config_dict['model_load'] = False
            config_dict['model_save'] = True,   #True if first_seed else False
            config_file = f'config_{experiment_name}_train.yaml'
            config_path = join_paths(experiment['folder'], config_file)
        else:
            #predict case
            config_dict['run_type']   = 'predict'
            config_dict['log_name']   = f"log_{experiment_name_w_seed}_pred"    
            config_dict['model_load'] = True
            config_dict['model_save'] = False
            config_file = f'config_{experiment_name}_predict.yaml'
            config_path = join_paths(experiment['folder'], config_file)

        #save the modified config file
        start.save_config(config_dict, config_path)

        return config_path



    def do_training(self, experiment, seed, first_seed):
        '''
        train the model given the specific config file
        '''
        print("\n####################################################")
        print(f"Starting training on model: {experiment['name']}, seed: {seed}")
        config_path = self.make_new_config(experiment, seed=seed, first_seed=first_seed, train=True)
        ###################################################
        #run the training with return_data=True to get the log_file_path
        start.run_model(config_path)
        ###################################################
        print(f"Done training on model: {experiment['name']}, seed: {seed}")
        print("####################################################\n")



    def do_predict(self, experiment, seed):
        '''
        predict with given model
        '''
        print("\n####################################################")
        print(f"Starting predict on model: {experiment['name']}, seed: {seed}")
        config_path = self.make_new_config(experiment, seed=seed, train=False)
        ###################################################
        #run the model in predict mode using the modified config
        start.run_model(config_path)
        ###################################################
        print(f"Done predict on model: {experiment['name']}, seed: {seed}")
        print("####################################################\n")


    def clear_colab_output(self):
        """Clear the output of the current cell in Google Colab."""
        try:
            from IPython.display import clear_output
            print('cleaning output')
            clear_output(wait=True)
        except ImportError:
            print("Clear output not supported in this environment.")


    def start_experiments(self):
        #open the experiments file
        data = load_from_json(self.experiments_path)
        #get the seeds list
        train_seeds = data['seeds']
        predict_seed = data['seeds'][0] 
        #Run the model for each experiment
        #if not confirm_action('Continue?  All experiment folders will be cleared and the experiments redone.'):
        #    print('cancelling')
        #    return
        
        for experiment in data['experiments']:
            experiment_name = f"{experiment['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            experiment['folder'] = join_paths(self.experiments_folder, experiment_name)
            if self.train:
                #do the training
                ###################################################
                #reset the particular experiment folder
                ensure_clean_directory(experiment['folder'])
                first_seed = True
                for seed in train_seeds:
                    self.do_training(experiment, seed, first_seed)
                    first_seed=False
                ###################################################

            if self.predict:
                ###################################################
                self.do_predict(experiment, predict_seed)    
                ###################################################

            self.clear_colab_output()

        exit()




if __name__ == "__main__":
    #Instantiate the class with the path to the application
    base_config_path = "config.yaml"
    experiments_folder = 'experiments'
    train_flag = True
    predict_flag = True

    #experiments_file = 'experiments-rel pooling.json'
    #experiments_file = 'experiments-rel pooling-spanattn-relcrossattn.json'
    #experiments_file = 'experiments-teacher forcing.json'
    #experiments_file = 'experiments-graph-lstm_temp.json'
    #experiments_file = 'experiments-separate-bert.json'
    #experiments_file = 'experiments-baseline.json'
    #experiments_file = 'experiments-marking.json'  #doesn't work
    #experiments_file = 'experiments-topk.json'
    #experiments_file = 'experiments-rel context.json'
    #experiments_file = 'experiments-backbones.json'
    #experiments_file = 'experiments-spanbert-rel calc.json'
    #experiments_file = 'experiments-spanbert_rel_calc_temp_tf.json'
    #experiments_file = 'experiments-spanbert_orig_config.json'
    #experiments_file = 'experiments-spanbert-semeval-conll04.json'
    experiments_file = 'experiments-bert-conll04.json'
    
    orchestrator = Orchestrator(experiments_folder, experiments_file, base_config_path, train=True, predict=True)

    #kick off the experiments
    orchestrator.start_experiments()



'''
so you need th ebase config in the app root as config.yaml
you also need a fil in the experiments folder called experiments.json
This is a list of dicts one item per experiement
keys are:
'name' = the experiment name, must be unique
'seeds' = a list with the seeds to use, model will train 1x for each seed
'params' = a dict with all the params to change and their their values for the experiment, the base config will be edited with these params
'''
