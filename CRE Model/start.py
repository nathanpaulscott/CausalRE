import torch
import argparse, yaml, logging, os
from pathlib import Path
from types import SimpleNamespace

from modules.train import Trainer
from modules.config_manager import Config
from modules.logger import Logger




def create_parser():
    '''
    create the arg parser, allows specifying the config file and the log file location
    '''
    parser = argparse.ArgumentParser(description="Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser




def load_config(config_path):
    """
    Load a YAML configuration file to a config object containing a namespace
    """
    if not config_path.exists():
        raise FileNotFoundError(f"The configuration file {str(config_path)} does not exist.")
    
    try:
        with config_path.open("r") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise
    
    if config_dict is None:
        raise ValueError('the config is blank, exiting....')
    
    return config_dict




if __name__ == "__main__":
    '''
    This is the main kick off code for training
    setup the arg parser and then read in the configs
    then instantiate the trainer which takes in the config and internally sets up the model, reads the dataset, and trains/evals/infers the model
    When we call the model, we call it in model.train() mode for train and model.eval() with no grad for eval (val and test)
    We have a 3rd mode model.eval(), and no grad with no labels which is used for the prediction class, see comments below
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'

    #for debugging only
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #os.environ['TORCH_USE_CUDA_DSA'] = "1"

    #get the configs
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        #load the config file to a dict
        app_path = Path(__file__).parent
        config_path = app_path / args.config
        config_dict = load_config(config_path)
        #make the config object
        main_configs = Config(config_dict)
        main_configs.update({'app_path':        app_path, 
                             'device':          device, 
                             'torch_precision': torch.float16 if config_dict['num_precision'] == 'half' else torch.float32,
                             'num_limit':       1e4 if config_dict['num_precision'] == 'half' else 1e10})
    except Exception as e:
        raise Exception(f"Failed to load and parse the configuration: {e}")

    #make the config as namespace and set the cwd
    config = main_configs.as_namespace
    os.chdir(config.app_path)
    
    #set up the logger
    log_folder = str(config.app_path / config.log_folder)
    logger = Logger(log_folder, enable_console_output=config.print_log)
    logger.write('Start...')
    #add logger to main_configs
    main_configs.update({'logger': logger})

    #make the Trainer object to orchestrate the training/inference
    trainer = Trainer(main_configs)
    #run the trainer
    trainer.run()




'''
The key concept is that long span detcection doest work with short span techniques and unaltered bert embeddings
Aslo relation extraction doesn't work with unaltered bert embeddings
Fro rel extraction, you needto mark the 2 spans and enrich the token embeddings through bert or an MHA layer then form the rel reps,it works much better => see unicausals technique!!!
Fro spans, this is a much harder problem to soplve, but potentially trigger id first, then enrich token reps w.r.t the trigger then search for slong spans withn a windoew aroudn the trigger etc using the enriched reps w.r.t. that trigger
This one is much mroe sketchy and a really hard problem => potentiallya token tagging approach after enriching the tokens for each trigger => that might be the way to go => good idea
Write all this up, this is the key stuff and plan out how to implememnt at least the relation one, I think the span one you may be able to fuck it off, but potentially the token tagging approach on enriched trigger specific token embeddings may bear some fruit













It just can't get the spans if they are long due to noise, even humans struggle
It is much better at learning short trigger span extraction
However relations are shite whether trigger (short span) or full spans (long span)
Either the problem is just that there is no clear signal (most likely) or the annotations are shite (also likely)





OK, I am trying to generate/autoannotate a dataset to train my model.  I think I will go with the current form and just adapt it to longer sequences and wider spans
Annotation:
1) gemini to extract events from maven train
2) gemini to determine causality between extracted events
3) clean up the dataset and split
4) gemini to find full event spans from the event triggers in semeval, it alreayd has the causality annotations so that is all
5) combine with maven




Model
----------
adapt it to handle longer sequences:
1) integrate bigbird
2) chunk the span search, instead of just making the span reps (batch, num possible spans, hidden)
chunk it to span batches of say 2-3K spans or whatever works well and process each in sequence, then merge filter scores before sorting and selecting the top_k_spans
the losses you just add for each chunk.
NOTE: the problem here is that this only helps the foward pass, not the backwards pass, there is no way to chunk the backwards pass!!!!!!
- You can used mixed precision training
- you can use gradient checkpointing whcih slows down the backward pass, but is stable, solves the backward memory issues:
eg.

no checkpointing
def forward(self, inputs):
    total_loss = 0
    for chunk in chunks:
        logits = filter_head(chunk)  # Standard forward pass (stores activations)
        loss = loss_fn(logits, labels)
        total_loss += loss
    return total_loss + pred_loss  # Standard backward pass uses more memory

using checkpointing
------------------------------
import torch.utils.checkpoint as checkpoint

def forward(self, inputs):
    total_loss = 0
    for chunk in chunks:
        logits = checkpoint.checkpoint(filter_head, chunk)  # Recomputes activations
        loss = loss_fn(logits, labels)
        total_loss += loss
    return total_loss + pred_loss  # Still a single loss, but backward uses less memory


No, you don’t need to checkpoint every layer.
Best practice: Checkpoint every 2-3 layers or memory-heavy layers only.
This balances memory savings & training speed.
---------------------------------------------------------------

Look at the 2 stage event span search procedure:
---------------------------------------------------
Two-Stage Event Extraction Process (Trigger + Windowed Span Search)
Stage 1: Trigger Detection
Detect event triggers (short spans, usually verbs/nouns).  binary trigger filter head
Requires trigger annotations for training.
Output: (batch, num_triggers, 2) tensor → (start, end) indices of detected triggers.
Stage 2: Windowed Span Search
For each trigger, define a search window:
window_start = trigger_start - max_width
window_end = trigger_end + max_width
Filter spans from span_ids (batch, 100K, 2):
Keep only spans within the search window.
Use a mask tensor (batch, 100K) for efficient filtering.
Extract valid spans from span_reps for classification.
✅ Key Benefits:

Massively reduces search space (no need to check all spans).
Maintains recall while optimizing speed/memory.
Simple heuristic, easy to tune (max_width).
🚀 This makes LSLS event extraction feasible for long sequences.







with full TF, I am getting 87, 67, 68, which is better than turning off TF at some point => prob the best
full TF no graph transformer 85, 60, 64
full TF only on the span filter, no TF on teh rel filter => 86,68,69... probably the best setup
TF on span only to 1000 then no TF, no TF on rel => 86, 65, 69 => almost as good as TF
TF on span only to 1000 then no TF and no lost rel loss, no TF on rel => 85, 64, 67 => so gets close wihtout all that crap
TF to 100 then all off, only span and rel class loss => real bad, did not train

***Basically, it trains with or without TF or lost rel loss, so both of them are redundant, but full TF seems to give it a boost, so it is still a good thing


#test some settings on this config
TF span always, TF rel never, batch of 2 or 4 what ever GPU allows, no accumulation
--------------------------------
!!!!Seems like the top_k can be quite low, just make sure it is above the number of pos cases per obs, say 30 or 40
top_k_spans/rels = 50, 50 => 86, 68, 72  
top_k_spans/rels = 40, 40 => 86, 67, 70  => this is good as it runs faster and uses less memory
top_k_spans/rels = 30, 30 => (87, 68, 72) => batch of 4

try single for filter heads
top_k_spans/rels = 40, 40 => 87, 70, 72  => so single works as good or better

try zero embedding for no context in the rels => makes no difference

rel type to no_context => 88, 67, 69 => rel f1 is not as high
rel_type window_context => 87, 67, 70 => between was better

rel_type between, span firstlast => 86, 71, 74   GOOD
rel_type between, span_type nathan => 87, 72, 74!!!!   Highest
token pooling set to first_last => 88, 72, 74 => also good

##################
lstm => 87, 74, 75 => best so far
##################

increase span filter loss x 10 added 1% to span F1


Best seems to be:
accum always off
batch 4 or 8
span rep => nathan
rel_rep => between
token pooling maxpool/firstlast
rel thd => 0.3
filter percentile 30
filter type => single for all (works better than double for some reason)
include graph transformer
inlcude lstm
top_k_spans, rels => 30/30 just make sure is higher than the max num spans/rels x 2
graph heads/layers 4/2 but more is better
TF always on for span filter, never for rel filter n graph filter
use all losses
no prompting, but it does work also



Build the spert model as a separate model.py
Think about how to build the event trigger-role-relation model.py



########################
Make another model whcih mimics spert, i.e. only the span and rel classification heads, random neg sampling
the output of the span class head is binaried to is span and is not span and the rels are built from the is span rels

See if the perf is close to what we have now
########################





I would do a config to use random neg sampling for both span and rel filter to take out the loss there and bypass the grpah
see if you can get similar numbers





lost rels are when the actual labels for rels do not make it to the universe of possible rels, so we can not account for them in the rel filter loss
thus I add the lost rel loss.  However, this doesn't help the rel classifier head as it will not be able to predict on those lost rels





looks like using token pooling is key, that works way better than no pooling

prompting works, but I seems to get better results with no prompting

using all losses
no lstm => test again
maybe need to try and recheck everythign now that I have a config that I knwo works
graph trans yes/no => I woudl say yes, but test
filter types => try again
pos weights => try again
score qualtile => have at 0 right now to allow as many as poss in
span rep => spert seesm best
top k spans, top k rels  (64,100) try 1000 for rels, spans can't go up much
rel pred thd seems to be better lower, 0.1 or even 0 so that it is easier to ppredict a rel => to be tested



some problem with the binary class weights, they should be just a single value tensor

'''