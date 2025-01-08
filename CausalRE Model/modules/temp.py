import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


'''
REMEMBER THAT EVEN THOUGH i HAVE A TEACHER FORCING OPTION, IF i TURN IT OFF, POS SPANS AND THUS RELS ARE NOT GUARANTEED TO BE IN THE CAND_SPANS AND CAND_RELS, ESSPECIALLY PROBLEMATIC FOR THE
CANDIDATE SPANS AS IF A POS SPAN IS LEFT OUT, THEN THE REL THAT HAS THAT SPAN AS HEAD OR TAIL NEVER GETS THE CHANCE TO GET PREDICTED ON.  MY SOLUTION TO THIS WAS PENALISE THE MODEL LOSS FOR THE SPAN HEAD AND REL HEAD
I GUESS IN THIS MODEL, WE WOULD PENALISE THE FILTER LOSS POTENTIALLY, OR MAYBE ADD THE PENALITY TO ALL LOSSES => SEE WHAT i DID IN SPERT....  LOOK FOR MISSED SPANS AND MISSED RELS 

So you need to just go back through flow in spert and try to understand what is going on so you can adpat it to this
1) loss => no real probs as I shadow the reps with labels and masks, so I think ok, but check again

2) preds => easy, see the spert code

3) metrics => this requires formatted and aligned preds and labels, I think not so hard as the labels and masks shadow the logits, so they remain aligned throughout for spans and rels
read the new model, but it is ok.  span_reps, span_masks, spans_ids and span_labels are aligned, then we shortlist to cand_span_reps/ids/masks/labels (all aligned), then we form all possible rels from this
thus we form rels_reps/ids/masks/labels (all aligned), then we filter and shortlist to cand_rel_reps/ids/masks/labels (all aligned) so getting loss and metrics requires no alignment.


This code is from my spert version
It may not be relevant, but some ideas are....

it has prep data for metrics code (for the spans case (easy) and for the rels case (hard with alignment) using sets (good ideas from spert)

it also has some prep code for loss for rels, again needing alignment

'''


#How to get the preds that would be sent into the prep_data_for_metrics
#the labels just need ot be detached
'''
#calc preds/probs
preds_E = torch.argmax(F.softmax(logits_Eg.detach(), dim=-1), dim=-1)

############################################3

preds_R = torch.sigmoid(logits_Rg.detach())
preds_R = (preds_R >= params['gp'].relation_thd).float()



#calc metrics
preds, labels = prep_data_for_metrics_LE(preds_E, labels_Eg.detach(), ignore_mask_Eg)
metrics_E = quick_metrics(preds, labels, params=params)

##############################################

preds, labels = prep_data_for_metrics_R(preds=preds_R, labels=labels_Rg.detach(), ignore_mask_preds= ~result['pair_mask'], ignore_mask_labels=ignore_mask_Rg, pred_pair_map=result['pair_map'], label_pair_map=label_pair_map, preds_E=preds_E, labels_E=labels_E, val_flag=False, params=params, mode='ntf-train')
metrics_R = quick_metrics(preds, labels, params=params)

'''



def quick_metrics(preds, labels, params=None):
    '''
    this gets the current overall metrics for use during training

    inputs:
        - aligned, cpu'd, numpy'd, flattened, filtered preds list
        - aligned, cpu'd, numpy'd, flattened, filtered labels list
        - params object

    outputs:
        - metrics dict

    '''
    if len(labels) == 0:
        accuracy, precision, recall, f1 = 0,0,0,0
    else:
        # Calculate overall  metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=params['f1_ave'], zero_division=0)

    return dict(Support   = len(labels),
                Accuracy  = accuracy,
                Precision = precision,
                Recall    = recall,
                F1        = f1)




def prep_data_for_metrics_LE(preds, labels, ignore_mask):
    '''
    Flattens the predictions and labels, moves them to the CPU, converts to numpy arrays,
    and filters out elements based on the ignore mask.
    only have one ignore mask as preds/labels are always aligned, so the ignore mask is the same for both
    '''
    # Flatten tensors, move to CPU, and convert to numpy arrays
    preds_flat = preds.view(-1).cpu().numpy()
    labels_flat = labels.view(-1).cpu().numpy()
    ignore_flat = ignore_mask.view(-1).cpu().numpy()

    #Keep elements that are False in the ignore mask (so invert the ignore mask)
    labels_filt = labels_flat[~ignore_flat]
    preds_filt = preds_flat[~ignore_flat]

    # Check if lengths of filtered predictions and labels are the same
    if len(preds_filt) != len(labels_filt):
        raise ValueError('Predictions and labels have different lengths after filtering!')

    return preds_filt, labels_filt





def prep_data_for_metrics_R(preds, labels, ignore_mask_preds, ignore_mask_labels, pred_pair_map, label_pair_map, preds_E, labels_E, val_flag, params, mode):
    '''
    This needs to do relation alignment along with moving to the cpu and filtering by the ignore mask
    It converts the vect format preds/labels to relations first then does the merge

    tf mode => the preds and labels are aligned always, so the label_ignore mask is sent in for both pred and label

    ntf mode => the preds and labels are not aligned, so the ignore mask will be different for preds and labels

    Outputs:
    - cpu'd, flattened, filtered, aligned R preds list => just a list of rel-type indices, eg. [1,0,3,4,2,1,4,2,2,2,4,4,4,4,4]
    - cpu'd, flattened, filtered, aligned R labels list => just a list of rel-type indices, eg. [1,0,3,4,2,1,4,2,2,2,4,4,4,4,4]
    '''
    none_rel_idx = params['gp'].neg_idx_R

    # Convert ignore masks to boolean masks for valid entries
    if mode in ['tf', 'ntf']:
        valid_labels_mask = (ignore_mask_labels == 0)
        valid_preds_mask = (ignore_mask_preds == 0)
    else:   #mode == 'ntf-train', we need to merge the label_ignore_mask with pred_pair_map
        valid_labels_mask = (ignore_mask_labels == 0)
        valid_preds_mask = create_special_valid_pred_mask(label_pair_map, pred_pair_map, ignore_mask_labels)

    # Helper function to process either preds or labels
    def process_relations(data, pair_map, entity_types, valid_mask):
        batch_size, seq_len, _ = data.shape
        relations = []
        # Iterate over each observation
        for i in range(batch_size):
            obs_relations = []
            for j in range(seq_len):
                if not valid_mask[i, j]:
                    continue
                head_idx, tail_idx = pair_map[i, j]
                head_type = entity_types[i, head_idx]
                tail_type = entity_types[i, tail_idx]
                # Identify relation types
                rel_types = (data[i, j] > 0).nonzero(as_tuple=True)[0]
                if rel_types.numel() == 0:
                    obs_relations.append((head_idx.item(), head_type.item(), tail_idx.item(), tail_type.item(), none_rel_idx))
                else:
                    for rel_type in rel_types:
                        obs_relations.append((head_idx.item(), head_type.item(), tail_idx.item(), tail_type.item(), rel_type.item()))
            relations.append(obs_relations)
        return relations

    # Process predictions and labels
    preds_rel = process_relations(preds, pred_pair_map, preds_E, valid_preds_mask)
    labels_rel = process_relations(labels, label_pair_map, labels_E, valid_labels_mask)

    # Flatten and merge operations

    #temp
    temp_p, temp_l, temp_t, cnt = 0, 0, 0, 0
    #temp

    preds_flat, labels_flat = [], []
    for preds_obs, labels_obs in zip(preds_rel, labels_rel):
        # Creating sets for unique relations per observation
        preds_set = set(preds_obs)
        labels_set = set(labels_obs)
        union_set = preds_set | labels_set

        #temp
        cnt+=1
        temp_p+=len(preds_set)
        temp_l+=len(labels_set)
        temp_t+=len(union_set)
        #temp

        for relation in union_set:
            #do preds
            if relation in preds_set:
                preds_flat.append(relation[-1])
            else:
                preds_flat.append(none_rel_idx)
            #do labels
            if relation in labels_set:
                labels_flat.append(relation[-1])
            else:
                labels_flat.append(none_rel_idx)

    if val_flag:
        #temp
        print(f'pred_set_ave_size: {round(temp_p/cnt,2)}, label_set_ave_size: {round(temp_l/cnt,2)}, union_set_ave_size: {round(temp_t/cnt,2)}')
        #temp

    return preds_flat, labels_flat












def create_special_valid_pred_mask(label_pair_map, pred_pair_map, ignore_mask):
    device = label_pair_map.device
    zero_pair = torch.tensor([0, 0], dtype=torch.long, device=device)

    # Find the length of non-zero entries in the label_pair_map across the batch
    non_zero_label_mask = ~(label_pair_map == zero_pair).all(dim=2)
    non_zero_label_lengths = non_zero_label_mask.sum(dim=1)

    # Create a combined mask for the entire batch
    batch_size, max_len, _ = pred_pair_map.shape
    valid_pred_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    # Iterate through each batch to set the mask correctly
    for i in range(batch_size):
        non_zero_length = non_zero_label_lengths[i].item()
        # Inverse ignore mask up to non_zero_length
        valid_pred_mask_first_part = (~ignore_mask[i])[:non_zero_length]
        # Non-zero entries in the pred_pair_map starting from non_zero_length
        valid_pred_mask_second_part = (~(pred_pair_map[i] == zero_pair).all(dim=1))[non_zero_length:]
        # Combine the two parts of the mask for each batch item
        valid_pred_mask[i, :non_zero_length] = valid_pred_mask_first_part
        valid_pred_mask[i, non_zero_length:non_zero_length + valid_pred_mask_second_part.size(0)] = valid_pred_mask_second_part

    return valid_pred_mask





def create_valid_masks(label_pair_map, pred_pair_map, ignore_mask=None):
    device = label_pair_map.device
    zero_pair = torch.tensor([0, 0], dtype=torch.long, device=device)

    # Create masks for non-zero pairs across the entire batch
    non_zero_label_mask = ~(label_pair_map == zero_pair).all(dim=2)
    non_zero_pred_mask = ~(pred_pair_map == zero_pair).all(dim=2)

    if ignore_mask is None:   #ntf mode, no neg sampling
        valid_label_mask = non_zero_label_mask
        valid_pred_mask = non_zero_pred_mask
    else:    #ntf-train mode, so has neg sampling
        valid_label_mask = ~(ignore_mask)
        valid_pred_mask = create_special_valid_pred_mask(label_pair_map, pred_pair_map, ignore_mask)

    return valid_label_mask, valid_pred_mask




def prep_data_for_loss_R(logits, labels, pred_pair_map, label_pair_map, ignore_mask=None):
    '''
    Aligns logits and labels based on pair mapping, handling pairs unique to each source.
    Processes each batch element independently.
    '''
    no_match_vect = torch.zeros(logits.size(2), dtype=logits.dtype, device=logits.device)
    batch_size = logits.size(0)
    all_logits_flat = []
    all_labels_flat = []

    # Compute valid masks for the entire batch
    valid_label_mask, valid_pred_mask = create_valid_masks(label_pair_map, pred_pair_map, ignore_mask)

    for i in range(batch_size):
        valid_pred_pairs = pred_pair_map[i][valid_pred_mask[i]]
        valid_logits = logits[i][valid_pred_mask[i]]
        valid_label_pairs = label_pair_map[i][valid_label_mask[i]]
        valid_labels = labels[i][valid_label_mask[i]]

        # Concatenate and find unique pairs
        all_pairs = torch.cat([valid_pred_pairs, valid_label_pairs], dim=0)
        unique_pairs, inverse_indices = torch.unique(all_pairs, return_inverse=True, dim=0)
        # Indices in concatenated unique pairs
        pred_indices = inverse_indices[:len(valid_pred_pairs)]
        label_indices = inverse_indices[len(valid_pred_pairs):]
        # Initialize tensors for aligned logits and labels
        logits_flat = no_match_vect.repeat(len(unique_pairs), 1)
        labels_flat = no_match_vect.repeat(len(unique_pairs), 1)
        # Populate logits and labels where actual data exists
        logits_flat[pred_indices] = valid_logits
        labels_flat[label_indices] = valid_labels
        # Store processed tensors
        all_logits_flat.append(logits_flat)
        all_labels_flat.append(labels_flat)

    # Concatenate all batch elements
    logits_flat_final = torch.cat(all_logits_flat, dim=0)
    labels_flat_final = torch.cat(all_labels_flat, dim=0)

    return logits_flat_final, labels_flat_final


#prep data for loss R is needed in ntf mode as there is no guarantee that positive label cases are in the preds (non teacher forced!!)  So we have to do some alignment
#we call it like this before we run loss
'''
flat_logits, flat_labels = prep_data_for_loss_R(logits_Rg, labels_Rg, result['pair_map'], label_pair_map, ignore_mask_Rg)
loss_R = loss_fn_R(flat_logits, flat_labels).mean()
'''

'''
The alignment problem explained
-------------------------------
facts:
- the model works in 2 modes => tf and ntf
- the spans are not affected by alignment issues as they work form gp.amx_spans which is a list of all possible spans within max_seq_len regardless of tf mode or ntf mode
- we only have to deal with the alignment problem for pairs/relations

Spans
-----------
tf and ntf mode work the same in the model, it just gets the representations for all possible spans (from gp.max_spans)
and masks the ones that are invalid based on the span_mask, but the order of the logits and labels remains the same, it doesn't change
So we can effectitvely run loss and metrics on the logits and labels without any extra alignment steps, we only have to apply the ignore mask prior to the loss/metrics
span ignore mask
    for tf mode, the ignore mask should mask out all non chosen neg samples(spans) as well as pad samples(spans), i.e. set them to -100 as that is the ignore value we use in the CELoss and metrics
    for ntf mode, the ignore mask should only mask out the pad samples as we do not need to use negative sampling (check spert), just use the inverse span_mask!!

################################
Modification to do
NOTE: to speed up training, you could modify the tf mode to send the span_map to the model and the model then use this to only generate span reps for the pos/chosen neg samples and pad to the batch
However, if you did this, you would need to generate the labels to conform to the same span_map for each obs so that the logits and labels would be aligned correctly.  I will do this later, for now I will continue as is
Obviosuly in the model, you would then have an if statement for the spans also, with tf mode using the incoming span_map and ntf mode just using all possible spans from gp.max_spans
################################

Relations
--------------
relations are more complex due to several factors, they derice from the entities and also one pair can map to multiple relations, so there is alignment work to be done
TF mode => we send the pair_map into the model and the model makes pair representations only for these pairs and classifies them, so the logits will be aligned to the labels as both are derived from the same pair_map
    Loss => Thus, in tf mode for the loss calc, there is no need to align the logits and labels by pair, we just need to apply the ignore mask to both of them masks out non-chosen pairs and pad pairs.
            Currently we set the masked logits to -1e9 and labels to 0, later I will follow spert, by using no reduce in the loss function and applying the mask AFter the loss
            I think this change is critical for the relations using BCELoss
    Metrics => pair alignment is not enough for the metrics, as we are comparing relations, not pairs,
            we need to convert the vect format labels and preds to relations tuples => (start, end, headtype, start, end, tailtype, reltype)
            we then use the set method (from spert) to align these relation sets, adding missing tuples to each set with the reltype - NoneRelation,
            then extracting just the reltype from each tuple for each list and using that for the metrics.
            Call this relation alignment
NTF mode => we do not send the pair map to the model, we generate the pred_pair_map from the predicted spans.  In this case we have to do pair alignment and relation alignment:
    Loss => we need to do pair_map alignment between the logits and labels, where there is a missing pair in each list add it with the pred/label => all zero vect  (check spert for this, I know they do it for the relation alignment)
            note: you will have to also adjust the ignore map and pair_map and pair_mask etc during this process
            then we just put the aligned logits/labels/ignore mask into the BCELoss and also we should make the update to the loss to use no reduction as described above
            NOTE: the ignore mask should have just pad pairs, no neg sampling for ntf mode
            This is the most complex part....
    Metrics => same as for the tf case, we need to do relation alignment, this should be no different from the tf case though as the preds and labels are already aligned.

###################################
###################################
###################################
So in summary I will make it do this.....
for tf mode
    we do not need to pair align for loss as it is already aligned,
    for metrics we need to relation align => NOTE: make it so we can do metrics or disable train metrics for speed
for ntf mode
    we pair align for loss (again make it so I can disable loss and pair align for speed),
    for metrics, we need to relation align, we must do metrics, so this cannnot be turned off
    NOTE: make it so that the pair align for the loss is not done on the data that goes to the relation align, if you did that we would get double align and that is not good, i.e. split off data and pair align it for loss, do not edit the main data pipe
    similarly, split off the data for matric relation align, do not edit the main data
###################################
###################################
###################################

'''