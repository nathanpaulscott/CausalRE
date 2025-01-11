

def config_validator(config):
    '''
    this validates the contents of config
    '''
    #check that the span_labels and rel_labels value is valid
    #DESCRIPTION ON WHY i AM ONLY ALLOWING SPANS TO BE UNILABEL
    #I changed it to only allow span_labels to be unilabel as it makes the final rel predictions to complicated to extract if spans are multilabel
    #and spans never need to be multilabel.  The issue is that the annotation format encodes the span type and rel type for each rel in a convoluted way
    #this is NOT maintained in the data that comes out of the model, nor the model internal workings, i.e. the predicted rel type gives us the head and tail cand_span_ids, but this only 
    #allows us to get the head/tail start,end positions, but for a multilabel case it allows us to get the binary vector of head/tail span type preds, not
    #just one pred for each as in the unilabel case, thus we could have more than 1 type for the head or tail span, thus the way data is stored in the model
    #allows this uncertainty.  To avoid this uncertainty we would actually have to encode each predicted span type into the rel_reps etc and that is NOT what we are doing
    #it woudl be very impractical to do it also as we woudl expand the rel_reps even more for the quadratic of num_head_types * num_tail_types * top_k_spans**2!!!!.  So do not even think about it. 
    #The most practical solution is to just fix the span labels to unilabel so that each span can only have one label, this solves that complexity problem and it reflects the reality also.
    if config.span_labels != 'unilabel' or config.rel_labels not in ['unilabel', 'multilabel']:
    #if config.span_labels not in ['unilabel', 'multilabel'] or config.rel_labels not in ['unilabel', 'multilabel']:
        raise ValueError('the span_labels must be "unilabel" and rel_labels can only be "unilabel" or "multilabel", exiting.....')

    #check that there are span_Types adn rel_types
    if ((config.span_types is None or not isinstance(config.span_types, list) or len(config.span_types) == 0) or
        (config.rel_types is None or not isinstance(config.rel_types, list) or len(config.rel_types) == 0)):
        raise ValueError('TTER needs to ba one or more span_types and rel_types in the schema and it should match the annotated data types, exiting.....')

    #check that none_span and none_rel do not exist as a type in the schema
    if 'none' in config.span_types or 'none' in config.rel_types:
        raise ValueError('none is a reserved type for the negative classification, it should not be in the schema, exiting.....')
