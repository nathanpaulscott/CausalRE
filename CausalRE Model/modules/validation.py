








def config_validator(config):
    '''
    this validates the contents of config
    '''
    #check that the span_labels and rel_labels value is valid
    if config.span_labels not in ['unilabel', 'multilabel'] or config.rel_labels not in ['unilabel', 'multilabel']:
        raise ValueError('the span_labels and rel_labels can only be "unilabel" or "multilabel", exiting.....')

    #check that there are span_Types adn rel_types
    if ((config.span_types is None or not isinstance(config.span_types, list) or len(config.span_types) == 0) or
        (config.rel_types is None or not isinstance(config.rel_types, list) or len(config.rel_types) == 0)):
        raise ValueError('TTER needs to ba one or more span_types and rel_types in the schema and it should match the annotated data types, exiting.....')

    #check that none_span and none_rel do not exist as a type in the schema
    if 'none' in config.span_types or 'none' in config.rel_types:
        raise ValueError('none is a reserved type for the negative classification, it should not be in the schema, exiting.....')
