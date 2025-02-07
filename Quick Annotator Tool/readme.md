# Quick Annotation Tool
This is an html/css/js based single page app for quick annotation.  The focus is on efficiency of the annotation process, which is a repetitive human task.  Every mouse movement, mouse click, keyboard stroke and delay matters.  All data is stored in-memory and imported/exported in .json format.

## Imports
The tool is simple, you can load in a set of unannotated or pre-annotated documents, samples are given.   

The json format is as follows:   
An object with 2 keys: 'data', 'schema'.   


data key contains a list of observations/documents, each observation is an object with 3 keys: 'tokens', 'spans', 'relations'.   
- data[doc_idx].tokens key contains a list of strings, one string per word token.   
- data[doc_idx].spans key contains a list of spans, each span is an object with 4 keys: 'id', 'start', 'end', 'type'.   
- data[doc_idx].spans[span_idx].id key contains the unique span id, of the form 'Edoc_idx_N' where N is an unique integer (not span_idx).   
- data[doc_idx].spans[span_idx].start key contains the start token index of the span in data[doc_idx].tokens.   
- data[doc_idx].spans[span_idx].end key contains the end token index + 1 of the span in data[doc_idx].tokens.   
- data[doc_idx].spans[span_idx].type key contains the span type (from the set of span type names in schema.spans_types).   
- data[doc_idx].relations key contains a list of relations, each relations is an object with 4 keys: 'id', 'head', 'tail', 'type'.   
- data[doc_idx].relations[rel_idx].id key contains the unique relation id, of the form 'Rdoc_idx_N' where N is an unique integer (not rel_idx).   
- data[doc_idx].relations[rel_idx].head key contains the head span id from the spans list.   
- data[doc_idx].relations[rel_idx].tail key contains the tail span id from the spans list.    
- data[doc_idx].relations[rel_idx].type key contains the relation type (from the set of relation type names in schema.relations_types).   


schema key contains 2 keys: 'span_types', 'relation_types'.    
- schema.span_types[span_type_idx].name key contains the unique span type name.   
- schema.span_types[span_type_idx].color key contains the span type color as hex or rgba.   
- schema.relation_types[rel_type_idx].name key contains the unique relation type name.    
- schema.relation_types[rel_type_idx].color key contains the relation type color as hex or rgba.    

NOTE: for unannotated data, leave out the spans and relations keys:   

## Annotation
You can then start annotating or modifying the annotation of each document   

### Span Mode
- Click and drag the span and select the type from the menu that pops up at the mouse release point and the span will be highlighted and added to the database.   
- Right-click on an annotated span to edit the type, boundaries or delete it.

### Relation Mode (view all tails for a selected head)  
- Left click on any annotated span and the tool will go to relation mode, the clicked span will become the head span (flashing red border). All current tail spans are shown with a black border.   
- Right-click on any span to add/remove the relation with that span as tail to the flashing head span.  The option to add/remove will be determined by the current state of that tail span.   
- Press ESC to go back to span mode.

### Reverse Relation Mode (view all heads for a selected tail)
- Ctrl-Left click on any annotated span and the tool will go to reverse relation mode, the clicked span will become the tail span (flashing black border). All current head spans are shown with a red border.   
- Right-click on any span to add/remove the relation with that span as head to the flashing tail span.  The option to add/remove will be determined by the current state of that head span.   
- Press ESC to go back to span mode.


## View Results and Export
Near the top of the screen there are 2 buttons:  
- Export => The export button exports the annotations to .json in a format that you can just reload later and continue annotating.    
  NOTE: the annotated_text key is included in the export for human readability, it is just ignored on re-import.    
  NOTE: the span key is included in each span object for human readability also, it is just ignored on re-import.     
- View Results => The view results button just displays the current export file in a new tab.    