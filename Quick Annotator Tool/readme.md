# Quick Annotation Tool
This is an html/css/js based single page app for quick annotation.  The focus is on efficiency of the annotation process, which is a repetitive human task.  Every mouse movement, mouse click, keyboard stroke and delay matters.  All data is stored in-memory and imported/exported in .json format.

## Imports
The tool is simple, you can load in a set of unannotated or pre-annotated documents, samples are given.  

The json format is as follows, for unannotated data, leave out the spans and relations keys:
{
    'data': [
        {
            'tokens': [list of strings (word tokens)],
            'spans': [
                {
                    'id': the span id,
                    'start': the start token idx,
                    'end': the end token idx + 1,
                    'type': the span type
                },...
            ],
            'relations': [
                {
                    'id': the relation id,
                    'head': the head span id,
                    'tail': the tail span id,
                    'type': the span type
                },...
            ]
        },
        {next obs},
        {next obs},...
    ],
    'schema': {
    	'span_types':[
        	{
	            "name": the span type name,
        	    "color":"rgba(135,206,250, 0.3)"
        	},...
    	],
    	'relation_types':[
        	{
	            "name": the relation type name",
        	    "color": "rgba(235,100,200, 0.3)"
	        },...
	]
}

## Annotation
You can then start annotating or modifying the annotation of each document

### Span Mode
Click and drag the span and select the type from the menu that pops up at the mouse release point and the span will be highlighted and added to the database. Right-click on an annotated span to edit the type, boundaries or delete it.

### Relation Mode (view all tails for a selected head)
Left click on any annotated span and the tool will go to relation mode, the clicked span will become the head span (flashing red border). All current tail spans are shown with a black border.  Right-click on any span to add/remove the relation with that span as tail to the flashing head span.  The option to add/remove will be determined by the current state of that tail span. Press ESC to go back to span mode.

### Reverse Relation Mode (view all heads for a selected tail)
Ctrl-Left click on any annotated span and the tool will go to reverse relation mode, the clicked span will become the tail span (flashing black border). All current head spans are shown with a red border.  Right-click on any span to add/remove the relation with that span as head to the flashing tail span.  The option to add/remove will be determined by the current state of that head span. Press ESC to go back to span mode.


## View Results and Export
Near the top of the screen there are 2 buttons:
- Export => The export button exports the annotations to .json in a format that you can just reload later and continue annotating.   
  NOTE: the annotated_text key is included in the export for human readability, it is just ignored on re-import.
  NOTE: the span key is included in each span object for human readability also, it is just ignored on re-import.  
- View Results => The view results button just displays the current export file in a new tab.