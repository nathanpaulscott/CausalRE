# Quick Annotation Tool
This is an html/css/js based single page app for quick annotation.  The focus is on efficiency of the annotation process, which is a repetitive human task.  Every mouse movement, mouse click, keyboard stroke and delay matters.  All data is stored in-memory and imported/exported in .json format.

## Imports
The tool is simple, you can load in a set of unannotated or pre-annotated documents, samples are given.  

The unannotated file file should be a .json object with 2 keys (raw_docs, schema). The raw docs are a list of the documents to annotate, the schema is an object that defines the allowed span and relation types.

The pre-annotated file is the same as the unannotated case with additional keys for spans and relations.

NOTE: The tool currently does not support overlapping spans.

## Annotation
You can then start annotating or modifying the annotation of each document

### Span Mode
Click and drag the span and select the type from the menu that pops up at the mouse release point and the span will be highlighted and added to the database. Right-click on an annotated span and you can delete it from the screen and database.  Left-click on the span and modify it's type.

### Relation Mode (view all tails for a selected head)
Ctrl-left click on any annotated span and the tool will go to relation mode, the clicked span will become the head span and will have a flashing red border. To add a candidate span as tail, you left click on any of the other annotated spans in that document and the tool will give you a menu to select the relation type to add, the black border around the tail spans, indicates if that span is a tail span and the thickness indicates how many types it has as relations. To remove an existing relation, you right click on an annotated span that has a black border in relation mode, the tool will give you a menu at the click point to select which relation type to remove.  Press ESC to go back to span mode.

### Reverse Relation Mode (view all heads for a selected tail)
Shift-left click on any annotated span and the tool will go to reverse relation mode, the clicked span will become the tail span and will have a flashing black border. To add a candidate span as head, you left click on any of the other annotated spans in that document and the tool will give you a menu to select the relation type to add, the red border around the head spans, indicates if that span is a head span and the thickness indicates how many types it has as relations. To remove an existing relation, you right click on an annotated span that has a red border in reverse relation mode, the tool will give you a menu at the click point to select which relation type to remove.  Press ESC to go back to span mode.


## View Results and Export
Near the top of the screen there are 2 buttons:
- Export => The export button exports the annotations to .json in a format that you can just reload later and continue annotating. NOTE: the annotated_docs key is included in the export for human readability, it is just ignored on re-import.
- View Results => The view results button just displays the current export file in a new tab.