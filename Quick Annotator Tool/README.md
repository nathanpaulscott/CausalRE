# Quick Annotation Tool

## Background
I have a wealth of experience from an extensive engineering career developing simple apps in the domain of repetitive human UX.  Having been a user and developer gives me a unique insight into the nuance of designing a UX with maximum efficiency so that it keeps out of the way of the human user (every delay matters, every mouse movement matters, every mouse click matters, every keystroke matters).  This nuance is typically the undoing of even the most accomplished software engineers, leading to almost invariably poor usability software, even from well regarded professional publishers.  This is problematic if the software is to be used for repetitive human tasks.  In this case the task is annotation which is very repetitive.

This tool was written in a few days, so it was not cleaned up, I will do that at some point, but it works and it's super useful.

## Imports
The tool is simple, you can load in a set of unannotated documents a sample is given, it is a simple json list of text strings.  In this case you would also want to load in a schema which defines the allowed span types and span-pair (relation) types.  You can see the sample schema for the .json format.

If you have previously annotated data, you can just choose to import that, the sample file gives the required format.  This .json will include the schema.  This is the format that the tool exports.  NOTE: the "annotated_data" part is not required for import here, it is only for human consumption.

## Annotation
You can then start annotating each document:
### Span Mode
Just click and drag the span and selecct the type from teh menu that opos up at the mouse release point and the span will be highlighted and added to the database.  Right-click on an annotated span and you can delete it from the screen and database.
### Relation Mode
Just left click on any annotated span and the tool will go to relation mode, the clicked span will become the head span in the relation and will have a flashing red border.  To add a relation (head, tail, type), you left click on any of the other annotated spans in that document and the tool will give you a menu to select the relation type to add, the black border aroudn the tail spans, indicates if that span is a tail span and the thickness indicates how many types it has as relations.  To remove an existing relation, you right click on an annotated span that has a black border (thickness indicating how many types it has relations for) in relation mode, the tool will give you a menu at the click point to select which relation type to remove. 

## View Results and Export
Near the top of the screen there are 2 buttons: Export and View Results.  The export button exports the annotations to .json in a format that you can just reload later and continue annotating.  The view results button just displays the current export file in a new tab.




