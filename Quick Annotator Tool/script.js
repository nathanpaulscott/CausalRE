///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//TO DO
// add in function to right click in span mode and delete a span with a warning
// add in function to right click in rel mode and delete a relation, this one needs a menu based on what relations there are and then the user chooses the one to remove or chooses the ALL option





///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//GLOBAL PARAMS
//data storage vars
let documents = [];
let spans = [];
let relations = [];

//counters
let counters = [];
let relationCounters = [];

//control vars
let activeRelation = { active: false, headElement: null, headIndex: null };
let tool_state = 'span_mode'; // Possible values: 'span_mode', 'relation_mode'
let mouseDownDocIndex = null;

//set the offset for the popup messages near the click point
let msg_offset_x = 40;
let msg_offset_y = -40;

//make a default schema object
schema = {
    "span_types":[
        {
            "name":"E_type1",
            "color":"rgb(135,206,250)"
        },
        {
            "name":"E_type2",
            "color":"rgb(144,238,144)"
        },
        {
            "name":"E_type3",
            "color":"rgb(255,182,193)"
        },
        {
            "name":"E_type4",
            "color":"rgb(255,165,0)"
        }
    ],
    "relation_types":[
        {
            "name":"R_type1",
            "color": "rgb(135,206,250)"
        },
        {
            "name":"R_type2",
            "color": "rgb(144,238,144)"
        },
        {
            "name":"R_type3",
            "color": "rgb(255,182,193)"
        },
        {
            "name":"R_type4",
            "color": "rgb(255,165,0)"
        }
    ]
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//Modify html document

// Inject CSS for flashing border animation
const style = document.createElement('style');
style.innerHTML = `
@keyframes flashing-border {
    0% { border-color: transparent; }
    50% { border-color: red; }
    100% { border-color: transparent; }
}

.flashing-border {
    border: 4px dashed red;
    animation: flashing-border 0.5s linear infinite; /* Flashes 2x per second */
}


.tail-border-1 { border: 2px dashed black; }
.tail-border-2 { border: 4px dashed black; }
.tail-border-3 { border: 6px dashed black; }
.tail-border-4 { border: 8px dashed black; }

#current_mode {
   color: red;
   font-weight: bold;
   font-size: 14px;
}

/* Adjust body padding to accommodate instruction div */
body { padding-top: 150px; }


div[id*="InputContainer"] {
    border: 2px solid #000000; /* Sets a black border with a thickness of 2px */
    padding: 10px; /* Adds space between the border and the content inside the div */
    margin: 10px; /* Adds space outside the border */
}

`; // Close the CSS string and statement properly

document.head.appendChild(style);




function delete_span_styles() {
    //delets all span_style from <style>...</style>
    // Find the <style> element
    let styleElement = document.querySelector('style');

    const sheet = styleElement.sheet;
    if (!sheet) return; // Skip if the stylesheet is not accessible

    // Access the rules in the stylesheet
    const rules = sheet.cssRules || sheet.rules;

    // Iterate backwards to avoid index issues when deleting
    let rulesDeleted = false;
    for (let i = rules.length - 1; i >= 0; i--) {
        const rule = rules[i];
        // Check if the rule's selector starts with "[span-type="
        if (rule.selectorText && rule.selectorText.startsWith('[span-type=')) {
            sheet.deleteRule(i); // Delete the rule
            //console.log(`Deleted rule: ${rule.selectorText}`);
            rulesDeleted = true;     
        }
    }

    // If rules were deleted, update the <style> element's content
    if (rulesDeleted) {
        // Get the updated list of rules and regenerate the <style> content
        let updatedStyles = '';
        for (let i = 0; i < sheet.cssRules.length; i++) {
            updatedStyles += `${sheet.cssRules[i].cssText}\n`;
        }
        styleElement.textContent = updatedStyles;
        //console.log('Updated <style> content after deletion.');
    }

}



//update the span styles from the schema
function update_span_styles_from_schema() {
    //clear the span_styles first
    delete_span_styles();
    
    //now add new ones from the schema
    // Find the <style> element
    let styleElement = document.querySelector('style');
    // Build CSS rules based on the schema
    let newStyles = '';
    schema['span_types'].forEach(type => {
        const type_name = type['name'];
        const type_color = type['color'];
        newStyles += `\n[span-type="${type_name}"] {background-color: ${type_color};}\n`;
    });

    // Append new styles to the style element
    styleElement.appendChild(document.createTextNode(newStyles));
}



//add the span_styles from the schema
update_span_styles_from_schema();



// Create and style instruction and buttons text at the top of the screen
const instructions = document.createElement('div');
instructions.id = 'topInstructions'; // Set the ID to 'topInstructions'
instructions.style.position = 'fixed';
instructions.style.top = '0';
instructions.style.width = '100%';
instructions.style.backgroundColor = '#f9f9f9';
instructions.style.padding = '10px';
instructions.style.textAlign = 'left';
instructions.style.fontSize = '14px';
instructions.style.fontFamily = 'Arial, sans-serif';
instructions.style.borderBottom = '1px solid #ddd';
instructions.style.zIndex = '1000';

instructions.innerHTML = `
    <strong>INSTRUCTIONS</strong><br>
    <strong>Span Mode:</strong> Click and drag to select spans of text to annotate.<br>
    Click on annotated spans to select a head span and enter Relation Mode.<br>
    <strong>Relation Mode:</strong> Click on tail spans to add that relation.<br>
    Press <strong>ESC</strong> to go back to Span Mode.<br>
    <strong>Current Mode:</strong> <span id=current_mode>span_mode</span>
`;

const topContainer = document.getElementById('topContainer');
// Append the instructions div to 'topContainer'
topContainer.appendChild(instructions);


///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//LISTENERS AND GLOBAL FUNCTIONS
//right click listener
document.getElementById('dataContainer').addEventListener('contextmenu', function(event) {
    // Prevent the default context menu from appearing
    event.preventDefault();
    
    const target = event.target;

    //close other open menus if we did not click on them
    close_other_menus(target);

    //check we clicked on a span
    if (!target || target.tagName !== "SPAN") return;
    
    //check the parent div is an editable div
    docDiv = get_parent_div_for_mouse_event(target);
    if (!docDiv) return;
    //all good so get the docDiv object and extract the docIndex
    docIndex = parseInt(docDiv.id.replace('doc', ''));
    
    // check if the user clicked on an unannotated span, do nothing if so
    if (target.getAttribute('type') === "unannotated") {
        return;
    }

    //these are the processing cases.....
    //here the clicked span type is "annotated" and we are in span_mode so remove it
    if (tool_state === 'span_mode') {
        //Delete this span from spans and relations
        remove_span_handler(event, docIndex);
    }
    
    //here the clicked span type is "annotated" and we are in relation_mode, so check if it is a tail span and remove it if so
    else {  
        const spanId = target.getAttribute('span-id');
        //process relation if the clicked span is not the head and is in the same document
        if (docIndex === activeRelation.headIndex && spanId !== activeRelation.headElement.getAttribute('span-id')) {
            remove_relation_handler(event, docIndex);
        }
        //console.log('processing candidate tail span');
    }
});




//left click listener to handle span interactions and menu management
document.getElementById('dataContainer').addEventListener('click', function(event) {
    //console.log('Left-click detected at:', event.clientX, event.clientY);
    //Ignore non-primary clicks (right-clicks, etc.)
    //if (event.button !== 0) {console.log('ignore right click');return;}
    
    const target = event.target;

    //close other open menus if we did not click on them
    close_other_menus(target);

    //check we clicked on a span
    if (!target || target.tagName !== "SPAN") return;
    
    //check the parent div is an editable div
    docDiv = get_parent_div_for_mouse_event(target);
    if (!docDiv) return;
    //all good so get the docDiv object and extract the docIndex
    docIndex = parseInt(docDiv.id.replace('doc', ''));
    
    // check if the user clicked on an unannotated span, do nothing if so
    if (target.getAttribute('type') === "unannotated") {
        return;
    }

    //these are the processing cases.....
    //here the clicked span type is "annotated" so process it
    if (tool_state === 'span_mode') {
        // Enter relation mode and set the clicked span as the head
        const clickedSpan = target;
        const spanId = clickedSpan.getAttribute('span-id');
        enterRelationMode(clickedSpan, docIndex);
        //console.log('selecting head span, going to rel mode');
    }
    
    //we are already in rel mode with a selected head span
    else {  
        const clickedSpan = target;
        const spanId = clickedSpan.getAttribute('span-id');
        //process relation if the clicked span is not the head and is in the same document
        if (docIndex === activeRelation.headIndex && spanId !== activeRelation.headElement.getAttribute('span-id')) {
            add_relation_handler(event, docIndex);
        }
        //console.log('processing candidate tail span');
    }
});




document.getElementById('dataContainer').addEventListener('mousedown', function(event) {
    let target = event.target;
    
    // Check if the tool state is span_mode
    if (tool_state !== "span_mode") {
        //console.log('mousedown only used in span_mode');
        return;
    }
    // Check if the event is happening in a span and it is of type "unannotated"
    if (target.tagName !== "SPAN" || target.getAttribute('type') !== "unannotated") {
        //console.log('did not click on a valid area to capture mousedown, must be an unannotated span');
        return;
    }
    //check if the parent div is valid
    let docDiv = get_parent_div_for_mouse_event(target);
    if (!docDiv) return;
    
    //all good so get the doc index
    mouseDownDocIndex = parseInt(docDiv.id.replace('doc', ''));
});




document.getElementById('dataContainer').addEventListener('mouseup', function(event) {
    // Capture the mouseDownDocIndex to use within the timeout
    let capturedMouseDownDocIndex = mouseDownDocIndex;
    // Reset the global mousedown index as we have an upmouse
    mouseDownDocIndex = null;
    
    let target = event.target;
    
    // Check if the tool state is span_mode
    if (tool_state !== "span_mode") {
        //console.log('mouseup only used in span_mode');
        return;
    }
    // Check if the event is happening in a span and it is of type "unannotated"
    if (target.tagName !== "SPAN" || target.getAttribute('type') !== "unannotated") {
        //console.log('did not click on a valid area to capture a mouseup, must be an unannotated span');
        return;
    }
    //check if the parent div is valid
    let docDiv = get_parent_div_for_mouse_event(target);
    if (!docDiv) return;
    
    //all good so extract the docIndex
    let docIndex = parseInt(docDiv.id.replace('doc', ''));
    
    //check if mousedown and mouseup are in the same document and if the selection represents an actual drag (non-zero length)
    if (capturedMouseDownDocIndex !== docIndex) {
        //console.log('got an invalid seleciton as mouseup was in a different doc');
        return;
    }

    //got an aceptable mouseup, so process it
    setTimeout(() => {
        const selection = window.getSelection();
        // Check if the selection range has more than zero length (implies dragging)
        if (selection.isCollapsed || selection.toString().length == 0) {
            //console.log('selected a zero span');
            return;
        }
        
        //got a click and drag so get the range and span of text
        const range = selection.getRangeAt(0);
        
        //check that the mousedown and up where in the same unannotated span tag, if not do not process as they would be overlapping spans
    if (range.startContainer !== range.endContainer) {
        //console.log('got an overlapping selection!!!!');
        return;
    } 
    
    //got to here so all good, now annotate the span
    add_span_handler(event, docIndex, range);
    }, 50);
});



//utility to close all open menus if the target is not within it
function close_other_menus(target) {
    const openMenus = document.querySelectorAll('div[style*="position: absolute"]');
    // Close any open menu if clicked outside of it
    if (openMenus.length > 0) {
        openMenus.forEach(menu => {
            if (!menu.contains(target)) {
                document.body.removeChild(menu);
            }
        });
    }
}



//utility to check we have clicked inside an acceptable div
function get_parent_div_for_mouse_event(target) {
    // Use closest to find the parent div that has id that starts with 'doc'
    let docDiv = target.closest('div[id^="doc"]');
    if (!docDiv) return null; // No matching div found

    //got to here so passed the check
    return docDiv;
}


//add the exit relation mode on esc button press event
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && tool_state === 'relation_mode') {
        exitRelationMode();
    }
});


// Utility function to remove existing menus
function removeExistingMenus() {
    const existingMenus = document.querySelectorAll('div[style*="position: absolute"]');
    existingMenus.forEach(menu => menu.parentNode.removeChild(menu));
}


///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//IMPORT/EXPORT
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const schemaInput = document.getElementById('schemaInput');
    const preannotatedInput = document.getElementById('PAfileInput');
    const topInstructions = document.getElementById('topInstructions');

    fileInput.addEventListener('change', function(event) {
        //this selects the data import .json file and loads it into the documents list and displays it on screen
        const file = event.target.files[0];
        if (!file) {
            alert("No file selected.");
            return;
        }
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                //read in the json list ot hte documents list
                documents = JSON.parse(e.target.result);
                //display the documents on screen
                display_documents("reset");
            } catch (error) {
                console.error("Error reading JSON: ", error);
                alert("Failed to load JSON file. Please ensure the file is correctly formatted.");
            }
        };
        reader.readAsText(file);
        fileInput.value = ''; 
    });
    
    schemaInput.addEventListener('change', function(event) {
        //this selects the schema import .json file and loads it into the associated js vars
        const file = event.target.files[0];
        if (!file) {
            alert("No file selected.");
            return;
        }
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                //read in the json list to the schema object
                let temp = JSON.parse(e.target.result);
                //did not get an error, so update schema
                schema = temp;
                //update the span_styles from the schema
                update_span_styles_from_schema();
            } catch (error) {
                console.error("Error reading JSON: ", error);
                alert("Failed to load JSON file. Please ensure the file is correctly formatted.");
            }
        };
        reader.readAsText(file);
        schemaInput.value = ''; 
    });
    
    preannotatedInput.addEventListener('change', function(event) {
        //this selects the preannoated import .json file and loads it into the associated js vars and modifies the display on screen
        //this file will have schema, raw text and spans and relations in it, the annotated data part is not used as that is just for human comprehension only
        const file = event.target.files[0];
        if (!file) {
            alert("No file selected.");
            return;
        }
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                //read in the json list to the schema object
                let temp = JSON.parse(e.target.result);
                //did not get an error, so update vars
                schema = temp["schema"];
                //update the span_styles from the schema
                update_span_styles_from_schema();
                //update the documents, spans and relations vars
                documents = temp["raw_data"];
                spans = temp["spans"];
                relations = temp["relations"];
                //fill out the counters
                counters = spans.map(innerList => innerList.length + 1);
                relationCounters = relations.map(innerList => innerList.length + 1);
                //display docs with annotations
                display_documents("load");
            } catch (error) {
                console.error("Error reading JSON: ", error);
                alert("Failed to load JSON file. Please ensure the file is correctly formatted.");
            }
        };
        reader.readAsText(file);
        preannotatedInput.value = ''; 
    });

    //add the export and view results buttons
    const buttonsContainer = document.createElement('div');

    const exportButton = document.createElement('button');
    exportButton.textContent = 'Export Data';
    exportButton.onclick = saveAnnotations;
    buttonsContainer.appendChild(exportButton);

    const viewResultsButton = document.createElement('button');
    viewResultsButton.textContent = 'View Results';
    viewResultsButton.onclick = viewAnnotations;
    buttonsContainer.appendChild(viewResultsButton);

    //add the topContainer to the topInstructions div
    topInstructions.appendChild(buttonsContainer);
});



function display_documents(option) {
    //this function loads in the documents to the browser and loads the annotations if otpion is "load", otherwise it resets the annotations if option is "reset"

    //this builds the div for each doc and displays it on the browser
    const container = document.getElementById('dataContainer');
    
    //clear any existing contents
    container.innerHTML = '';
    
    //create new contents from documents
    documents.forEach((text, index) => {
        //make the header element for the doc
        const header = document.createElement('h4');
        header.textContent = `doc${index}`;
        container.appendChild(header);

        //make the div to hold the document and fill it
        const docDiv = document.createElement('div');
        docDiv.id = `doc${index}`;
        docDiv.style.border = '1px solid #ccc';
        docDiv.style.padding = '10px';
        docDiv.style.lineHeight = '20px';
        docDiv.style.maxHeight = '200px';
        docDiv.style.overflowY = 'auto';
        docDiv.style.marginBottom = '20px';

        if (option === "reset") {
            //reset vars
            spans[index] = [];
            relations[index] = [];
            counters[index] = 1;
            relationCounters[index] = 1;

            //add the text tot eh docdiv
            docDiv.innerHTML = `<span type="unannotated">${text}</span>`;
        }
        else if (option ==="load") {
            //get the annotations
            const annotations = spans[index]; // Get annotations for this document
            let lastIndex = 0; // Track the last index of text processed
            let updatedInnerHTML = ''; // Build new HTML for the document
            // Sort annotations by start index to ensure proper text chunking
            annotations.sort((a, b) => a.start - b.start);
            // Iterate through each annotation and build updated inner HTML
            annotations.forEach(annotation => {
                // Add unannotated text before this annotation
                if (annotation.start > lastIndex) {
                    updatedInnerHTML += `<span type="unannotated">${text.slice(lastIndex, annotation.start)}</span>`;
                }
                // Add annotated text
                updatedInnerHTML += `<span type="annotated" span-id="${annotation.id}" span-type="${annotation.type}">${text.slice(annotation.start, annotation.end + 1)}</span>`;

                // Update lastIndex to the end of this annotation
                lastIndex = annotation.end + 1;
            });

            // Add any remaining unannotated text after the last annotation
            if (lastIndex < text.length) {
                updatedInnerHTML += `<span type="unannotated">${text.slice(lastIndex)}</span>`;
            }
            // Set the new HTML to the docDiv
            docDiv.innerHTML = updatedInnerHTML;
        }
        //add to the container
        container.appendChild(docDiv);
    });
}


function viewAnnotations() {
    //function to view the annotation data in a separate tab
    let annotated_data = [];
    documents.forEach((_, index) => {
        const docDiv = document.getElementById(`doc${index}`);
        //annotated_data.push(docDiv.innerHTML);
        converted_text = convertInnerHTML(docDiv, "view");
        annotated_data.push(converted_text);
    });

    const view_data = {
        raw_data: documents,
        annotated_data: annotated_data,
        spans: spans,
        relations: relations,
        schema: schema
    };

    const newWindow = window.open("", "_blank");
    newWindow.document.write(`<pre>${JSON.stringify(view_data, null, 4)}</pre>`);
}



function saveAnnotations() {
    //function to export the annotation data to a .json file
    let annotated_data = [];
    documents.forEach((_, index) => {
        const docDiv = document.getElementById(`doc${index}`);
        //annotated_data.push(docDiv.innerHTML);
        converted_text = convertInnerHTML(docDiv, "export");
        annotated_data.push(converted_text);
    });

    const export_data = {
        raw_data: documents,
        annotated_data: annotated_data,
        spans: spans,
        relations: relations,
        schema: schema
    };

    const jsonBlob = new Blob([JSON.stringify(export_data, null, 4)], {type: 'application/json'});
    const jsonLink = document.createElement('a');
    jsonLink.download = generateTimestampedFilename('annotated_docs', 'json');
    jsonLink.href = URL.createObjectURL(jsonBlob);
    jsonLink.click();
}


//utility function to add a timestampt to a filename
function generateTimestampedFilename(baseFilename, extension) {
    const date = new Date();
    // Create a timestamp format: YYYYMMDD-HHMMSS
    const timestamp = date.getFullYear().toString() +
                      (date.getMonth() + 1).toString().padStart(2, '0') +
                      date.getDate().toString().padStart(2, '0') + '-' +
                      date.getHours().toString().padStart(2, '0') +
                      date.getMinutes().toString().padStart(2, '0') +
                      date.getSeconds().toString().padStart(2, '0');
    // Construct the full filename with timestamp inserted before the extension
    return `${baseFilename}-${timestamp}.${extension}`;
}



//utility function to convert the innerHTML annotated text to somehting more human readbale for export
function convertInnerHTML(div, type) {
    if (!div) return ''; // Return empty if the div is not found

    let result = '';
    const children = Array.from(div.childNodes);
    children.forEach(child => {
        // Processs the span children of the doc div
        if (child.nodeType === Node.ELEMENT_NODE && child.tagName === 'SPAN') {
            if (child.getAttribute('type') === 'unannotated') {
                // Just append the text of unannotated spans
                result += child.innerText;
            } 
            else if (child.getAttribute('type') === 'annotated') {
                // Wrap annotated span text with tags from span-id
                const data_id = child.getAttribute('span-id');
                if (type === "export") 
                    result += `<${data_id}>${child.innerText}</${data_id}>`;
                else if (type === "view") 
                    result += `&lt;${data_id}&gt;${child.innerText}&lt;/${data_id}&gt;`;
            }
        }
    });
    return result;
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//SPANS
function add_span_handler(event, docIndex, range) {
    //make give the span type menu
    const menu = createSpanTypeMenu(event, docIndex, range, "add");
    document.body.appendChild(menu);
}


function remove_span_handler(event, docIndex) {
    //make the span type menu
    const menu = createSpanTypeMenu(event, docIndex, null, "rmv");
    document.body.appendChild(menu);
}



function createSpanTypeMenu(event, docIndex, range, action) {
    const menu = document.createElement('div');
    menu.style.position = 'absolute';
    menu.style.left = `${event.clientX + window.scrollX}px`;
    menu.style.top = `${event.clientY + window.scrollY}px`;
    menu.style.backgroundColor = 'white';
    menu.style.border = '1px solid black';
    menu.style.padding = '5px';
    menu.style.zIndex = '1000';

    if (action === "add") {
        schema["span_types"].forEach((type) => {
            const item = document.createElement('div');
            item.textContent = `Annotate as ${type["name"]}`;
            item.style.backgroundColor = type["color"];
            item.onclick = () => add_span(docIndex, type["name"], type["color"], range, menu);
            menu.appendChild(item);
        });
    }
    else if (action === "rmv") {
        const target_span = event.target; 
        const item = document.createElement('div');
        item.textContent = 'Remove Span?';
        item.style.color = 'red';
        item.style.fontWeight = 'bold';
        item.style.fontSize = '14px';
        //add the click listener
        item.onclick = () => remove_span(target_span, docIndex, menu);
        menu.appendChild(item);
    }
    return menu;
}


function add_span(docIndex, type, color, range, menu) {
    const spanId = `E${counters[docIndex]++}`;

    //range.startContainer is the text node, its parent is the span holding it, we will split this parent span
    //NOTE: we have already checked that .startContainer and .endContainer are the same as we are not handling overlap selections

    const originalSpan = range.startContainer.parentNode;
    const startOffset = range.startOffset;
    const endOffset = range.endOffset;

    // Split the text into before, selected, and after segments
    const fullText = originalSpan.textContent;
    const textBefore = fullText.substring(0, startOffset);
    const textAfter = fullText.substring(endOffset);
    const selectedText = range.toString();

    // Create spans for the text before and after the selected text
    const beforeSpan = document.createElement('span');
    beforeSpan.setAttribute('type', 'unannotated');
    beforeSpan.textContent = textBefore;
    const afterSpan = document.createElement('span');
    afterSpan.setAttribute('type', 'unannotated');
    afterSpan.textContent = textAfter;

    // Create a new span for the selected text
    const newSpan = document.createElement('span');
    newSpan.setAttribute('type', 'annotated');
    newSpan.setAttribute('span-id', spanId);
    newSpan.setAttribute('span-type', type);
    //newSpan.style.backgroundColor = color;    //do not need as we are using styles with the selector on attribute span-id
    newSpan.textContent = selectedText;

    // Insert new spans into the DOM, replacing the original span
    originalSpan.parentNode.insertBefore(beforeSpan, originalSpan);
    originalSpan.parentNode.insertBefore(newSpan, originalSpan);
    originalSpan.parentNode.insertBefore(afterSpan, originalSpan);
    originalSpan.parentNode.removeChild(originalSpan);

    // Calculate the absolute start index by summing lengths of all previous sibling's text
    //This is critical as without this the span start/end are not absolute for the doc
    let absoluteOffset = 0;
    let currentNode = beforeSpan.previousSibling;
    while (currentNode) {
        absoluteOffset += currentNode.textContent.length;
        currentNode = currentNode.previousSibling;
    }
    // make the absolute indices
    let absoluteStartIndex = absoluteOffset + startOffset;
    let absoluteEndIndex = absoluteOffset + endOffset - 1;        //remove one to make it consistent with python indexing

    // Update the annotations registry
    spans[docIndex].push({
        id: spanId,
        type: type,
        start: absoluteStartIndex,
        end: absoluteEndIndex,
        span: selectedText,
    });


    // Clean up the menu if it's part of the DOM
    if (menu && menu.parentNode) {
        menu.parentNode.removeChild(menu);
    }
}




function remove_span(target_span, docIndex, menu) {
    //this removes the chosen span from the spans[docIndex] list and the relations[docIndex] list
    const docDiv = document.querySelector(`#dataContainer #doc${docIndex}`);
    const span_id = target_span.getAttribute('span-id');

    //Filter out the object with the specified spanId from spans
    spans[docIndex] = spans[docIndex].filter(span => span.id !== span_id);
    
    //Filter out the object with the specified spanId from relations
    relations[docIndex] = relations[docIndex].filter(relation => relation.head !== span_id && relation.tail !== span_id);

    //remove the span from the docDiv
    //Step 1: Change the type attribute to "unannotated"
    target_span.setAttribute('type', 'unannotated');
    // Step 2: Merge all adjacent unannotated spans in the docDiv
    let i = 0;
    while (i < docDiv.children.length - 1) {
        const currentSpan = docDiv.children[i];
        const nextSpan = docDiv.children[i + 1];

        // Check if both current and next spans are "unannotated"
        if (currentSpan.getAttribute('type') === 'unannotated' && nextSpan.getAttribute('type') === 'unannotated') {
            // Merge the text content of the next span into the current span
            currentSpan.textContent += nextSpan.textContent;
            // Remove the next span from the DOM
            nextSpan.remove();
        } 
        else i++; // Move to the next pair only if no merge happened
    }

    // Clean up the menu if it's part of the DOM
    if (menu && menu.parentNode) {
        menu.parentNode.removeChild(menu);
    }
}    

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//RELATIONS
function enterRelationMode(span, docIndex) {
    //remove head span border if it is there already for some reason
    if (tool_state === 'relation_mode' && activeRelation.headElement) {
        //activeRelation.headElement.style.border = '';
        activeRelation.headElement.classList.remove('flashing-border');
    }
    
    //add the head span border
    span.classList.add('flashing-border');
    activeRelation = { active: true, headElement: span, headIndex: docIndex };
    tool_state = 'relation_mode';
    
    // Update the current mode display
    document.getElementById('current_mode').innerText = tool_state;

    // Update the tail span borders from the relations list
    update_tail_span_styles(span, docIndex, 'enter');
}


function exitRelationMode() {
    //remove head span border style
    if (activeRelation.headElement) {
        activeRelation.headElement.classList.remove('flashing-border');
    }
    
    //remove all tail span border styles
    remove_tail_span_styles();

    //reset the activeRelation object
    activeRelation = { active: false, headElement: null, headIndex: null };

    // Update the tool_state and current mode display
    tool_state = 'span_mode';
    document.getElementById('current_mode').innerText = tool_state;
}



function add_relation_handler(event, docIndex) {
    const tailId = event.target.getAttribute('span-id');
    const headId = activeRelation.headElement.getAttribute('span-id');
    showRelationTypeMenu(event, docIndex, headId, tailId, "add");
}



function remove_relation_handler(event, docIndex) {
    const tailId = event.target.getAttribute('span-id');
    const headId = activeRelation.headElement.getAttribute('span-id');
    showRelationTypeMenu(event, docIndex, headId, tailId, "rmv");
}



//check that a candidate relation has not already been added
function relationAlreadyExists(docIndex, headId, tailId, type) {
    return relations[docIndex].some(rel => rel.head === headId && rel.tail === tailId && rel.type === type);
}


function showRelationTypeMenu(event, docIndex, headId, tailId, action) {
    const menu = document.createElement('div');
    menu.style.position = 'absolute';
    menu.style.left = `${event.clientX + window.scrollX}px`;
    menu.style.top = `${event.clientY + window.scrollY}px`;
    menu.style.backgroundColor = 'white';
    menu.style.border = '1px solid black';
    menu.style.padding = '5px';
    menu.style.zIndex = '1001';

    // Filter relation types based on action
    let relationTypes = schema["relation_types"];
    if (action === "rmv") {
        //filter the relation_types to only those that exist for this head-tail pair as only these can be removed
        const existingRelations = relations[docIndex].filter(
            relation => relation.head === headId && relation.tail === tailId
        );
        relationTypes = relationTypes.filter(type =>
            existingRelations.some(relation => relation.type === type.name)
        );
    }

    // Create menu items for filtered relation types
    relationTypes.forEach((type) => {
        const relItem = document.createElement('div');
        relItem.textContent = action === "add" ? `Relate as ${type.name}` : `Remove relation: ${type.name}`;
        relItem.style.backgroundColor = type.color;

        relItem.onclick = () => {
            if (action === "add") {
                if (!relationAlreadyExists(docIndex, headId, tailId, type.name)) {
                    add_relation(docIndex, headId, tailId, type.name, type.color);
                } else {
                    show_relation_already_exists_msg(event.clientX + msg_offset_x, event.clientY + msg_offset_y);
                }
            } 
            else if (action === "rmv") {
                remove_relation(docIndex, headId, tailId, type.name);
            }
            document.body.removeChild(menu); // Close menu after selection
        };
        menu.appendChild(relItem);
    });
    document.body.appendChild(menu);
}




function show_relation_already_exists_msg(x,y) {
    //show dissappearing popup msg that this relation has already been added
    let messageBox = document.getElementById('NoAddRelMsg');
    messageBox.style.position = 'fixed';
    messageBox.style.display = 'block';
    messageBox.style.left = `${x}px`; // Position near the click horizontally
    messageBox.style.top = `${y}px`; // Position near the click vertically
    messageBox.style.backgroundColor = 'white';
    messageBox.style.border = '2px dashed red';
    messageBox.style.padding = '5px';
    messageBox.style.zIndex = '1500';                
    messageBox.style.color = 'red';
    messageBox.style.fontWeight = 'bold';
    messageBox.style.fontSize = '14px';

    setTimeout(function() {
        messageBox.style.display = 'none';
    }, 2000);  //hide after this many ms
}


function add_relation(docIndex, head_id, tail_id, type, color) {
    const rel_id = `R${relationCounters[docIndex]++}`;

    //doesn't use the color for now, may change this later
    relations[docIndex].push({
        id: rel_id,
        type: type,
        head: head_id,
        tail: tail_id
    });

    //update the border for the new relation
    updateBorderStyle(docIndex, tail_id, +1)
}



function remove_relation(docIndex, head_id, tail_id, rel_type) {
    //confirm with user first
    //const confirmAction = confirm(`Confirm to remove relation ${head_id +'_' + tail_id + '_' + rel_type} in doc ${docIndex}`);
    //if (!confirmAction) {
    //    // User clicked "Cancel"
    //    event.preventDefault();
    //    return;
    //}

    //this removes the given head-tail pair and type (relation) from the relations[docIndex] list
    const docDiv = document.querySelector(`#dataContainer #doc${docIndex}`);

    //Filter out the object with the specified tailId from relations
    relations[docIndex] = relations[docIndex].filter(relation => !(relation.head === head_id && relation.tail === tail_id && relation.type === rel_type));

    //update the border for the new relation
    updateBorderStyle(docIndex, tail_id, -1)
}    



//function to remove tail span styling on rel mode exit
function remove_tail_span_styles() {
    let elements = document.querySelectorAll('[class^="tail-border-"]');

    // Loop through the NodeList and log each element
    elements.forEach(function(element) {
         element.classList.remove('tail-border-1', 'tail-border-2', 'tail-border-3', 'tail-border-4');
    });
}



//function to update all tail span styles on entering relation mode
function update_tail_span_styles(span, docIndex) {
    const headId = span.getAttribute('span-id');
    const relationCounts = {};
    
    // Collect all tail IDs for the given head ID
    relations[docIndex].forEach(relation => {
        if (relation.head === headId) {
            if (relationCounts[relation.tail]) {
                relationCounts[relation.tail]++;
            } else {
                relationCounts[relation.tail] = 1;
            }
        }
    });

    // Update the style for each tail span
    Object.keys(relationCounts).forEach(tailId => {
        const tailSpans = document.querySelectorAll(`#doc${docIndex} [span-id='${tailId}']`);
        tailSpans.forEach(tailSpan => {
            level = Math.min(4, relationCounts[tailId]);
            tailSpan.classList.remove('tail-border-1', 'tail-border-2', 'tail-border-3', 'tail-border-4');
            tailSpan.classList.add(`tail-border-${level}`);
        });
    });

}


function updateBorderStyle(docIndex, tail_id, delta) {
    //update the border for the new relation
    const tail_spans = document.querySelectorAll(`#doc${docIndex} [span-id='${tail_id}']`);
    tail_spans.forEach(tail_span => {
        let class_list = tail_span.classList;
        // Find the first class that starts with 'tail-border-'
        let level = 0;
        let target_class = Array.from(class_list).find(cls => cls.startsWith('tail-border-'));
        if (target_class) {
             // Extract the part of the class string after 'tail-border-'
             level = target_class.substring('tail-border-'.length);
             level = parseInt(level,10);
        }
        level += delta;
        level = Math.min(4, level);

        tail_span.classList.remove('tail-border-1', 'tail-border-2', 'tail-border-3', 'tail-border-4');
        tail_span.classList.add(`tail-border-${level}`);
    });
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
