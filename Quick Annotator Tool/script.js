///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//TO DO



///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//GLOBAL PARAMS
//data storage vars
let source_data = {};   //holds the dict of source data, each key is a source id and the value is the source text
let sources = [];       //holds the source id for each doc
let offsets = [];       //holds the offset for the doc in the source given by sources[docIndex]
let raw_docs = [];      //holds the unannotated doc
let spans = [];
let relations = [];

//control vars
let active_span = {
    Element:    null, 
    Index:      null
};
let tool_state = 'span_mode'; // Possible values: 'span_mode', 'relation_mode'
let mouseDownDocIndex = null;
let input_format = 'min';
let instructions = add_instructions();

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
@keyframes red-flashing-border {
    0% { border-color: transparent; }
    50% { border-color: red; }
    100% { border-color: transparent; }
}
.red-flashing-border {
    border: 4px dashed red;
    animation: red-flashing-border 0.5s linear infinite; /* Flashes 2x per second */
}

@keyframes black-flashing-border {
    0% { border-color: transparent; }
    50% { border-color: black; }
    100% { border-color: transparent; }
}
.black-flashing-border {
    border: 4px dashed black;
    animation: black-flashing-border 0.5s linear infinite; /* Flashes 2x per second */
}

.tail-border-1 { border: 3px dashed black; }
.tail-border-2 { border: 5px dashed black; }
.tail-border-3 { border: 7px dashed black; }
.tail-border-4 { border: 9px dashed black; }

.head-border-1 { border: 3px dashed red; }
.head-border-2 { border: 5px dashed red; }
.head-border-3 { border: 7px dashed red; }
.head-border-4 { border: 9px dashed red; }

#current_mode {
   color: red;
   font-weight: bold;
   font-size: 14px;
}

/* Adjust body padding to accommodate instruction div */
body { padding-top: 30px; }


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


function add_instructions() {
    const instructions = document.createElement('div');
    instructions.id = 'topInstructions';
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
    instructions.style.maxHeight = '30px'; // Initially collapsed
    instructions.style.overflow = 'hidden';
    instructions.style.cursor = 'pointer'; // Indicate hover effect
    instructions.style.transition = 'max-height 0.3s ease-in-out';

    instructions.innerHTML = `
        <div id="instructions-header"><strong>INSTRUCTIONS</strong></div>
        <div id="instructions-content" style="display: none;">
            <br>
            <strong>Span Mode:</strong><br>
            - <strong>Click and drag</strong> to select spans of text to annotate.<br>
            - <strong>RightClick</strong> on any annotated span to remove that span.<br>
            <strong>Relation Mode:</strong><br>
            - <strong>ctrl-click</strong> on any span to move to relation mode (selected span as head) and see the selected span's tail spans.<br>
            - <strong>LeftClick</strong> on any span to add the relation with it as tail.<br>
            - <strong>RightClick</strong> on any highlighted tail span to remove the relation to it.<br>
            <strong>Reverse Relation Mode:</strong><br>
            - <strong>shift-click</strong> on any span to move to reverse relation mode (selected span as tail) and see the selected span's head spans.<br>
            - <strong>LeftClick</strong> on any span to add the relation with it as head.<br>
            - <strong>RightClick</strong> on any highlighted head span to remove the relation to it.<br>
            <strong>Go Back to Span Mode:</strong><br>
            - Press <strong>ESC</strong> or <strong>shift-click</strong> or <strong>ctrl-click</strong> on a non-span to go back to Span Mode.<br>
        </div>
    `;

    const topContainer = document.getElementById('topContainer');
    // Append the instructions div to 'topContainer'
    topContainer.appendChild(instructions);
    
    return instructions
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


function add_export_and_view_results_buttons() {
    //add the export and view results buttons
    const buttonsContainer = document.createElement('div');

    const statediv = document.createElement('div');
    statediv.style.padding = '10px';
    statediv.style.textAlign = 'left';
    statediv.style.fontSize = '14px';
    statediv.style.fontFamily = 'Arial, sans-serif';
    statediv.style.zIndex = '1000';
    statediv.innerHTML = '<strong>Current Mode:</strong> <span id="current_mode">span_mode</span>';
    buttonsContainer.appendChild(statediv);

    const exportButton = document.createElement('button');
    exportButton.textContent = 'Export Data';
    exportButton.onclick = function() {export_data("export");};
    exportButton.style.marginLeft = '10px';
    buttonsContainer.appendChild(exportButton);

    const viewResultsButton = document.createElement('button');
    viewResultsButton.textContent = 'View Results';
    viewResultsButton.onclick = function() {export_data("view");};
    viewResultsButton.style.marginLeft = '10px';
    buttonsContainer.appendChild(viewResultsButton);

    //add the topContainer to the topInstructions div
    topContainer.appendChild(buttonsContainer);
}

function add_tooltip_info() {
    // Create a tooltip element and add it to the document
    const tooltip = document.createElement('div');
    tooltip.id = 'tooltip_info';
    tooltip.style.position = 'absolute';
    tooltip.style.backgroundColor = '#333';
    tooltip.style.color = '#fff';
    tooltip.style.padding = '5px';
    tooltip.style.borderRadius = '5px';
    tooltip.style.fontSize = '12px';
    tooltip.style.display = 'none';
    tooltip.style.zIndex = '1000';

    //const container = document.getElementById('RawInputContainer');
    //container.insertAdjacentElement('beforestart', tooltip);
    const container = document.getElementById('RawInputContainer');
    container.parentNode.insertBefore(tooltip, container);
    
    return tooltip
}


function add_tooltip_caution() {
    // Create a tooltip element and add it to the document
    const tooltip = document.createElement('div');
    tooltip.id = 'tooltip_caution';
    tooltip.style.position = 'absolute';
    tooltip.style.backgroundColor = 'white'; // White text
    tooltip.style.color = 'red'; // Red text
    tooltip.style.padding = '5px';
    tooltip.style.borderRadius = '5px';
    tooltip.style.display = 'none';
    tooltip.style.zIndex = '1500';
    tooltip.style.fontWeight = 'bold';
    tooltip.style.fontSize = '14px';

    const container = document.getElementById('RawInputContainer');
    container.parentNode.insertBefore(tooltip, container);

    return tooltip
}


//add the span_styles from the schema
update_span_styles_from_schema();
//add the export and view results buttons
add_export_and_view_results_buttons();
//add the tooltip info div
const tooltip_info = add_tooltip_info();
//add the no add relation tooltip
const tooltip_caution = add_tooltip_caution();


///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//LISTENERS AND GLOBAL FUNCTIONS
// Add hover behavior to show/hide instructions
document.getElementById('topInstructions').addEventListener('mouseenter', () => {
    instructions.style.maxHeight = '300px';
    document.getElementById('instructions-content').style.display = 'block';
});

document.getElementById('topInstructions').addEventListener('mouseleave', () => {
    instructions.style.maxHeight = '30px';
    document.getElementById('instructions-content').style.display = 'none';
});


//right click listener
document.getElementById('dataContainer').addEventListener('contextmenu', function(event) {
    // Prevent the default context menu from appearing
    event.preventDefault();
    
    //get target
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
    if (target.getAttribute('type') === "unannotated") return;

    //these are the processing cases.....
    //here the clicked span type is "annotated" and we are in span_mode so remove it
    if (tool_state === 'span_mode') 
        edit_span_handler(event, docIndex, null, "rmv");
    
    //here the clicked span type is "annotated" and we are in relation_mode, so check if it is a tail span and remove it if so
    else {  
        const spanId = target.getAttribute('span-id');
        //process relation if the clicked span is not the head and is in the same document
        if (docIndex === active_span.Index && spanId !== active_span.Element.getAttribute('span-id'))
            edit_relation_handler(event, docIndex, 'rmv');
    }
});




//left click listener to handle span interactions and menu management
document.getElementById('dataContainer').addEventListener('click', function(event) {
    // Disable default Ctrl+Click and Shift+Click behaviors
    if (event.ctrlKey || event.shiftKey) {
        event.preventDefault(); // Prevent the default browser action
        event.stopPropagation(); // Stop the event from propagating further
    }
    
    //Determine the click type
    let ctrl_left_click = event.button === 0 && event.ctrlKey && !event.shiftKey;
    let shift_left_click = event.button === 0 && event.shiftKey && !event.ctrlKey;
    let left_click = event.button === 0 && !(event.ctrlKey || event.shiftKey);
    let ctrl_shift_left_click = event.button === 0 && event.ctrlKey && event.shiftKey;

    // Ignore Ctrl+Shift+Click, treat it as a normal left-click
    if (ctrl_shift_left_click) {
       ctrl_left_click = false;
       shift_left_click = false;
    }   

    //console.log('Left-click detected at:', event.clientX, event.clientY);
    const target = event.target;
    //close other open menus if we did not click on them
    close_other_menus(target);
    
    //check we clicked on a span tag and it was not span that had type as unannotated, exit if so
    //if (!target || target.tagName !== "SPAN" || target.getAttribute('type')  === "unannotated") return;
    if (!target || target.tagName !== "SPAN") return;
    //check the parent div is a docDiv, if not exit
    let docDiv = get_parent_div_for_mouse_event(target);
    if (!docDiv) return;
    //all good so extract the docIndex
    let docIndex = parseInt(docDiv.id.replace('doc', ''));
    
    //these are the processing cases.....
    if ((ctrl_left_click || shift_left_click) && target.getAttribute('type') === "unannotated") 
        exit_relation_mode();
    //ctrl left click, so enter relation mode from whatever state we are in
    else if (ctrl_left_click) 
        enter_relation_mode(target, docIndex, 'relation_mode');
    //shift left click, so enter reverse relation mode from whatever state we are in
    else if (shift_left_click) 
        enter_relation_mode(target, docIndex, 'rev_relation_mode');
    //if plain left click do nothing in span mode, add relation in any of the relation modes
    else if (left_click) {
        //if in span_mode enter relation mode on plain left click
        if (tool_state === 'span_mode') return;
        //if already in any of the relation modes, process add the selected span as a relation to add
        else if (tool_state !== 'span_mode' && target.getAttribute('type') !== "unannotated") {
            const spanId = target.getAttribute('span-id');
            //check that the clicked span was not the same as the active_span span and is in the same docDiv, if not exit
            if (docIndex !== active_span.Index || spanId === active_span.Element.getAttribute('span-id')) return;
            //all good, we clicked on an acceptable candidate span in the same docDiv, so process the add relation
            edit_relation_handler(event, docIndex, 'add');
            //console.log('processing candidate head/tail span');
        }
    }
});



//add the exit all relation modes on esc button press event
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape' && (tool_state === 'relation_mode' || tool_state === 'rev_relation_mode')) 
        exit_relation_mode();
});



document.getElementById('dataContainer').addEventListener('mousedown', function(event) {
    // Disable default Ctrl+Click and Shift+Click behaviors
    if (event.ctrlKey || event.shiftKey) {
        event.preventDefault(); // Prevent the default browser action
        event.stopPropagation(); // Stop the event from propagating further
    }

    let target = event.target;
    
    // Check if the tool state is span_mode
    if (tool_state !== "span_mode") return;
    // Check if the event is happening in a span and it is of type "unannotated"
    if (target.tagName !== "SPAN" || target.getAttribute('type') !== "unannotated") return;
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
    if (tool_state !== "span_mode") return;
    // Check if the event is happening in a span and it is of type "unannotated"
    if (target.tagName !== "SPAN" || target.getAttribute('type') !== "unannotated") return;
    //check if the parent div is valid
    let docDiv = get_parent_div_for_mouse_event(target);
    if (!docDiv) return;
    
    //all good so extract the docIndex
    let docIndex = parseInt(docDiv.id.replace('doc', ''));
    
    //check if mousedown and mouseup are in the same document and if the selection represents an actual drag (non-zero length)
    if (capturedMouseDownDocIndex !== docIndex) return;

    //got an aceptable mouseup, so process it
    setTimeout(() => {
        const selection = window.getSelection();
        // Check if the selection range has more than zero length (implies dragging)
        if (selection.isCollapsed || selection.toString().length == 0) return;
        
        //got a click and drag so get the range and span of text
        const range = selection.getRangeAt(0);
        
        //check that the mousedown and up where in the same unannotated span tag, if not do not process as they would be overlapping spans
        if (range.startContainer !== range.endContainer) return;
        
        //got to here so all good, now annotate the span
        edit_span_handler(event, docIndex, range, "add");
    }, 50);    //50ms timeout
});



document.getElementById('dataContainer').addEventListener('mouseover', function(event) {
    const target = event.target;

    // Check if the hovered element is an annotated span
    if (target.tagName === 'SPAN' && target.getAttribute('type') === 'annotated') {
        const spanId = target.getAttribute('span-id');
        // Display the tooltip with the content
        tooltip_info.textContent = spanId;
        tooltip_info.style.display = 'block';
        // Position the tooltip near the mouse cursor
        tooltip_info.style.left = `${event.clientX + window.scrollX - msg_offset_x}px`;
        tooltip_info.style.top = `${event.clientY + window.scrollY + msg_offset_y}px`;
    }
});


document.getElementById('dataContainer').addEventListener('mousemove', function(event) {
    // Move the tooltip along with the mouse
    if (tooltip_info.style.display === 'block') {
        tooltip_info.style.left = `${event.clientX + window.scrollX - msg_offset_x}px`;
        tooltip_info.style.top = `${event.clientY + window.scrollY + msg_offset_y}px`;
    }
});


document.getElementById('dataContainer').addEventListener('mouseout', function(event) {
    const target = event.target;
    // Hide the tooltip when the mouse leaves an annotated span
    if (target.tagName === 'SPAN' && target.getAttribute('type') === 'annotated') {
        tooltip_info.style.display = 'none';
    }
});



//utility to close all open menus if the target is not within it
function close_other_menus(target) {
    //const openMenus = document.querySelectorAll('div[style*="position: absolute"]');
    const existingMenus = document.querySelectorAll('div[id="menu"]');
    // Close any open menu if clicked outside of it
    if (existingMenus.length > 0) {
        existingMenus.forEach(menu => {
            if (!menu.contains(target)) {
                document.body.removeChild(menu);
            }
        });
    }
}


// Utility function to remove existing menus
function removeExistingMenus() {
    const existingMenus = document.querySelectorAll('div[id="menu"]');
    existingMenus.forEach(menu => menu.parentNode.removeChild(menu));
}




//utility to check we have clicked inside an acceptable div
function get_parent_div_for_mouse_event(target) {
    // Use closest to find the parent div that has id that starts with 'doc'
    let docDiv = target.closest('div[id^="doc"]');
    if (!docDiv) return null; // No matching div found

    //got to here so passed the check
    return docDiv;
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//IMPORT/EXPORT
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const preannotatedInput = document.getElementById('PAfileInput');
    const topInstructions = document.getElementById('topInstructions');

    fileInput.addEventListener('change', function(event) {
        //this selects the data import .json file and loads it into the raw_docs list and displays it on screen
        const file = event.target.files[0];
        if (!file) {
            alert("No file selected.");
            return;
        }
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                let temp = JSON.parse(e.target.result);

                if (input_format === 'min') {
                    raw_docs = temp.raw_docs;
                }
                else if (format === 'full') {
                    //This is for the more complex input case, not used for now
                    raw_docs = temp.raw_docs.map(x => x.raw_doc);
                    sources = temp.raw_docs.map(x => x.source);
                    offsets = temp.raw_docs.map(x => x.offset);
                    source_data = temp.source_data;
                }

                schema = temp.schema;
                //update the span_styles from the schema
                update_span_styles_from_schema();

                //display the raw_docs on screen
                display_documents("reset");
            } catch (error) {
                console.error("Error reading JSON: ", error);
                alert("Failed to load JSON file. Please ensure the file is correctly formatted.");
            }
        };
        reader.readAsText(file);
        fileInput.value = ''; 
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

                if (input_format === 'min') {
                    raw_docs = temp.raw_docs;
                }
                else if (format === 'full') {
                    //This is for the more complex input case, not used for now
                    raw_docs = temp.raw_docs.map(x => x.raw_doc);
                    sources = temp.raw_docs.map(x => x.source);
                    offsets = temp.raw_docs.map(x => x.offset);
                    source_data = temp.source_data;
                }

                spans = temp.spans;
                relations = temp.relations;
                schema = temp.schema;
                //update the span_styles from the schema
                update_span_styles_from_schema();
    
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
});


function display_documents(option) {
    //this function loads in the raw_docs to the browser and loads the annotations if otpion is "load", otherwise it resets the annotations if option is "reset"

    //this builds the div for each doc and displays it on the browser
    const container = document.getElementById('dataContainer');
    
    //clear any existing contents
    container.innerHTML = '';
    
    //create new contents from raw_docs
    raw_docs.forEach((text, index) => {
        //make the header element for the doc
        const header = document.createElement('div');
        header.textContent = `id: doc${index}`;
        header.style.fontWeight = 'bold';
        header.style.fontSize = '12px';
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

            //add the text to the docdiv
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


function export_data(option) {
    //option is either "view" or "export"
    //function to view the annotation data in a separate tab
    let annotated_docs = [];
    raw_docs.forEach((_, index) => {
        const docDiv = document.getElementById(`doc${index}`);
        //annotated_data.push(docDiv.innerHTML);
        converted_text = convertInnerHTML(docDiv, option);
        annotated_docs.push(converted_text);
    });

    const out_data = {
        raw_docs: raw_docs,
        annotated_docs: annotated_docs,
        spans: spans,
        relations: relations,
        schema: schema
    };

    if (option === "view") {
        const newWindow = window.open("", "_blank");
        newWindow.document.write(`<pre>${JSON.stringify(out_data, null, 4)}</pre>`);
    }
    else if (option === "export") {
        const jsonBlob = new Blob([JSON.stringify(out_data, null, 4)], {type: 'application/json'});
        const jsonLink = document.createElement('a');
        jsonLink.download = generateTimestampedFilename('annotated_docs', 'json');
        jsonLink.href = URL.createObjectURL(jsonBlob);
        jsonLink.click();
    }
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
function edit_span_handler(event, docIndex, range, action) {
    //make give the span type menu
    show_span_menu(event, docIndex, range, action);
}


function show_span_menu(event, docIndex, range, action) {
    const menu = document.createElement('div');
    menu.id = 'menu';
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
            item.onclick = () => {
                add_span(docIndex, type["name"], type["color"], range);
                document.body.removeChild(menu); // Close menu after selection
            }
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
        item.onclick = () => {
            remove_span(target_span, docIndex);
            document.body.removeChild(menu); // Close menu after selection
        }
        menu.appendChild(item);
    }
    document.body.appendChild(menu);
}


function get_next_span_id(docIndex) {
    //find the highest id number and add 1
    let max_id = 0;
    spans[docIndex].forEach(span => {
        const match = span.id.match(/^E\d+_(\d+)$/);
        if (match) {
            // Extract the value of x as an integer
            const id = parseInt(match[1], 10);
            if (!isNaN(id) && id > max_id)
                max_id = id;
        }
    });

    // Generate the next available ID by incrementing the highest found x
    return `E${docIndex}_${max_id + 1}`;
}



function add_span(docIndex, type, color, range) {
    const spanId = get_next_span_id(docIndex);

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
}


function remove_span(target_span, docIndex) {
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
}    

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//RELATIONS
function enter_relation_mode(span, docIndex, mode) {
    //remove all span styles
    remove_span_styles();
    //set the active_span
    active_span = {Element: span, Index: docIndex};
    //set the tool state
    tool_state = mode;
    // Update the current mode display
    document.getElementById('current_mode').innerText = tool_state;
    // Update all the span border styles, will use tool_state to determine exact action
    update_span_styles_on_entry(span, docIndex);
}


function exit_relation_mode() {
    //remove all span styles on exit
    remove_span_styles();
    //reset the active_span object
    active_span = {Element: null, Index: null};
    // Update the tool_state and current mode display
    tool_state = 'span_mode';
    document.getElementById('current_mode').innerText = tool_state;
}


function edit_relation_handler(event, docIndex, action) {
    let tailId = event.target.getAttribute('span-id');
    let headId = active_span.Element.getAttribute('span-id');
    if (tool_state === 'rev_relation_mode') [headId, tailId] = [tailId, headId];   //swap head and tail
    show_relation_menu(event, docIndex, headId, tailId, action);
}


function show_relation_menu(event, docIndex, headId, tailId, action) {
    const menu = document.createElement('div');
    menu.id = 'menu';
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
    if (relationTypes.length === 0) return;

    relationTypes.forEach((type) => {
        const relItem = document.createElement('div');
        if (action === 'add')       relItem.textContent = `Add relation: ${type.name}`;
        else if (action === 'rmv')  relItem.textContent = `Rmv relation: ${type.name}`;
        relItem.style.backgroundColor = type.color;

        relItem.onclick = () => {
            if (action === "add") {
                if (!relation_already_exists(docIndex, headId, tailId, type.name)) {
                    add_relation(docIndex, headId, tailId, type.name, type.color);
                } else {
                    show_relation_already_exists_msg(event.clientX + window.scrollX + msg_offset_x, event.clientY + window.scrollY + msg_offset_y);
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



//check that a candidate relation has not already been added
function relation_already_exists(docIndex, headId, tailId, type) {
    return relations[docIndex].some(rel => rel.head === headId && rel.tail === tailId && rel.type === type);
}



function show_relation_already_exists_msg_old(x,y) {
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

function show_relation_already_exists_msg(x,y) {
    //show dissappearing popup msg that this relation has already been added
    //tooltip_caution.style.position = 'fixed';
    tooltip_caution.style.display = 'block';
    tooltip_caution.style.left = `${x}px`; // Position near the click horizontally
    tooltip_caution.style.top = `${y}px`; // Position near the click vertically
    //tooltip_caution.style.backgroundColor = 'white';
    //tooltip_caution.style.border = '2px dashed red';
    //tooltip_caution.style.padding = '5px';
    //tooltip_caution.style.zIndex = '1500';                
    //tooltip_caution.style.color = 'red';
    //tooltip_caution.style.fontWeight = 'bold';
    //tooltip_caution.style.fontSize = '14px';
    tooltip_caution.textContent = 'Relation Already Exists'

    setTimeout(function() {
        tooltip_caution.style.display = 'none';
    }, 2000);  //hide after this many ms
}



function get_next_rel_id(docIndex) {
    //find the highest id number and add 1
    let max_id = 0;
    relations[docIndex].forEach(rel => {
        const match = rel.id.match(/^R\d+_(\d+)$/);
        if (match) {
            // Extract the value of x as an integer
            const id = parseInt(match[1], 10);
            if (!isNaN(id) && id > max_id)
                max_id = id;
        }
    });

    // Generate the next available ID by incrementing the highest found x
    return `R${docIndex}_${max_id + 1}`;
}



function add_relation(docIndex, head_id, tail_id, type, color) {
    const rel_id = get_next_rel_id(docIndex);

    //doesn't use the color for now, may change this later
    relations[docIndex].push({
        id: rel_id,
        head: head_id,
        tail: tail_id,
        type: type
    });

    //update the border for the new relation
    if (tool_state === 'relation_mode')          update_border_style(docIndex, tail_id, +1)
    else if (tool_state === 'rev_relation_mode') update_border_style(docIndex, head_id, +1)
}



function remove_relation(docIndex, head_id, tail_id, type) {
    //this removes the given head-tail pair and type (relation) from the relations[docIndex] list
    
    //Remove the relation from relations
    relations[docIndex] = relations[docIndex].filter(relation => !(relation.head === head_id && relation.tail === tail_id && relation.type === type));

    //update the border for the removed relation
    if (tool_state === 'relation_mode')          update_border_style(docIndex, tail_id, -1)
    else if (tool_state === 'rev_relation_mode') update_border_style(docIndex, head_id, -1)
}    




function remove_flashing_border() {
    if (active_span.Element) {
        active_span.Element.classList.forEach(cls => {
            if (cls.includes('flashing-border')) {
                active_span.Element.classList.remove(cls);
            }
        });
    }
}


//function to remove tail span styling on rel mode exit
function remove_span_styles() {
    remove_flashing_border();
    let elements = document.querySelectorAll('div[id^="doc"] > span[class*="border"]');
    // Loop through the NodeList and log each element
    elements.forEach(function(element) {
         element.classList = "";
    });
}


//function to update all (head)tail span styles on entering (rev_)relation mode
//it also updates the selected span border style
function update_span_styles_on_entry(span, docIndex) {
    const selected_span_id = span.getAttribute('span-id');
    const relationCounts = {};

    //update the selected span style
    //remove all classes for this element
    span.classList = "";
    if (tool_state === 'relation_mode')             span.classList.add('red-flashing-border');
    else if (tool_state === 'rev_relation_mode')    span.classList.add('black-flashing-border');

    // Collect all tail IDs for the given head ID
    relations[docIndex].forEach(relation => {
        if (tool_state === 'relation_mode' && relation.head === selected_span_id) {
            if (relationCounts[relation.tail]) 
                relationCounts[relation.tail]++;
            else 
                relationCounts[relation.tail] = 1;
        }
        else if (tool_state === 'rev_relation_mode' && relation.tail === selected_span_id) {
            if (relationCounts[relation.head]) 
                relationCounts[relation.head]++;
            else 
                relationCounts[relation.head] = 1;
        }
    });

    // Update the style for each tail span
    Object.keys(relationCounts).forEach(span_id => {
        const cand_span = document.querySelector(`#doc${docIndex} [span-id='${span_id}']`);
        if (cand_span) {
            level = Math.min(4, relationCounts[span_id]);
            cand_span.classList = "";
            if (tool_state === 'relation_mode')             cand_span.classList.add(`tail-border-${level}`);
            else if (tool_state === 'rev_relation_mode')    cand_span.classList.add(`head-border-${level}`);
        };
    });
}


function update_border_style(docIndex, span_id, delta) {
    //update the border for the new head/tail relation
    const cand_span = document.querySelector(`#doc${docIndex} [span-id='${span_id}']`);
    if (cand_span) {
        let class_list = cand_span.classList;
        // Find the first class that starts with 'tail-border-'
        let level = 0;
        let target_class = Array.from(class_list).find(cls => cls.includes('-border-'));
        if (target_class) level = parseInt(target_class.match(/-border-(\d+)/)?.[1], 10);
        level += delta;
        level = Math.min(4, level);

        cand_span.classList = "";
        if (tool_state === 'relation_mode')             cand_span.classList.add(`tail-border-${level}`);
        else if (tool_state === 'rev_relation_mode')    cand_span.classList.add(`head-border-${level}`);
    };
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
