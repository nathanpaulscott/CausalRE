///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//TO DO
//potentially add in a comment key to the spans/relations, each one being a list of dicts with keys 'text','datetime','person'
//then add edit span functionality, where we left click on a span, it brings up the comments chain, we click on it to add to the chain, which brings up a text dialog allowing a comment ot be added, datetime is auto, person is also voluntary
//similarly for the relations, add a comment key, then in relation_mode/rev_relation_mode if we doubleclick on a tail/head, it brings up the current relations menu, if we click on one of them, it brings up the comments chain, if we click on it it allows adding to the chain etc.  More complex than the span case.


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
let input_format = 'min';     //hard coded to min for now
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
    border: 4px dashed red;  // !important; 
    animation: red-flashing-border 0.5s linear infinite; /* Flashes 2x per second */
}

@keyframes black-flashing-border {
    0% { border-color: transparent; }
    50% { border-color: black; }
    100% { border-color: transparent; }
}
.black-flashing-border {
    border: 4px dashed black;  // !important;
    animation: black-flashing-border 0.5s linear infinite; /* Flashes 2x per second */
}

.tail-border-1 { border: 3px dashed black !important; }
.tail-border-2 { border: 5px dashed black !important; }
.tail-border-3 { border: 7px dashed black !important; }
.tail-border-4 { border: 9px dashed black !important; }

.head-border-1 { border: 3px dashed red !important; }
.head-border-2 { border: 5px dashed red !important; }
.head-border-3 { border: 7px dashed red !important; }
.head-border-4 { border: 9px dashed red !important; }

#current_mode {
   color: red;
   font-weight: bold;
   font-size: 14px;
}

/* Adjust body padding to accommodate instruction div */
body { padding-top: 50px; }


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
        newStyles += `\n
            [span-type="${type_name}"] {\n
                background-color: ${type_color};\n
                border: 1px solid grey;\n
            }\n`;
    });

    //Append new styles to the top of the styles element, so it is least important
    styleElement.insertBefore(document.createTextNode(newStyles), styleElement.firstChild);
}


function add_instructions() {
    const instructions = document.createElement('div');
    instructions.id = 'topInstructions';
    instructions.style.position = 'fixed';
    instructions.style.top = '0px';
    instructions.style.left = '400px';
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
            - <strong>Click and Drag</strong> to select spans of text to annotate.<br>
            - <strong>Left-Click</strong> on any annotated span to edit the type.<br>
            - <strong>Right-Click</strong> on any annotated span to remove that span.<br>
            <strong>Relation Mode:</strong><br>
            - <strong>Ctrl-Click</strong> on any span to move to relation mode (selected span as head) and see the selected span's tail spans.<br>
            - <strong>Left-Click</strong> on any span to add the relation with it as tail.<br>
            - <strong>Right-Click</strong> on any highlighted tail span to remove the relation to it.<br>
            <strong>Reverse Relation Mode:</strong><br>
            - <strong>Shift-Click</strong> on any span to move to reverse relation mode (selected span as tail) and see the selected span's head spans.<br>
            - <strong>Left-Click</strong> on any span to add the relation with it as head.<br>
            - <strong>Right-Click</strong> on any highlighted head span to remove the relation to it.<br>
            <strong>Go Back to Span Mode:</strong><br>
            - Press <strong>ESC</strong> or <strong>Shift-Click</strong> or <strong>Ctrl-Click</strong> on a non-span to go back to Span Mode.<br>
        </div>
    `;

    const topContainer = document.getElementById('topContainer');
    // Append the instructions div to 'topContainer'
    topContainer.appendChild(instructions);
    
    return instructions
}



function add_export_and_view_results_buttons() {
    //add the export and view results buttons
    const buttonsContainer = document.createElement('div');
    buttonsContainer.style.position = 'fixed';
    buttonsContainer.style.backgroundColor = 'white';
    buttonsContainer.style.width = '100%';
    buttonsContainer.style.top = '0px';
    buttonsContainer.style.padding = '10px';
    buttonsContainer.style.textAlign = 'left';
    buttonsContainer.style.fontSize = '14px';
    buttonsContainer.style.fontFamily = 'Arial, sans-serif';
    buttonsContainer.style.zIndex = '1000';
    instructions.style.borderBottom = '1px solid #ddd';

    const statediv = document.createElement('div');
    statediv.style.padding = '0px';
    statediv.innerHTML = '<strong>Current Mode:</strong> <span id="current_mode">span_mode</span>';
    buttonsContainer.appendChild(statediv);

    const exportButton = document.createElement('button');
    exportButton.textContent = 'Export Data';
    exportButton.onclick = function() {export_data("export");};
    exportButton.style.marginLeft = '0px';
    exportButton.style.marginTop = '10px';
    buttonsContainer.appendChild(exportButton);

    const viewResultsButton = document.createElement('button');
    viewResultsButton.textContent = 'View Results';
    viewResultsButton.onclick = function() {export_data("view");};
    viewResultsButton.style.marginLeft = '5px';
    viewResultsButton.style.marginTop = '10px';
    buttonsContainer.appendChild(viewResultsButton);

    //add the buttonsContainer to the topContainer div
    const container = document.getElementById('topContainer');
    container.prepend(buttonsContainer);
}


function add_tooltip_info() {
    // Create a tooltip element and add it to the document
    const tooltip = document.createElement('div');
    tooltip.id = 'tooltip_info';
    tooltip.style.position = 'absolute';
    tooltip.style.backgroundColor = '#333';
    tooltip.style.color = '#fff';
    tooltip.style.padding = '5px';
    //tooltip.style.border = '2px solid red';
    tooltip.style.borderRadius = '5px';
    tooltip.style.fontSize = '12px';
    //tooltip.style.fontWeight = 'bold';
    tooltip.style.fontFamily = 'Arial, sans-serif';
    tooltip.style.display = 'none';
    tooltip.style.zIndex = '1000';

    const container = document.getElementById('InputContainer');
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
    tooltip.style.border = '2px solid red';
    //tooltip.style.borderRadius = '5px';
    tooltip.style.display = 'none';
    tooltip.style.zIndex = '1500';
    tooltip.style.fontSize = '14px';
    tooltip.style.fontWeight = 'bold';
    tooltip.style.fontFamily = 'Arial, sans-serif';

    const container = document.getElementById('InputContainer');
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
        if (tool_state === 'span_mode' && target.getAttribute('type') === "annotated")
            edit_span_handler(event, docIndex, null, "edit");
        //if already in any of the relation modes, process add the selected span as a relation to add
        else if (tool_state !== 'span_mode' && target.getAttribute('type') !== "unannotated") {
            const spanId = target.getAttribute('span-id');
            //check that the clicked span was not the same as the active_span span and is in the same docDiv, if not exit
            if (docIndex !== active_span.Index || spanId === active_span.Element.getAttribute('span-id')) return;
            //all good, we clicked on an acceptable candidate span in the same docDiv, so process the add relation
            edit_relation_handler(event, docIndex, 'add');
            //console.log('processing candidate head/tail span');
        }
        else 
            return;
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
        const span_id = target.getAttribute('span-id');
        const span_type = target.getAttribute('span-type');
        // Display the tooltip with the content
        tooltip_info.textContent = `${span_id} (${span_type})`;
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
function reset_vars() {
    source_data =   {};
    sources =       [];
    offsets =       [];
    raw_docs =      [];
    spans =         [];
    relations =     [];
}



document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const topInstructions = document.getElementById('topInstructions');

    fileInput.addEventListener('change', function(event) {
        //this selects the file import .json file and loads it into the associated js vars and modifies the display on screen
        //this file will have schema, raw text and spans and relations in it, the annotated data part is not used as that is just for human comprehension only
        const file = event.target.files[0];
        if (!file) {
            alert("No file selected.");
            return;
        }

        reset_vars();
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
                //load the schema
                schema = temp.schema;
                update_span_styles_from_schema();

                //load spans and relations and display docs
                if ("spans" in temp && "relations" in temp && temp.spans.length === raw_docs.length && temp.relations.length === raw_docs.length) {
                    //load the annotations if they are there
                    spans = temp.spans;
                    relations = temp.relations;
                    display_documents("load");
                }
                
                else if (!("spans" in temp) && !("relations" in temp)) {
                    display_documents("reset");
                }
                else {
                    alert('if you give annotations, you must give both spans and relations keys and they both need to be list of lists of objects with one outer list element per document')
                    display_documents("reset");
                }
            } 
            catch (error) {
                alert('Failed to load JSON file.  See console for details');
                console.log(`Failed to load JSON file. ${error}`);
            }
        };
        reader.readAsText(file);
        //fileInput.value = ''; 
    });
});



function checkOverlap(a, b) {
    return a.start <= b.end && b.start <= a.end
}

function filterOverlappingAnnotations(annotations) {
    removed = []
    // Sort annotations by start, then by end in descending order to prioritize longer spans
    annotations.sort((a, b) => a.start - b.start || b.end - a.end);
    // Iterate through copied annotations and nullify shorter overlapping spans
    for (let i = 0; i < annotations.length; i++) {
        if (annotations[i] === null) continue; // Skip already nullified annotations
        let longestSpan = annotations[i];
        for (let j = i + 1; j < annotations.length && annotations[j].start <= longestSpan.end; j++) {
            if (annotations[j] === null) continue;
            if (checkOverlap(longestSpan, annotations[j])) {
                // There is an overlap, check which one is longer
                if (longestSpan.end - longestSpan.start < annotations[j].end - annotations[j].start) {
                    // The j-th span is longer
                    removed.push({ ...annotations[i] }); // Push a copy of the current span to removed
                    annotations[i] = null; // Nullify the current span
                    longestSpan = annotations[j]; // Update the longestSpan
                    i = j - 1; // Move the outer loop's index to just before j
                    break;
                } else {
                    // The i-th span is longer or they are equal, nullify the j-th span
                    removed.push({ ...annotations[j] }); // Push a copy of the current span to removed
                    annotations[j] = null;
                }
            }
        }
    }

    //return the non null elements
    return [annotations.filter(span => span !== null), removed];
}


function display_documents(option) {
    //this function loads in the raw_docs to the browser and loads the annotations if otpion is "load", otherwise it resets the annotations if option is "reset"

    //this builds the div for each doc and displays it on the browser
    const container = document.getElementById('dataContainer');
    
    //clear any existing contents
    container.innerHTML = '';
    
    removed_all = [];

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
            //Sort annotations by start index to ensure proper text chunking
            spans[index].sort((a, b) => a.start - b.start);
            //Filter out overlapping annotations and keep the longest
            //NOTE: currently I am not supporting overlapping spans, so I remove the shortest of any overlaps
            //NOTE: currently I am not supporting overlapping spans, so I remove the shortest of any overlaps
            //NOTE: currently I am not supporting overlapping spans, so I remove the shortest of any overlaps
            let result = filterOverlappingAnnotations(spans[index]);
            if (result[1].length > 0) removed_all = removed_all.concat(result[1]);
            spans[index] = result[0];
            // Filter relations to keep only those that involve valid annotation IDs
            const validIds = new Set(spans[index].map(x => x.id));
            relations[index] = relations[index].filter(x => 
                validIds.has(x.head) && validIds.has(x.tail)
            );
            //NOTE: currently I am not supporting overlapping spans, so I remove the shortest of any overlaps
            //NOTE: currently I am not supporting overlapping spans, so I remove the shortest of any overlaps
            //NOTE: currently I am not supporting overlapping spans, so I remove the shortest of any overlaps

            //DANGER
            //This now modifies the spans in teh docDiv to show the annotated spans, it does not handle overlapping spans
            //I have put code above to remove any overlapping spans, 
            //but if you ever allow overlapping spans, you MUST modify this code so that it doesn't cause repeated text in teh docDiv
            //DANGER
            let lastIndex = 0; // Track the last index of text processed
            let updatedInnerHTML = ''; // Build new HTML for the document
            // Iterate through each annotation and build updated inner HTML
            spans[index].forEach(x => {
                // Add unannotated text before this annotation
                if (x.start > lastIndex) 
                    updatedInnerHTML += `<span type="unannotated">${text.slice(lastIndex, x.start)}</span>`;
                // Add annotated text
                updatedInnerHTML += `<span type="annotated" span-id="${x.id}" span-type="${x.type}">${text.slice(x.start, x.end + 1)}</span>`;
                lastIndex = x.end + 1;
            });
            // Add any remaining unannotated text after the last annotation
            if (lastIndex < text.length)
                updatedInnerHTML += `<span type="unannotated">${text.slice(lastIndex)}</span>`;
            // Set the new HTML to the docDiv
            docDiv.innerHTML = updatedInnerHTML;
        }
        //add to the container
        container.appendChild(docDiv);
    });

    // Alert the removed spans
    if (removed_all.length > 0) alert(`There were some overlapping spans removed, the longest of the overlapping were kept, the removed spans were.\n${JSON.stringify(removed_all, null, 2)}`);
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
    menu.style.fontWeight = 'bold';
    menu.style.fontSize = '14px';
    menu.style.fontFamily = 'Arial, sans-serif';

    if (action === "add") {
        schema["span_types"].forEach(x => {
            const item = document.createElement('div');
            item.textContent = `Annotate as ${x["name"]}`;
            item.style.backgroundColor = x["color"];
            item.style.padding = '5px';
            item.onclick = () => {
                add_span(docIndex, x["name"], x["color"], range);
                document.body.removeChild(menu); // Close menu after selection
            }
            menu.appendChild(item);
        });
    }
    else if (action === "edit") {
        const target_span = event.target;
        const current_type = target_span.getAttribute('span-type');
        const choice_types = schema["span_types"].filter(x => x.name !== current_type);
        choice_types.forEach(x => {
            const item = document.createElement('div');
            item.textContent = `Set type as ${x["name"]}`;
            item.style.backgroundColor = x["color"];
            item.style.padding = '5px';
            item.onclick = () => {
                edit_span(target_span, docIndex, x["name"]);
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
        item.style.padding = '5px';
        menu.style.border = '2px solid red';
        //add the click listener
        item.onclick = () => {
            remove_span(target_span, docIndex);
            document.body.removeChild(menu); // Close menu after selection
        }
        menu.appendChild(item);
    }
    document.body.appendChild(menu);
}



function get_next_cnt(list_of_dicts) {
    // Extract all IDs using map by matching the last sequence of digits in the ID
    //list_of_dicts is either the span list or relations list for the given docIndex
    const ids = list_of_dicts.map(x => {
        const match = x.id.match(/\d+$/);
        return match ? parseInt(match[0], 10) : null;
    }).filter(id => !isNaN(id));  // Filter out non-numeric values to account for potential parsing failures

    // Find the maximum ID using Math.max, defaulting to 0 if the array is empty
    const max_id = ids.length > 0 ? Math.max(...ids) : 0;

    // Generate the next available ID by incrementing the highest found ID
    return max_id + 1;
}



function add_span(docIndex, type, color, range) {
    const next_cnt = get_next_cnt(spans[docIndex]);
    const span_id = `E${docIndex}_${next_cnt}`;

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
    newSpan.setAttribute('span-id', span_id);
    newSpan.setAttribute('span-type', type);
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
        id:     span_id,
        type:   type,
        start:  absoluteStartIndex,
        end:    absoluteEndIndex,
        span:   selectedText,
    });
}


function edit_span(target_span, docIndex, type) {
    //change the span type in the DOM
    target_span.setAttribute('span-type', type);

    //change the span type in the spans object
    const span_id = target_span.getAttribute('span-id');
    //find the selected span and edit it
    for (const x of spans[docIndex]) {
        if (x.id === span_id) {
            x.type = type;
            break;
        }
    }
}



function remove_span(target_span, docIndex) {
    //this removes the chosen span from the spans[docIndex] list and the relations[docIndex] list
    const docDiv = document.querySelector(`#dataContainer #doc${docIndex}`);
    const span_id = target_span.getAttribute('span-id');

    //remove the selected span from spans
    spans[docIndex] = spans[docIndex].filter(span => span.id !== span_id);
    //remove any relation with the selected span and head or tail from relations
    relations[docIndex] = relations[docIndex].filter(relation => relation.head !== span_id && relation.tail !== span_id);

    //remove the span from the docDiv
    //Step 1: Change the type attribute to "unannotated" and remove attributes span-id and span-type
    target_span.setAttribute('type', 'unannotated');
    target_span.removeAttribute('span-id');
    target_span.removeAttribute('span-type');

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
    //remove all open menus
    removeExistingMenus();
}


function exit_relation_mode() {
    //remove all span styles on exit
    remove_span_styles();
    //reset the active_span object
    active_span = {Element: null, Index: null};
    // Update the tool_state and current mode display
    tool_state = 'span_mode';
    document.getElementById('current_mode').innerText = tool_state;
    //remove all open menus
    removeExistingMenus();
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
    menu.style.fontWeight = 'bold';
    menu.style.fontSize = '14px';
    menu.style.fontFamily = 'Arial, sans-serif';

    // Filter relation types based on action
    let relationTypes = schema["relation_types"];
    if (action === "rmv") {
        //filter the relation_types to only those that exist for this head-tail pair as only these can be removed
        const existingRelations = relations[docIndex].filter(x => x.head === headId && x.tail === tailId);
        relationTypes = relationTypes.filter(x => existingRelations.some(y => y.type === x.name));
    }

    // Create menu items for filtered relation types
    if (relationTypes.length === 0) return;

    relationTypes.forEach((type) => {
        const item = document.createElement('div');
        if (action === 'add')       item.textContent = `Add relation: ${type.name}`;
        else if (action === 'rmv')  item.textContent = `Rmv relation: ${type.name}`;
        item.style.backgroundColor = type.color;
        item.style.padding = '5px';

        item.onclick = () => {
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
        menu.appendChild(item);
    });
    document.body.appendChild(menu);
}



//check that a candidate relation has not already been added
function relation_already_exists(docIndex, headId, tailId, type) {
    return relations[docIndex].some(rel => rel.head === headId && rel.tail === tailId && rel.type === type);
}



function show_relation_already_exists_msg(x,y) {
    //show dissappearing popup msg that this relation has already been added
    //tooltip_caution.style.position = 'fixed';
    tooltip_caution.style.display = 'block';
    tooltip_caution.style.left = `${x}px`; // Position near the click horizontally
    tooltip_caution.style.top = `${y}px`; // Position near the click vertically
    tooltip_caution.textContent = 'Relation Already Exists'

    setTimeout(function() {
        tooltip_caution.style.display = 'none';
    }, 2000);  //hide after this many ms
}


function add_relation(docIndex, head_id, tail_id, type, color) {
    const next_cnt = get_next_cnt(relations[docIndex]);
    const rel_id = `R${docIndex}_${next_cnt}`;

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
