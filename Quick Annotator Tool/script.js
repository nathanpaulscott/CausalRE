//TO DO
//19Jan2025
//VIP => you need to convert the incoming format to require idx format, i.e. no 'id' key, the head and tail are the list idx of the spans in the span list (call this the idx format or the spert format)
//Internally you should add unique ids to the spans and rels and move each of them to a dict of dicts (object of objetcs) where the key is the unique id and the head/tail value is the span id as opposed to the idx.
//WHY: This format is easier to remove spans without affecting the head/tail values.  This is almost what you are doing internally at the moment.  So you really need just the import conversion code.  
//I think currently internally you are still using a list of dicts and each dict has an id key, this is stupid
//You also need to then make an export conversion to convert the internal format to the idx format on view or export
//in summary you need to do 3 things:
//1) make the import require the idx format and then convert it to the internal id format
//2) change the current internal format from list of objects to object of objects with the id as key
//3) add in a conversion function to convert the internal id format back to idx format on view or export data

//For the time being I am converting my mistaken list of objects form in my model code back to idx FormData, but 










///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//GLOBAL PARAMS
let source_data = {};   //holds the dict of source data, each key is a source id and the value is the source text
let data = [];      // holds the list of observations, each obs has keys: tokens, spans, relations and optionally source, offset
let schema = {};

//control vars
let active_span = {span_id: null, docIndex: null};
let tool_state = 'span_mode'; // Possible values: 'span_mode', 'relation_mode'
let mouseDownDocIndex = null;
let input_format = 'min';     //hard coded to min for now
let filename = '';
let app_ver = 1;
let doubleclick_flag = false;
const span_id_prefix = 'E';    // new spans will have id => `${span_id_prefix}${next_count}`
const relation_id_prefix = 'R';   // new relations will have id => `${relation_id_prefix}${next_count}`

//set the offset for the popup messages near the click point
const msg_offset_x = 40;
const msg_offset_y = -40;
const info_offset_x = -20;
const info_offset_y = 20;

/*
import data format....
{
    'data': [
        {
            'tokens': [list of strings],
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
    'schema': {same format as previously}
}
*/

//make a default schema object
schema = {
    "span_types":[
        {
            "name":"E_type1",
            "color":"rgba(135,206,250, 0.3)"
        },
        {
            "name":"E_type2",
            "color":"rgba(144,238,144, 0.3)"
        },
        {
            "name":"E_type3",
            "color":"rgba(255,182,193, 0.3)"
        },
        {
            "name":"E_type4",
            "color":"rgba(255,165,0, 0.3)"
        }
    ],
    "relation_types":[
        {
            "name":"R_type1",
            "color": "rgba(135,206,250, 0.3)"
        },
        {
            "name":"R_type2",
            "color": "rgba(144,238,144, 0.3)"
        },
        {
            "name":"R_type3",
            "color": "rgba(255,182,193, 0.3)"
        },
        {
            "name":"R_type4",
            "color": "rgba(255,165,0, 0.3)"
        }
    ]
}

//make the instructions msg
const instructions_text = `
<div id="instructions-header"><strong>INSTRUCTIONS</strong></div>
<div id="instructions-content" style="display: none;">
    <br>
    <strong>Span Mode:</strong><br>
    - <strong>Click and Drag</strong> to select spans of text to annotate.<br>
    - <strong>RIght-Click</strong> on any span (while in span_mode) to edit/remove the span.<br>
    <strong>Relation Mode:</strong><br>
    - <strong>Left-Click</strong> on any span to move to relation mode (selected span as head, flashing red) and see all tail spans (black border).<br>
    - <strong>Right-Click</strong> on any span (while in relation mode) to add/remove the relation with that span as tail to the flashing head span.<br>
    <strong>Reverse Relation Mode:</strong><br>
    - <strong>Ctrl-Left-Click</strong> on any span to move to rev_relation mode (selected span as tail, flashing, black) and see all head spans (red border).<br>
    - <strong>Right-Click</strong> on any span (while in rev_relation mode) to add/remove the relation with that span as head to the flashing tail span.<br>
    <strong>Go Back to Span Mode:</strong><br>
    - Press <strong>ESC</strong>
    <br>
    <strong>Ver: ${app_ver}:</strong><br>
    - NOTE: span start is the actual start token idx, end is the actual end token idx + 1<br>
    - NOTE: overlapping spans are supported<br>
</div>`;


///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//Modify html document
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//Inject CSS
const style = document.createElement('style');
style.innerHTML = `
#current_mode {
   color: red;
   font-weight: bold;
   font-size: 14px;
}

/* Adjust body padding to accommodate instruction div */
body {
    padding-top: 85px;
    font-family: 'Arial', sans-serif;
    font-size: 14px;
}

/*style for the doc header*/
div[id^="header"] {
    font-weight:    bold;
    /*font-size:      12px;*/
}

/*style for the doc div*/
div[id^="doc"] {
    display:        block;
    max-width:      100%;
    border:         1px solid #ccc;
    padding:        10px;
    max-height:     200px;
    overflow-x:     auto;
    overflow-y:     auto;
    margin-bottom:  20px;
}


/*set the dafault span padding which spaces the words*/
div[id^="doc"] > span {
    /*font-size: 14px;*/
    display: inline-block;
    white-space: normal;
    overflow-wrap: normal;

    padding-left: 2px;      /*this affects the word spacing*/
    padding-right: 2px;     /*this affects the word spacing*/
    margin-top: 2px;
    padding-top: 2px;
    padding-bottom: 2px;
    margin-bottom: 2px;
}

/*////////////////////////////////////////////////////*/
/*////////////////////////////////////////////////////*/
/*////////////////////////////////////////////////////*/
/*for the tooltip*/
div[id="tooltip_info"] {
    position: absolute;
    background-color: #333;
    color: #fff;
    padding: 5px;
    /*border: 2px solid red;*/
    border-radius: 5px;
    font-size: 12px;
    /*font-weight: bold;*/
    font-family: Arial, sans-serif;
    display: none;
    z-index: 1000;
}

div[id="tooltip_caution"] {
    position: absolute;
    background-color: white;
    color: red;
    padding: 5px;
    border: 2px solid red;
    /*border-radius: 5px;*/
    display: none;
    z-index: 1500;
    font-size: 14px;
    font-weight: bold;
    font-family: Arial, sans-serif;
}

div[id="topInstructions"] {
    position:       fixed;
    top:            0px;
    left:           400px;
    width:          100%;
    background-color:   #f9f9f9;
    padding:        10px;
    text-align:      left;
    font-size:       14px;
    font-family:     Arial, sans-serif;
    border-bottom:   1px solid #ddd;
    z-index:         1000;
    max-height:      30px;
    overflow:       hidden;
    cursor:         pointer;
    transition:     max-height 0.3s ease-in-out;
}

div[id="import_container"] {
    background-color:    white;
    width:              100%;
    position:           fixed;
    top:                60px;
    border-bottom:       2px solid grey;
    padding:            10px 10px 10px 10px;
    text-align:          left;
    font-size:           14px;
    font-family:         Arial, sans-serif;
    z-index:             1000;
}

div[id="buttons_container"] {
    position:       fixed;
    background-color: white;
    width:          100%;
    top:            0px;
    padding:        10px;
    text-align:      left;
    font-size:       14px;
    font-family:     Arial, sans-serif;
    z-index:         1000;
}

div[id="menu"] {
    position:           absolute;
    background-color:   white;
    border: 1px solid   black;
    padding:            5px;
    z-index:            1000;
    font-weight:        bold;
    font-size:          14px;
    font-family:        Arial, sans-serif;
}
div[id="menu"] > div {
    background-color: rgba(221,221,221,1);
    padding: 7px;
}
div[id="menu"] > div:hover {
    padding: 5px;
    border: 2px solid black;
}


/*////////////////////////////////////////////////////*/
/*set the border styles for the annotated spans*/
/*////////////////////////////////////////////////////*/
div[id^="doc"] > span[span-boundary*="span"] {
    border-top: 1px solid rgba(100,100,100,1);
    border-bottom: 1px solid rgba(100,100,100,1);
    padding-top: 1px;
    padding-bottom: 1px;
}
div[id^="doc"] > span[span-boundary*="span-start"] {
    border-left: 1px solid rgba(100,100,100,1);
    padding-left: 1px;
}
div[id^="doc"] > span[span-boundary*="span-end"] {
    border-right: 1px solid rgba(100,100,100,1);
    padding-right: 1px;
}
div[id^="doc"] > span[span-boundary*="span-both"] {
    border-left: 1px solid rgba(100,100,100,1);
    border-right: 1px solid rgba(100,100,100,1);
    padding-left: 1px;
    padding-right: 1px;
}

/*////////////////////////////////////////////////////*/
/*set the borders for the selected source relation*/
/*////////////////////////////////////////////////////*/
@keyframes red-flashing-border {
    0%, 100% { border-color: red; }
    50% { border-color: transparent; }
}
@keyframes black-flashing-border {
    0%, 100% { border-color: black; }
    50% { border-color: transparent; }
}
div[id^="doc"] > span[class*="red-flashborder"] { 
    animation: red-flashing-border 0.5s linear infinite;
}
div[id^="doc"] > span[class*="black-flashborder"] { 
    animation: black-flashing-border 0.5s linear infinite; 
}
div[id^="doc"] > span[class*="-flashborder"] {
    border-style: solid;
    border-width: 3px 0px 3px 0px;    /*top right bottom left*/
    /*need this so the text doesn't move*/
    margin-top: 1px;
    padding-top: 0px;
    padding-bottom: 0px;
    margin-bottom: 1px;
}
div[id^="doc"] > span[class*="-flashborder-start"] {
    border-width: 3px 0px 3px 2px;    /*top right bottom left*/
    padding-left: 0px;      /*this affects the word spacing*/
}
div[id^="doc"] > span[class*="-flashborder-end"] {
    border-width: 3px 2px 3px 0px;    /*top right bottom left*/
    padding-right: 0px;      /*this affects the word spacing*/
}
div[id^="doc"] > span[class*="-flashborder-both"] {
    border-width: 3px 2px 3px 2px;    /*top right bottom left*/
    padding-left: 0px;      /*this affects the word spacing*/
    padding-right: 0px;      /*this affects the word spacing*/
}

/*////////////////////////////////////////////////////*/
/*set the borders of the other relations to the selected side*/
/*////////////////////////////////////////////////////*/
div[id^="doc"] > span[class*="head-border"]   { 
    border-color: red; 
}
div[id^="doc"] > span[class*="tail-border"]   { 
    border-color: black; 
}
div[id^="doc"] > span[class*="-border"]      { 
    border-style: solid;
    border-width: 3px 0px 3px 0px; 
    /*need this so the text doesn't move*/
    margin-top: 1px;
    padding-top: 0px;
    padding-bottom: 0px;
    margin-bottom: 1px;
}
div[id^="doc"] > span[class*="-border-start"]      { 
    border-width: 3px 0px 3px 2px; 
    padding-left: 0px;
}
div[id^="doc"] > span[class*="-border-end"]      { 
    border-width: 3px 2px 3px 0px; 
    padding-right: 0px;
}
div[id^="doc"] > span[class*="-border-both"]      { 
    border-width: 3px 2px 3px 2px; 
    padding-left: 0px;
    padding-right: 0px;
}

`; // Close the CSS string and statement properly



///////////////////////////////////////////
//Define some generic functions
///////////////////////////////////////////
function hasSpanId(element) {
    //to test if a span has an attribute, this is fast:
    return element.hasAttribute("span-id");
}

//utility to check we have clicked inside an acceptable div
function get_parent_div_for_mouse_event(target) {
    // Use closest to find the parent div that has id that starts with 'doc'
    let docDiv = target.closest('div[id^="doc"]');
    if (!docDiv) return null; // No matching div found
    //got to here so passed the check
    return docDiv;
}

// Utility function to remove existing menus
function remove_menus() {
    const existingMenus = document.querySelectorAll('div[id="menu"]');
    existingMenus.forEach(menu => menu.parentNode.removeChild(menu));
}

function delete_span_styles() {
    //delets all span_style from <style>...</style>
    // Find the <style> element
    let styleElement = document.querySelector('style');

    const sheet = styleElement.sheet;
    if (!sheet) return; // Skip if the stylesheet is not accessible

    // Access the rules in the stylesheet
    const rules = sheet.cssRules// || sheet.rules;

    // Iterate backwards to avoid index issues when deleting
    let rulesDeleted = false;
    for (let i = rules.length - 1; i >= 0; i--) {
        const rule = rules[i];
        // Check if the rule's selector starts with "[span-type="
        if (rule.selectorText && rule.selectorText.startsWith('div[id^="doc"] > span[span-type~=')) {
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
        newStyles += `div[id^="doc"] > span[span-type~="${type_name}"] { background-color: ${type_color}; }\n`;
    });
    //Append new styles to the top of the styles element, so it is least important
    styleElement.insertBefore(document.createTextNode(newStyles), styleElement.firstChild);
}

function add_instructions() {
    const instructions = document.createElement('div');
    instructions.id = 'topInstructions';
    instructions.innerHTML = instructions_text;
    const topContainer = document.getElementById('topContainer');
    // Append the instructions div to 'topContainer'
    topContainer.appendChild(instructions);
    
    return instructions
}

function add_import_button() {
    // Get the container where the elements will be added
    const importContainer = document.createElement('div');
    importContainer.id = 'import_container';
    
    // Create and configure the label
    const label = document.createElement('label');
    label.setAttribute('for', 'fileInput');
    label.textContent = 'Data Import';
    label.style.marginRight = '10px'; // Style the label

    // Create and configure the file input
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.id = 'fileInput';
    fileInput.accept = '.json';

    // Add label and file input to the container
    importContainer.appendChild(label);
    importContainer.appendChild(fileInput);

    const container = document.getElementById('topContainer');
    container.prepend(importContainer);
}

function add_export_and_view_results_buttons() {
    //add the export and view results buttons
    const buttonsContainer = document.createElement('div');
    buttonsContainer.id = 'buttons_container';

    const statediv = document.createElement('div');
    statediv.style.padding = '0px';
    statediv.innerHTML = '<strong>Current Mode:</strong> <span id="current_mode">span_mode</span>';
    buttonsContainer.appendChild(statediv);

    const exportButton = document.createElement('button');
    exportButton.textContent = 'Export Data';
    exportButton.onclick = function() {export_data("export");};
    exportButton.style.marginLeft = '0px';
    exportButton.style.marginTop = '10px';
    exportButton.style.fontSize = '12px';
    buttonsContainer.appendChild(exportButton);

    const viewResultsButton = document.createElement('button');
    viewResultsButton.textContent = 'View Results';
    viewResultsButton.onclick = function() {export_data("view");};
    viewResultsButton.style.marginLeft = '5px';
    viewResultsButton.style.marginTop = '10px';
    viewResultsButton.style.fontSize = '12px';
    buttonsContainer.appendChild(viewResultsButton);

    //add the buttonsContainer to the topContainer div
    const container = document.getElementById('topContainer');
    container.prepend(buttonsContainer);
}

function add_tooltip_info() {
    // Create a tooltip element and add it to the document
    const tooltip = document.createElement('div');
    tooltip.id = 'tooltip_info';

    const container = document.getElementById('dataContainer');
    container.parentNode.insertBefore(tooltip, container);
    
    return tooltip
}

function add_tooltip_caution() {
    // Create a tooltip element and add it to the document
    const tooltip = document.createElement('div');
    tooltip.id = 'tooltip_caution';

    const container = document.getElementById('dataContainer');
    container.parentNode.insertBefore(tooltip, container);

    return tooltip
}

//add the styles to the html doc
document.head.appendChild(style);
//add the span_styles from the schema
update_span_styles_from_schema();
//add instructions
const instructions = add_instructions();
//add the import button
add_import_button();
//add the export and view results buttons
add_export_and_view_results_buttons();
//add the tooltip info div
const tooltip_info = add_tooltip_info();
//add the no add relation tooltip
const tooltip_caution = add_tooltip_caution();

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//SPANS
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//incoming handler
///////////////////////////////////////////////////////////////////
function edit_span_handler(event, docIndex, selected_span_tags, action) {
    //make give the span type menu
    if (action==='add')
        show_span_menu_add(event, docIndex, selected_span_tags);
    else
        show_span_menu_choose(event, 'Choose Span: ').then(span_id => {
            show_span_menu_edit(event, docIndex, span_id);
        });
}

///////////////////////////////////////////////
//menus
///////////////////////////////////////////////
function show_span_menu_add(event, docIndex, selected_span_tags) {
    const menu = document.createElement('div');
    menu.id = 'menu';
    menu.style.left = `${event.clientX + window.scrollX}px`;
    menu.style.top = `${event.clientY + window.scrollY}px`;

    schema["span_types"].forEach(span_type => {
        const item = document.createElement('div');
        item.textContent = `Annotate as ${span_type["name"]}`;
        item.style.backgroundColor = span_type["color"];
        item.onclick = () => {
            add_span(docIndex, span_type["name"], selected_span_tags);
            document.body.removeChild(menu); // Close menu after selection
        }
        menu.appendChild(item);
    });

    document.body.appendChild(menu);
}

function show_span_menu_choose(event, msg) {
    return new Promise((resolve, reject) => {
        const menu = document.createElement('div');
        menu.id = 'menu';
        menu.style.left = `${event.clientX + window.scrollX}px`;
        menu.style.top = `${event.clientY + window.scrollY}px`;

        const target_span = event.target;
        const current_ids = target_span.getAttribute('span-id').split(' ');
        if (current_ids.length === 1) {
            resolve(current_ids[0]); // If there is only one id, resolve it immediately
            return;
        }
        //otherwise show the menu to select an id
        current_ids.forEach(id => {
            const item = document.createElement('div');
            item.textContent = `${msg}: ${id}`;
            item.onclick = () => {
                document.body.removeChild(menu); // Close menu after selection
                resolve(id); // Resolve the promise with the chosen id
            };
            menu.appendChild(item);
        });
        document.body.appendChild(menu);
    });
}

function show_span_menu_edit(event, docIndex, span_id) {
    //get the start and end idx of the chosen span
    const selected_span_tags = find_span_tags_by_span_id(docIndex, span_id);
    let span_start = parseInt(selected_span_tags[0].getAttribute('token-id'));
    let span_end = parseInt(selected_span_tags[selected_span_tags.length-1].getAttribute('token-id')) + 1;    //to make it python list style

    const menu = document.createElement('div');
    menu.id = 'menu';
    menu.style.left = `${event.clientX + window.scrollX}px`;
    menu.style.top = `${event.clientY + window.scrollY}px`;

    /////////////////////////////////////
    // Remove button for deleting the span
    /////////////////////////////////////
    const removeButton = document.createElement('button');
    removeButton.textContent = 'Remove Span';
    removeButton.style.margin = '5px';
    removeButton.style.color = 'red';
    removeButton.style.fontWeight = 'bold';
    removeButton.style.display = 'block';
    removeButton.onclick = () => {
        //set the start/end to -1 to signal delete span
        edit_span(docIndex, current_type, -1, -1, selected_span_tags, span_id);
        document.body.removeChild(menu);
    };

    /////////////////////////////////////
    // Input fields for start and end indices
    /////////////////////////////////////
    const createInputField = (labelText, initialValue, onChangeCallback) => {
        const container = document.createElement('div');

        const input = document.createElement('input');
        input.type = 'number';
        input.style.width = '50px'; // Set a specific width
        input.setAttribute('max', '100000000');
        input.setAttribute('min', '0');
        input.value = initialValue;

        const label = document.createElement('span');
        label.textContent = labelText;

        container.appendChild(label);
        container.appendChild(input);
        container.onclick = (e) => {
            e.stopPropagation(); // Prevent the menu from closing when interacting with the input
            input.focus();
        };

        input.onchange = onChangeCallback;
        return container;
    };
    //make the input handlers
    const startInputField = createInputField("Start: ", span_start, () => {
        span_start = parseInt(startInputField.querySelector('input').value);
    });
    const endInputField = createInputField("End: ", span_end, () => {
        span_end = parseInt(endInputField.querySelector('input').value);
    });

    /////////////////////////////////////
    // Apply button for confirming changes
    /////////////////////////////////////
    const applyButton = document.createElement('button');
    applyButton.textContent = 'Apply Start/End';
    applyButton.style.margin = '5px';
    applyButton.style.display = 'block';
    applyButton.style.fontWeight = 'bold';
    applyButton.onclick = () => {
        span_start = parseInt(startInputField.querySelector('input').value);
        span_end = parseInt(endInputField.querySelector('input').value);
        edit_span(docIndex, current_type, span_start, span_end, selected_span_tags, span_id);
        document.body.removeChild(menu);
    };

    menu.appendChild(removeButton);
    menu.appendChild(startInputField);
    menu.appendChild(endInputField);
    menu.appendChild(applyButton);

    /////////////////////////////////////
    //make the type menu
    /////////////////////////////////////
    const target_span = event.target;
    //get current span_id idx
    let idx = target_span.getAttribute('span-id').split(' ').indexOf(span_id);
    //get current span type
    const current_type = target_span.getAttribute('span-type').split(' ')[idx];
    // Filter out span types that are already assigned to the current span  
    const choice_types = schema["span_types"].filter(x => !current_type.includes(x.name));
    choice_types.forEach(type => {
        const item = document.createElement('div');
        item.textContent = `Change to: ${type["name"]}`;
        item.style.backgroundColor = type["color"];
        item.onclick = () => {
            edit_span(docIndex, type["name"], span_start, span_end, selected_span_tags, span_id);
            document.body.removeChild(menu); // Close menu after selection
        }
        menu.appendChild(item);
    });

    //add the menu div to the dom
    document.body.appendChild(menu);
}



/////////////////////////////////////////////////////
//add/edit span functions
/////////////////////////////////////////////////////
function add_span(docIndex, new_type, selected_span_tags) {
    // Get the next count to create a unique span ID
    const new_span_id = `${span_id_prefix}${get_next_cnt(data[docIndex].spans)}`;

    //update the dom
    add_to_span_tags(selected_span_tags, new_span_id, new_type);

    //update the data structure
    //Collect text from all spans
    let selected_text = [];
    selected_span_tags.forEach((selected_span_tag, idx) => {selected_text.push(selected_span_tag.textContent);});
    selected_text = selected_text.join(' ');
    //Get token-id attributes for start and end of the selection
    let start = parseInt(selected_span_tags[0].getAttribute('token-id'));
    let end = parseInt(selected_span_tags[selected_span_tags.length - 1].getAttribute('token-id')) + 1;
    // Append new span entry to the data structure
    add_span_to_spans(docIndex, new_type, start, end, new_span_id, selected_text);
}

function edit_span(docIndex, type, new_start, new_end, current_span_tags, span_id) {
    //new_start/end are the python style bounds (i.e real end + 1)
    //validate the start and end
    if (!validate_boundary(new_start, new_end)) return;
    //check for the delete signal
    let edit_flag = true;
    if (new_start === -1 && new_end === -1) edit_flag = false;
    //update in the dom
    const positions = get_current_positions(current_span_tags, span_id);
    if (!positions) return;
    //remove the old span range
    remove_from_span_tags(current_span_tags, positions);
    //update in data[docIndex].spans
    remove_span_from_spans(docIndex, span_id);
    
    //edit_flag is true so we have to add back to spans
    if (edit_flag) {
        //get the new range of spans
        const new_spans = find_span_tags_by_start_end(docIndex, new_start, new_end);
        //add them to the dom
        add_to_span_tags(new_spans, span_id, type);
        //Collect text from all spans
        let new_text = [];
        new_spans.forEach((new_span, idx) => {
            new_text.push(new_span.textContent);
        });
        new_text = new_text.join(' ');
        add_span_to_spans(docIndex, type, new_start, new_end, span_id, new_text);
    }
    //edit_flag is false so we are removing the span, so we have to remove it from the relations
    else {   
        remove_span_from_relations(docIndex, span_id);
    }
}

/////////////////////////////////////////////////////////////////
//helper functions for add/edit spans
/////////////////////////////////////////////////////////////////
function validate_boundary(start, end) {
    if (isNaN(start) || isNaN(end)) return false;
    if (start != -1 && end != -1 && start >= end) return false;
    return true;
}

function get_next_cnt(items) {
    /*
    get the next integer for the items list.  Looks at all the entries and thier last digits and finds the max and returns max+1
    */
    const ids = items.map(x => {
        const match = x.id.match(/\d+$/);
        return match ? parseInt(match[0]) : null;
    }).filter(id => id != null);
    const max_id = ids.length > 0 ? Math.max(...ids) : -1;
    return max_id + 1;
}

function find_span_tags_by_span_id(docIndex, span_id) {
    /*
    this finds all the child span tags containing the given span_id
    */
    const docDiv = document.getElementById(`doc${docIndex}`);
    return Array.from(docDiv.children).filter(tag => 
        tag.tagName === 'SPAN' && 
        tag.getAttribute('span-id')?.split(' ').includes(span_id)
    );
}

function find_span_tags_by_start_end(docIndex, start, end) {
    /*
    this finds the span tags between start/end including start, NOT inluding end (python style)
    */
    const docDiv = document.getElementById(`doc${docIndex}`);
    return Array.from(docDiv.children).filter(tag => 
        tag.tagName === 'SPAN' && 
        parseInt(tag.getAttribute('token-id')) >= start && 
        parseInt(tag.getAttribute('token-id')) < end
    );
}

function get_current_positions(selected_span_tags, span_id) {
    /*
    Find the index of the given span_id in each selected_span_tag 'span-id' attribute
    */
    const positions = [];
    selected_span_tags.forEach(selected_span_tag => {
        const ids = selected_span_tag.getAttribute('span-id').split(' ');
        positions.push(ids.indexOf(span_id));
    });
    return positions;
}

function remove_from_span_tags(selected_span_tags, positions) {
    /*
    remove the attributes at the given positions from teh given span tags, 
    each position in positions corresponds to each span tag in selected_span_tags
    */
    selected_span_tags.forEach((selected_span_tag, index) => {
        // Extract and modify the span-id attribute
        let spanIds = selected_span_tag.getAttribute('span-id').split(' ');
        if (positions[index] < spanIds.length) {
            spanIds.splice(positions[index], 1); // Remove the id at the specified position
            if (spanIds.length === 0) 
                selected_span_tag.removeAttribute('span-id'); // Remove attribute if empty
            else 
                selected_span_tag.setAttribute('span-id', spanIds.join(' ')); // Set the modified list back as the attribute
        }
        // Extract and modify the span-type attribute
        let spanTypes = selected_span_tag.getAttribute('span-type').split(' ');
        if (positions[index] < spanTypes.length) {
            spanTypes.splice(positions[index], 1); // Remove the type at the specified position
            if (spanTypes.length === 0) 
                selected_span_tag.removeAttribute('span-type'); // Remove attribute if empty
            else 
                selected_span_tag.setAttribute('span-type', spanTypes.join(' ')); // Set the modified list back as the attribute
        }
        // Extract and modify the span-boundary attribute
        let spanBoundaries = selected_span_tag.getAttribute('span-boundary').split(' ');
        if (positions[index] < spanBoundaries.length) {
            spanBoundaries.splice(positions[index], 1); // Remove the type at the specified position
            if (spanBoundaries.length === 0) 
                selected_span_tag.removeAttribute('span-boundary'); // Remove attribute if empty
            else 
                selected_span_tag.setAttribute('span-boundary', spanBoundaries.join(' ')); // Set the modified list back as the attribute
        }
    });
}

function add_to_span_tags(selected_span_tags, new_span_id, new_type) {
    /*
    add new span_id and type to the given span tags
    the attributes for span-id, span-type and span-boundary are appended with the new values
    */
    selected_span_tags.forEach((selected_span_tag, idx) => {
        //update the span-id
        const current_span_id = selected_span_tag.getAttribute('span-id');
        if (current_span_id) 
            selected_span_tag.setAttribute('span-id', `${current_span_id} ${new_span_id}`);
        else 
            selected_span_tag.setAttribute('span-id', new_span_id);

        //update the span-type
        const current_span_type = selected_span_tag.getAttribute('span-type');
        if (current_span_type) 
            selected_span_tag.setAttribute('span-type', `${current_span_type} ${new_type}`);
        else 
            selected_span_tag.setAttribute('span-type', new_type);

        //update the span-boundary, NOTE: using relative idx here
        const current_text = selected_span_tag.getAttribute('span-boundary');
        if (current_text) 
            selected_span_tag.setAttribute('span-boundary', `${current_text} span${get_boundary_text(idx, 0, selected_span_tags.length)}`);
        else
            selected_span_tag.setAttribute('span-boundary', `span${get_boundary_text(idx, 0, selected_span_tags.length)}`);
    });
}

function get_boundary_text(token_id, start, end) {
    /*
    gets the boundary text based on the python list style start, end values and the current token_id
    all inputs should be integer
    */
    const real_start = start;
    const real_end = end - 1;
    let text = '';
    if (token_id === real_start && token_id === real_end) text ='-both';
    else if (token_id === real_start) text ='-start';
    else if (token_id === real_end) text ='-end';
    return text;
}

function remove_span_from_spans(docIndex, span_id) {
    /*
    Filter the spans array to remove the span with the matching span_id
    */
    data[docIndex].spans = data[docIndex].spans.filter(
        span => span.id !== span_id
    );
}


function remove_span_from_relations(docIndex, span_id) {
    /*
    Filter the spans array to remove the span with the matching span_id
    */
    data[docIndex].relations = data[docIndex].relations.filter(relation => 
        relation.head !== span_id && 
        relation.tail !== span_id
    );
}


function add_span_to_spans(docIndex, type, start, end, span_id, new_text) {
    // Append new span entry to the data structure
    data[docIndex].spans.push({
        id: span_id,
        type: type,
        start: start,
        end: end,
        span: new_text,
    });
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//RELATIONS
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
////////////////////////////////
//enter exit and handker code
////////////////////////////////
function enter_relation_mode(event, docIndex, mode) {
    //get the user choice and set the active_span
    show_span_menu_choose(event, 'Choose Span: ').then(span_id => {
        //set the tool state
        tool_state = mode;
        //Update the current mode display
        document.getElementById('current_mode').innerText = tool_state;
        //set the active span
        active_span = {span_id: span_id, docIndex: docIndex};
        // Update all the span border styles, will use tool_state to determine exact action
        update_border_classes(docIndex, span_id);
        //remove all open menus
        remove_menus();
    });
}


function exit_relation_mode() {
    //remove all span styles on exit
    remove_border_classes();
    //reset the active_span object
    active_span = {span_id: null, docIndex: null};
    // Update the tool_state and current mode display
    tool_state = 'span_mode';
    document.getElementById('current_mode').innerText = tool_state;
    //remove all open menus
    remove_menus();
}


function edit_relation_handler(event, docIndex) {
    let headId = active_span.span_id;
    let tailId;
    //get the user choice for the span
    show_span_menu_choose(event, 'Choose Span: ').then(selected_span_id => {
        //set the tailId
        tailId = selected_span_id;
        //validity check
        if (tailId === headId) return;
        //reverse head and tail if in rev_relation_mode 
        if (tool_state === 'rev_relation_mode') [headId, tailId] = [tailId, headId];
        //show the relation type choice menu
        show_relation_menu(event, docIndex, headId, tailId);
    });
}

////////////////////////////////
//relation menus
////////////////////////////////
function show_relation_menu(event, docIndex, headId, tailId) {
    const menu = document.createElement('div');
    menu.id = 'menu';
    menu.style.left = `${event.clientX + window.scrollX}px`;
    menu.style.top = `${event.clientY + window.scrollY}px`;

    const relationTypes = schema["relation_types"];
    relationTypes.forEach((type) => {
        const rel_exists = relation_exists(docIndex, headId, tailId, type.name);
        const text = rel_exists ? 'Rmv Relation:' : 'Add Relation:';
        const item = document.createElement('div');
        item.textContent = `${text} ${type.name}`;
        item.style.backgroundColor = type.color;
        item.style.padding = '5px';
        item.onclick = () => {
            if (rel_exists) remove_relation(docIndex, headId, tailId, type.name);
            else add_relation(docIndex, headId, tailId, type.name, type.color);
            document.body.removeChild(menu); // Close menu after selection
        };
        menu.appendChild(item);
    });
    document.body.appendChild(menu);
}

//check that a candidate relation has not already been added
function relation_exists(docIndex, headId, tailId, type) {
    return data[docIndex].relations.some(rel => 
        rel.head === headId && 
        rel.tail === tailId && 
        rel.type === type
    );
}

////////////////////////////////
//realtion add/rmv code
////////////////////////////////
function add_relation(docIndex, head_id, tail_id, type, color) {
    /* adds relation to relations and updates the border style if in rel mode */
    const rel_id = `${relation_id_prefix}${get_next_cnt(data[docIndex].relations)}`;
    //update the data structure, doesn't use the color for now, may change this later
    data[docIndex].relations.push({
        id: rel_id,
        head: head_id,
        tail: tail_id,
        type: type
    });
    if (tool_state === 'relation_mode') update_border_classes(docIndex, head_id);
    else if (tool_state === 'rev_relation_mode') update_border_classes(docIndex, tail_id);
}

function remove_relation(docIndex, head_id, tail_id, type) {
    /* this removes the given head-tail pair and type (relation) from the data[docIndex].relations list */
    //Remove the relation from the data structure
    data[docIndex].relations = data[docIndex].relations.filter(relation => 
        !(relation.head === head_id && 
        relation.tail === tail_id && 
        relation.type === type)
    );
    if (tool_state === 'relation_mode') update_border_classes(docIndex, head_id);
    else if (tool_state === 'rev_relation_mode') update_border_classes(docIndex, tail_id);   
} 

////////////////////////////////////////////
//border class updaing code for relations
//////////////////////////////////////////
function remove_border_classes() {
    /* remove all source and target border classes */
    let tags = document.querySelectorAll('div[id^="doc"] > span[class*="border"]');
    tags.forEach(tag => {
        tag.classList = "";
    });
}

function update_border_classes(docIndex, source_span_id) {
    /*
    function to update all (head)tail span styles on entering (rev_)relation mode
    it also updates the selected span border style
    */
    //remove all span styles
    remove_border_classes();
    //set the source span flashing border
    update_source_border_class(docIndex, source_span_id);
    //update the border class of the target spans
    update_target_border_classes(docIndex, source_span_id);
}


function update_source_border_class(docIndex, source_span_id) {
    /*
    update the source border class
    */
    let tags = find_span_tags_by_span_id(docIndex, source_span_id);
    tags.forEach((tag, idx) => {
        if (tag) {
            const boundary_text = get_boundary_text(idx, 0, tags.length);    //using relative token idx here
            tag.classList = "";    //reset the classList
            if (tool_state === 'relation_mode') tag.classList.add(`red-flashborder${boundary_text}`);
            else if (tool_state === 'rev_relation_mode') tag.classList.add(`black-flashborder${boundary_text}`);
        }
    });
}

function update_target_border_classes(docIndex, source_span_id) {
    /*
    update the target border classes for the targets of the given source span
    */
    data[docIndex].relations.forEach(relation => {
        if (tool_state === 'relation_mode' && relation.head === source_span_id) update_target_border_class(docIndex, relation.tail);
        else if (tool_state === 'rev_relation_mode' && relation.tail === source_span_id) update_target_border_class(docIndex, relation.head);
    });
}

function update_target_border_class(docIndex, target_span_id) {
    /*
    update the border class for the given target span id
    */
    let tags = find_span_tags_by_span_id(docIndex, target_span_id);
    tags.forEach((tag, idx) => {
        if (tag) {
            const boundary_text = get_boundary_text(idx, 0, tags.length);    //using relative token idx here
            tag.classList = "";
            if (tool_state === 'relation_mode') tag.classList.add(`tail-border${boundary_text}`);
            else if (tool_state === 'rev_relation_mode') tag.classList.add(`head-border${boundary_text}`);
        };
    });
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//IMPORT/EXPORT
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    const topInstructions = document.getElementById('topInstructions');

    fileInput.addEventListener('change', function(event) {
        //this selects the file import .json file and loads it into the associated js vars and modifies the display on screen
        //this file will have schema, raw text and spans and relations in it, the annotated data part is not used as that is just for human comprehension only
        const file = event.target.files[0];
        if (!file) return;
        filename = file.name

        reset_vars();
        const reader = new FileReader();
        reader.onload = function(e) {
            let result;
            try {
                result = JSON.parse(e.target.result);      //read in the json list to the schema object
            } 
            catch (error) {
                alert('There was an error reading your json.\nSee console.log for details...')
                console.error(`Json reading error details: ${error}`);
                return;
            } 
            finally {
                fileInput.value = '';      // Reset the file input after processing to allow reloading the same file
            }

            if (!verifyData(result, input_format)) return;    //verify format
            
            //load the data structure
            data = result.data;
            schema = result.schema;
            if (input_format === 'full') source_data = result.source_data;
            
            //update the styles from schema and display the data
            update_span_styles_from_schema();
            display_data();
        };
        reader.readAsText(file);
    });
});

function reset_vars() {
    source_data =   {};
    data = [];
    schema = {};
    document.getElementById('dataContainer').innerHTML = '';
}

function verifyData(import_data, input_format) {
    /*
    verifies all the required keys are in teh imported data
    */

    let errors = [];
    let requiredKeys = ['data', 'schema'];
    requiredKeys.forEach(key => {
        if (!(key in import_data)) {
            errors.push(`Missing primary key '${key}' in imported data`);
        }
    });

    if (!Array.isArray(import_data.data)) 
        errors.push("Provided data needs to be a list of objects.");
    else {
        requiredKeys = ['tokens', 'spans', 'relations'];
        if (input_format === 'full') 
            requiredKeys.push('source', 'offset');
        import_data.data.forEach((obs, idx) => {
            requiredKeys.forEach(key => {
                if (!(key in obs)) 
                    errors.push(`Missing key '${key}' in document: ${idx}`);
            });
        });
    }

    if (errors.length) {
        console.error(errors.join("\n"));
        return false;
    }
    return true;
}

function display_data() {
    /*
    This function loads in the data to the browser and loads the annotations if they are there.
    No need to pass data as it is a global 
    */
    const container = document.getElementById('dataContainer');
    container.innerHTML = ''; // Clear any existing contents

    // Create new contents from data
    data.forEach((obs, doc_id) => {
        //add the doc header text
        const header = document.createElement('div');
        header.id = `header${doc_id}`;
        header.textContent = `Document ID: doc${doc_id}`;
        container.appendChild(header);

        //add the docDiv
        const docDiv = document.createElement('div');
        docDiv.id = `doc${doc_id}`;
        container.appendChild(docDiv);

        //Adding tokens as individual span tags to docDiv
        obs.tokens.forEach((token, token_id) => {
            const tokenSpan = document.createElement('span');
            tokenSpan.textContent = token;
            tokenSpan.setAttribute('token-id', `${token_id}`); // Set custom attribute for token ID
            docDiv.appendChild(tokenSpan);
        });

        //update the span tags with the span annotations
        obs.spans.forEach(span => {
            update_span_tags(docDiv, span);
        });
    });
}

function update_span_tags(docDiv, span) {
    /*
    for the given span (object from data[docIndex].spans), update the attributes in the appropritate span tags
    */
    const new_span_start = parseInt(span.start);
    const new_span_end = parseInt(span.end);
    const new_span_type = span.type;
    const new_span_id = span.id;
    for (let token_id = new_span_start; token_id < new_span_end; token_id++) {    //note the new_span_end is the real end + 1
        //get the span tag to update
        const span_tag = docDiv.querySelector(`[token-id="${token_id}"]`);
        //update span tag attributes
        if (!span_tag) {
            console.log(`span_tag with token-id = ${token_id} is missing from the DOM, serious error....`)
            return;
        }
        const new_boundary_text = `span${get_boundary_text(token_id, new_span_start, new_span_end)}`;
        //read from span_tag
        let current_span_id = span_tag.getAttribute('span-id');
        let current_span_type = span_tag.getAttribute('span-type');
        let current_span_boundary = span_tag.getAttribute('span-boundary');
        //append if there already is data
        if (current_span_id) {
            current_span_id += ' ' + new_span_id;
            current_span_type += ' ' + new_span_type;
            current_span_boundary += ' ' + new_boundary_text;
        } 
        //set if there no existing data
        else {        
            current_span_id = new_span_id;
            current_span_type = new_span_type;
            current_span_boundary = new_boundary_text;
        }
        //write back to the span_tag
        span_tag.setAttribute('span-id', current_span_id);
        span_tag.setAttribute('span-type', current_span_type);
        span_tag.setAttribute('span-boundary', current_span_boundary);
    }
}

function export_data(option) {
    //option is either "view" or "export"
    //function to view the annotation data in a separate tab
    let annotated_docs = [];
    data = data.map(obs => {
        const docDiv = document.getElementById(`doc${data.indexOf(obs)}`);
        //update the span text in each span
        update_span_text(obs);
        //add annotated text and reorder keys
        //let annotated_text = convertInnerHTML(docDiv, option);
        let annotated_text = make_annotated_text(obs, option);
        return reorderObsKeys(obs, annotated_text);
    });

    const out_data = {
        data: data,
        schema: schema
    };

    if (option === "view") {
        const newWindow = window.open("", "_blank");
        newWindow.document.write(`<pre>${JSON.stringify(out_data, no_indent_specifier, 4)}</pre>`);
    }
    else if (option === "export") {
        const jsonBlob = new Blob([JSON.stringify(out_data, null, 4)], {type: 'application/json'});
        const jsonLink = document.createElement('a');
        jsonLink.download = generateTimestampedFilename(extract_file_basename(filename), 'json');
        jsonLink.href = URL.createObjectURL(jsonBlob);
        jsonLink.click();
        document.body.removeChild(jsonLink); // Clean up
    }
}

function reorderObsKeys(obs, annotated_text) {
    // This function reorders the keys of an observation to ensure a specific order
    return {
        tokens:         obs.tokens,
        annotated_text: annotated_text,
        spans:          obs.spans,
        relations:      obs.relations
    };
}

function update_span_text(obs) {
    /*
    updates the span text (in the .span key) in the spans object
    */
    obs.spans.forEach((span) => {
        const start = parseInt(span.start);   
        const end = parseInt(span.end);   
        span.span = obs.tokens.slice(start, end).join(' ').trim();
    });
}

function no_indent_specifier(key, value) {
    /*
    this ensure that the given keys do not have indents for the view operation
    */
    const no_indent_keys = ['tokens'];
    if (no_indent_keys.includes(key)) {
            // Convert array to a JSON string, then replace double quotes with single quotes
            return JSON.stringify(value).replace(/"/g, "'");
    }
    return value;
}

function extract_file_basename(path) {
    // Extract the last part of the path as the filename
    const filename = path.split('/').pop();
    // Remove the file extension
    let baseName = filename.split('.').slice(0, -1).join('.') || filename;
    // Regex to find the pattern \_annotated_d{8}-\d{6} (e.g., _annotated_12345678-123456)
    const pattern = /_annotated_\d{8}-\d{6}$/;
    // Remove the pattern from the base name if it exists
    baseName = baseName.replace(pattern, '');
    // Trim any trailing dots left after removing the pattern
    return baseName.replace(/\.$/, '');
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
    return `${baseFilename}_annotated_${timestamp}.${extension}`;
}

function find_new_ids(string1, string2) {
    if (string1 === null) string1 = '';
    if (string2 === null) string2 = '';
    const set1 = new Set(string1.trim().split(' '));
    const set2 = new Set(string2.trim().split(' '));
    // Find new members: items in set2 not in set1
    const new_ids = Array.from(set2).filter(item => !set1.has(item));
    return new_ids;
}

function find_lost_ids(string1, string2) {
    if (string1 === null) string1 = '';
    if (string2 === null) string2 = '';
    const set1 = new Set(string1.trim().split(' '));
    const set2 = new Set(string2.trim().split(' '));
    // Find removed members: items in set1 not in set2
    const lost_ids = Array.from(set1).filter(item => !set2.has(item));
    return lost_ids;
}

function make_annotated_text(obs, type) {
    /*
    utility function to add span tags around the annotated spans for the human to understand
    */
    let tokens = obs.tokens.slice(); // Create a copy of the tokens to work with
    const tags = new Array(tokens.length).fill(null); // Initialize tags array of the same length as tokens, filled with null

    //fill the tags array from obs.spans
    obs.spans.forEach(span => {
        let start = parseInt(span.start);
        let end = parseInt(span.end);
        if (type === "export") {
            tags[start] = `<${span.id}>`;
            tags[end] = `</${span.id}>`;
        } 
        else if (type === "view") {
            tags[start] = `&lt;${span.id}&gt;`;
            tags[end] = `&lt;/${span.id}&gt;`;
        }
    });

    //merge the tags with the tokens
    let shift = 0; // This will track the shift in index due to the tags being inserted
    tags.forEach((tag, idx) => {
        if (tag != null) {
            tokens.splice(idx + shift, 0, tag);
            shift += 1;
        }
    });
 
    return tokens.join(' ').trim();
}
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
//LISTENERS AND GLOBAL FUNCTIONS
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
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
    
    //close other open menus if we did not click on them
    remove_menus();
    //get target
    const target = event.target;
    //check we clicked on a span
    if (!target || target.tagName !== "SPAN" || !hasSpanId(target)) return;
    //check the parent div is an editable div
    docDiv = get_parent_div_for_mouse_event(target);
    if (!docDiv) return;
    //all good so get the docDiv object and extract the docIndex
    docIndex = parseInt(docDiv.id.replace('doc', ''));
    //these are the processing cases.....
    if (tool_state === 'span_mode') 
        edit_span_handler(event, docIndex, null, "edit");                                           //here the right-clicked span type is annotated and we are in span_mode so edit it
    else if (tool_state !== 'span_mode' && docIndex === active_span.docIndex) 
        edit_relation_handler(event, docIndex);    //here the right-clicked span type is annotated and we are in a relation_mode, so edit the relation, validity checks will be done later
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

    //close other open menus if we did not click on them
    remove_menus();

    //console.log('Left-click detected at:', event.clientX, event.clientY);
    const target = event.target;
    //check we clicked on a span tag
    if (!target || target.tagName !== "SPAN" || !hasSpanId(target)) return;
    //check the parent div is a docDiv, if not exit
    let docDiv = get_parent_div_for_mouse_event(target);
    if (!docDiv) return;
    //all good so extract the docIndex
    let docIndex = parseInt(docDiv.id.replace('doc', ''));

    //these are the processing cases.....
    if (left_click) enter_relation_mode(event, docIndex, 'relation_mode');                  //if plain left click enter relation mode if not already there and choose a new head
    else if (ctrl_left_click) enter_relation_mode(event, docIndex, 'rev_relation_mode');    //go to rev-rel mode on ctrl-left clikc on a valid tail
});

//add the exit all relation modes on esc button press event
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape'){
        remove_menus();
        clearSelection();
        if (tool_state === 'relation_mode' || tool_state === 'rev_relation_mode') 
            exit_relation_mode();
    }
});

document.getElementById('dataContainer').addEventListener('dblclick', function(event) {
    event.preventDefault();
    event.stopImmediatePropagation(); // Optional: stop further event propagation
    clearSelection(); // Function call to clear any current text selections
    doubleclick_flag = true;
    return;
});

document.getElementById('dataContainer').addEventListener('mousedown', function(event) {
    //disable doubleclick
    if (doubleclick_flag) {
        event.preventDefault();
        event.stopImmediatePropagation(); // Optional: stop further event propagation
        clearSelection(); // Function call to clear any current text selections
        return; 
    }

    // Disable default Ctrl+Click and Shift+Click behaviors
    if (event.ctrlKey || event.shiftKey) {
        event.preventDefault(); // Prevent the default browser action
        event.stopPropagation(); // Stop the event from propagating further
    }

    let target = event.target;
    // Check if the tool state is span_mode
    if (tool_state !== "span_mode") return;
    // Check if the event is happening in a span
    if (target.tagName !== "SPAN") return; 
    //check if the parent div is valid
    let docDiv = get_parent_div_for_mouse_event(target);
    if (!docDiv) return;
    //all good so get the doc index
    mouseDownDocIndex = parseInt(docDiv.id.replace('doc', ''));
});

document.getElementById('dataContainer').addEventListener('mouseup', function(event) {
    //disable doubleclick
    if (doubleclick_flag) {
        doubleclick_flag = false;
        event.preventDefault();
        event.stopImmediatePropagation(); // Optional: stop further event propagation
        clearSelection(); // Function call to clear any current text selections
        return; 
    }

    // Capture the mouseDownDocIndex to use within the timeout
    let capturedMouseDownDocIndex = mouseDownDocIndex;
    // Reset the global mousedown index as we have an upmouse
    mouseDownDocIndex = null;

    let target = event.target;    //target is the span that the mouse was released on
    // Check if the tool state is span_mode
    if (tool_state !== "span_mode") return;
    //check if the parent div is valid
    let docDiv = get_parent_div_for_mouse_event(target);
    if (!docDiv) return;
    //all good so extract the docIndex
    const docIndex = parseInt(docDiv.id.replace('doc', ''));
    const docDiv_id = `doc${docIndex}`
    //check if mousedown and mouseup are in the same document and if the selection represents an actual drag (non-zero length)
    if (capturedMouseDownDocIndex !== docIndex) return;

    // Timeout to handle mouse up processing
    setTimeout(() => {
        const selection = window.getSelection();
        // Check if the selection range has more than zero length (implies dragging)
        if (selection.isCollapsed || selection.toString().trim().length === 0) return;
        // Check if selection spans multiple nodes and gather all spans involved
        let range = selection.getRangeAt(0);
        let startContainer = range.startContainer;
        let endContainer = range.endContainer;

        //Check if the start/end containers are text nodes and assign their parent, otherwise assign the container itself
        let start = -1;
        if (startContainer.nodeType === Node.TEXT_NODE) start = startContainer.parentNode.getAttribute('token-id');
        else start = startContainer.getAttribute('token-id');
        let end = -1;
        if (endContainer.nodeType === Node.TEXT_NODE) end = endContainer.parentNode.getAttribute('token-id');
        else end = endContainer.getAttribute('token-id');
        start = parseInt(start);
        end = parseInt(end) + 1;   //to make it python list style
        // Extract all spans between start and end spans if they are different
        let spans = find_span_tags_by_start_end(docIndex, start, end);
        //process the selected tokens
        edit_span_handler(event, docIndex, spans, "add");
        // Clear the selection after processing
        selection.removeAllRanges();
    }, 50);    //50ms timeout
});

document.getElementById('dataContainer').addEventListener('mouseover', function(event) {
    const target = event.target;
    // Check if the hovered element is an annotated span
    if (target.tagName === 'SPAN' && hasSpanId(target)) {
        tooltip_info.style.display = 'block';
        //Position the tooltip near the mouse cursor
        tooltip_info.style.left = `${event.clientX + window.scrollX - info_offset_x}px`;
        tooltip_info.style.top = `${event.clientY + window.scrollY + info_offset_y}px`;
        //Display the tooltip with the content
        tooltip_info.innerHTML = get_info_text(target, true);
    }
});

document.getElementById('dataContainer').addEventListener('mousemove', function(event) {
    // Move the tooltip along with the mouse
    if (tooltip_info.style.display === 'block') {
        tooltip_info.style.left = `${event.clientX + window.scrollX - info_offset_x}px`;
        tooltip_info.style.top = `${event.clientY + window.scrollY + info_offset_y}px`;
    }
});

document.getElementById('dataContainer').addEventListener('mouseout', function(event) {
    const target = event.target;
    // Hide the tooltip when the mouse leaves an annotated span
    if (target.tagName === 'SPAN' && hasSpanId(target)) {
        tooltip_info.style.display = 'none';
    }
});

// Utility function to clear text selection
function clearSelection() {
    if (window.getSelection) {
        if (window.getSelection().empty) window.getSelection().empty();   //chrome
        else if (window.getSelection().removeAllRanges) window.getSelection().removeAllRanges();  //firefox
    } 
    else if (document.selection) document.selection.empty();   //IE
}

//helper function for info display
function get_info_text(target, split=true) {
    const id_text = target.getAttribute('span-id');
    const type_text = target.getAttribute('span-type');
    let info_text = `${id_text} (${type_text})`;
    if (split===true) {
        // Split the values by spaces
        const spanIds = id_text.split(" ");
        const spanTypes = type_text.split(" ");
        // Create a string that combines each span-id with its corresponding span-type
        info_text = spanIds.map((id, index) => {
            const type = spanTypes[index] || 'not-found'; 
            return `${id} (${type})`;
        }).join("<br>");
    }
    return info_text
}
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////