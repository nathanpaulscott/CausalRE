the input format is a dict with keys schema and data
data is a list of obs, each obs has 3 keys with a list in each: "tokens", "spans", "relations"
The tokens is a list of word tokens
The spans is a list of span dicts, each span dict has the unique span id, the start and end (these are the word token id in tokens, end is actual+1), and the span type
The relations is a list of rel dicts, each rel dict has the unique rel id, the head span id and end span id (these are the unique span ids from teh spans list), and the rel type

NOTE: this format uses unique span ids and unique rel ids, which is different to the format where the rel hea dnad tail use the span idx in the spans list, however this conversion is trivial

{
    'schema':{
        'span_types':[{'name':'type name', 'color': 'rgba(x,x,x,0.3)'}, ...],
        'relation_types':[{'name':'type name', 'color': 'rgba(x,x,x,0.3)'}, ...]
     },
     'data': [
         {'tokens': ['word token 0', 'word token 1',....],
          'spans':      [{'id': 'span id', 'start': x, 'end': y, 'type': 'type name'}, {...}, ...],
          'relations':  [{'id': 'rel id', 'head': 'head span id', 'tail': 'tail span id', 'type': 'type name'}, {...}, ...]},
         {...},
         {...}
      ]
}
