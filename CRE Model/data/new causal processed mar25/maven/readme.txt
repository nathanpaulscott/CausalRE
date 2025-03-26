make a tool to convert this to/from the annotation tool format




to annotation tool
----------------------
read json
schema is unchanged
get the required splits ['train'. 'val', 'test'] modify if required

merge the splits, add a key n each obs indicating the split => call it 'split': 'train' etc...

then add unqiue ids to spans
i.e. id: span list idx
add unique ids to rels
i.e id: rel list idx
NOTE: if you just use the span list idx, you will not have to modify the head and tail values