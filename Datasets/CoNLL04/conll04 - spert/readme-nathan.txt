an observation is a dict, 
the tokens key has a list of the tokens, which are word tokens
the entities key has a list of dicts, one for each entity, 
	each entity dict has the type, the start token idx and end token idx, so the actual entity span is start to end-1
the relations key has a list of relation dicts
	each relation dict has the type, teh head idx (from the entity list) and the tail idx (from the entity list)

The entity types and relation types are the keys found in conll04_types.json

Types are formatted here:
{
"entities": {
	"Loc": {"short": "Loc", "verbose": "Location"}, 
	"Org": {"short": "Org", "verbose": "Organization"}, 
	"Peop": {"short": "Peop", "verbose":"People"}, 
	"Other": {"short": "Other", "verbose": "Other"}
	}, 
"relations": {
	"Work_For": {"short": "Work", "verbose": "Work for", "symmetric": false}, 
	"Kill": {"short": "Kill", "verbose": "Kill", "symmetric": false}, 
	"OrgBased_In": {"short": "OrgBI", "verbose": "Organization based in", "symmetric": false}, 
	"Live_In": {"short": "Live", "verbose": "Live in", "symmetric": false}, 
	"Located_In": {"short": "LocIn", "verbose": "Located in", "symmetric": false}
	}
}


As with the original dataset, the types are limited.


NOTE: this is an annotated dataset

The data splits are:
dev => 		231 obs
test => 	288 obs
train = 	922 obs
train_dev =>	1153 obs 

the file: conll04_prediction_example.json is just showing you the 3 ways you can input an observation to the model for prediction, 
ie. it accepts any of:
1) a dict with the key 'tokens' and a list of word tokens
2) a list of word tokens
3) a simple string


NOTE:
The original conll04 dataset seems to have 5925 obs, the spert processed version seems to hawe 1441 (922 train + 231 dev + 288 test)
I think they have filtered out to only include observations of relevance.
