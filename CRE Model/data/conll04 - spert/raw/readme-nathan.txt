ok, so this dataset format is the same as that used for maintie, python based numbering

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


As with the original dataset, the types are limited, that is how they get such good scores


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
I think they have filtered out to only include observations of relevance, I think this is a cop out, but hey...

Bert large
--------------------
An entity is considered correct if the entity type and span is predicted correctly

                type    precision       recall     f1-score      support
               Other        75.26        61.86        67.91          118
                 Org        82.76        84.71        83.72          170
                Peop        94.44        96.11        95.27          283
                 Loc        89.29        93.17        91.19          322

               micro        88.16        88.35        88.26          893
               macro        85.44        83.96        84.52          893

--- Relations ---

Without named entity classification (NEC)
A relation is considered correct if the relation type and the spans of the two related entities are predicted correctly (entity type is not considered)

                type    precision       recall     f1-score      support
               OrgBI        64.41        50.00        56.30           76
                Kill        75.47        95.24        84.21           42
               LocIn        69.33        80.00        74.29           65
                Live        62.86        72.53        67.35           91
                Work        69.44        72.46        70.92           69

               micro        67.58        71.72        69.59          343
               macro        68.30        74.05        70.61          343

With named entity classification (NEC)
A relation is considered correct if the relation type and the two related entities are predicted correctly (in span and entity type)

                type    precision       recall     f1-score      support
               OrgBI        64.41        50.00        56.30           76
                Kill        75.47        95.24        84.21           42
               LocIn        69.33        80.00        74.29           65
                Live        62.86        72.53        67.35           91
                Work        69.44        72.46        70.92           69

               micro        67.58        71.72        69.59          343
               macro        68.30        74.05        70.61          343
               
               
Bert base
-----------------------------
An entity is considered correct if the entity type and span is predicted correctly

                type    precision       recall     f1-score      support
                Peop        92.49        95.76        94.10          283
                 Loc        89.36        91.30        90.32          322
               Other        64.29        61.02        62.61          118
                 Org        83.24        84.71        83.97          170

               micro        86.11        87.46        86.78          893
               macro        82.34        83.20        82.75          893

--- Relations ---

Without named entity classification (NEC)
A relation is considered correct if the relation type and the spans of the two related entities are predicted correctly (entity type is not considered)

                type    precision       recall     f1-score      support
               OrgBI        64.71        57.89        61.11           76
                Kill        69.09        90.48        78.35           42
               LocIn        66.22        75.38        70.50           65
                Live        63.92        68.13        65.96           91
                Work        69.57        69.57        69.57           69

               micro        66.39        70.26        68.27          343
               macro        66.70        72.29        69.10          343

With named entity classification (NEC)
A relation is considered correct if the relation type and the two related entities are predicted correctly (in span and entity type)

                type    precision       recall     f1-score      support
               OrgBI        64.71        57.89        61.11           76
                Kill        69.09        90.48        78.35           42
               LocIn        64.86        73.85        69.06           65
                Live        63.92        68.13        65.96           91
                Work        69.57        69.57        69.57           69

               micro        66.12        69.97        67.99          343
               macro        66.43        71.98        68.81          343
               
               
               
               
Deberta Base
-----------------------------
An entity is considered correct if the entity type and span is predicted correctly

                type    precision       recall     f1-score      support
               Other        61.54        54.24        57.66          118
                 Org        83.23        75.88        79.38          170
                 Loc        86.47        91.30        88.82          322
                Peop        93.49        96.47        94.96          283

               micro        85.30        85.11        85.20          893
               macro        81.18        79.47        80.21          893

--- Relations ---

Without named entity classification (NEC)
A relation is considered correct if the relation type and the spans of the two related entities are predicted correctly (entity type is not considered)

                type    precision       recall     f1-score      support
                Kill        73.58        92.86        82.11           42
               OrgBI        55.84        56.58        56.21           76
               LocIn        59.49        72.31        65.28           65
                Live        58.88        69.23        63.64           91
                Work        64.52        57.97        61.07           69

               micro        61.38        67.64        64.36          343
               macro        62.46        69.79        65.66          343

With named entity classification (NEC)
A relation is considered correct if the relation type and the two related entities are predicted correctly (in span and entity type)

                type    precision       recall     f1-score      support
                Kill        73.58        92.86        82.11           42
               OrgBI        55.84        56.58        56.21           76
               LocIn        59.49        72.31        65.28           65
                Live        58.88        69.23        63.64           91
                Work        64.52        57.97        61.07           69

               micro        61.38        67.64        64.36          343
               macro        62.46        69.79        65.66          343







Spanbert-large
-----------------------------
An entity is considered correct if the entity type and span is predicted correctly

                type    precision       recall     f1-score      support
                Peop        93.79        96.11        94.94          283
                 Org        77.66        85.88        81.56          170
               Other        53.33        47.46        50.22          118
                 Loc        85.10        92.24        88.52          322

               micro        82.73        86.34        84.49          893
               macro        77.47        80.42        78.81          893

--- Relations ---

Without named entity classification (NEC)
A relation is considered correct if the relation type and the spans of the two related entities are predicted correctly (entity type is not considered)

                type    precision       recall     f1-score      support
                Kill        77.08        88.10        82.22           42
               OrgBI        53.62        48.68        51.03           76
                Work        62.67        68.12        65.28           69
                Live        57.43        63.74        60.42           91
               LocIn        58.54        73.85        65.31           65

               micro        60.53        66.18        63.23          343
               macro        61.87        68.50        64.85          343

With named entity classification (NEC)
A relation is considered correct if the relation type and the two related entities are predicted correctly (in span and entity type)

                type    precision       recall     f1-score      support
                Kill        77.08        88.10        82.22           42
               OrgBI        53.62        48.68        51.03           76
                Work        62.67        68.12        65.28           69
                Live        57.43        63.74        60.42           91
               LocIn        58.54        73.85        65.31           65

               micro        60.53        66.18        63.23          343
               macro        61.87        68.50        64.85          343





bert-base-NER
-----------------------------
An entity is considered correct if the entity type and span is predicted correctly

                type    precision       recall     f1-score      support
                Peop        93.71        94.70        94.20          283
                 Org        80.00        77.65        78.81          170
                 Loc        90.97        90.68        90.82          322
               Other        65.09        58.47        61.61          118

               micro        86.67        85.22        85.94          893
               macro        82.44        80.38        81.36          893

--- Relations ---

Without named entity classification (NEC)
A relation is considered correct if the relation type and the spans of the two related entities are predicted correctly (entity type is not considered)

                type    precision       recall     f1-score      support
               LocIn        79.03        75.38        77.17           65
               OrgBI        65.00        51.32        57.35           76
                Work        60.00        60.87        60.43           69
                Live        62.37        63.74        63.04           91
                Kill        71.15        88.10        78.72           42

               micro        66.77        65.60        66.18          343
               macro        67.51        67.88        67.34          343

With named entity classification (NEC)
A relation is considered correct if the relation type and the two related entities are predicted correctly (in span and entity type)

                type    precision       recall     f1-score      support
               LocIn        79.03        75.38        77.17           65
               OrgBI        65.00        51.32        57.35           76
                Work        60.00        60.87        60.43           69
                Live        62.37        63.74        63.04           91
                Kill        71.15        88.10        78.72           42

               micro        66.77        65.60        66.18          343
               macro        67.51        67.88        67.34          343    