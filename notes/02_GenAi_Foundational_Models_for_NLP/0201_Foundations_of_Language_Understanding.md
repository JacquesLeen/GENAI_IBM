# Foundations of Language Understanding

## Converting Words into Features

### One Hot Encoding

Imagine to have the following dataset

```python
data=[
    "I like dogs",
    "I dont like cats",
    "I am neutral towards hippos"
]
```
and wanting to parse it with a ML Model for some scope. Computers work with numbers not words so this dictionary has to be converted to a set of numbers. The idea of one hot encoding
is to encode every word in a dictionary with a vector of booleans. The encoding of the word "I" for example could be something like that $[1,0,0,0,0,0,0,0,0]$

|Word|Value|
|----|-----|
| I | 1 |
|like| 0 |
|dogs| 0 |
|dont| 0 |
|cats|0|
|am| 0 |
|neutral| 0 |
|towards| 0 |
|hippos| 0 |

whereas for exaple "neutral" will be $[0,0,0,0,0,0,1,0,0]$

### Bag of Words

Bag of Words allows then to combine words from the dict into a vector for the sentence

 - I $[1,0,0,0,0,0,0,0,0]$
 - Like $[0,1,0,0,0,0,0,0,0]$
 - Cats $[0,0,0,0,1,0,0,0,0]$
 - I like cats $[1,1,0,0,1,0,0,0,0]$