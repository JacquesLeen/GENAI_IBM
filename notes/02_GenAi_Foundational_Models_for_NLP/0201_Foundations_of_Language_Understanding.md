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
is to encode every word in a dictionary with a vector of booleans. The encoding of the word "I" for example could be something like that $[1,0,0,0,0,0,0,0,0]$.

| Word    | Value |
| ------- | ----- |
| I       | 1     |
| like    | 0     |
| dogs    | 0     |
| dont    | 0     |
| cats    | 0     |
| am      | 0     |
| neutral | 0     |
| towards | 0     |
| hippos  | 0     |

In this case the vocabulary is small and there is no need for pretokenization, sometimes thisn might be useful in that we can recognize similarities between words and reduce them to a token.
whereas for exaple "neutral" will be $[0,0,0,0,0,0,1,0,0]$.

We can now

- I $[1,0,0,0,0,0,0,0,0]$
- Like $[0,1,0,0,0,0,0,0,0]$
- Cats $[0,0,0,0,1,0,0,0,0]$
- I like cats

$$
\begin{bmatrix}
     1,0,0,0,0,0,0,0,0\\
     0,1,0,0,0,0,0,0,0\\
     0,0,0,0,1,0,0,0,0
\end{bmatrix}
$$

This approach presents advantages but also some disadvantages

- No relationship between words is conveyed
- Sparse Matrix which is bad for ML training and might generate overfitting
- Big vocabulary is very hard to work with

### Bag of Words

Bag of Words allows then to combine words from the dict into a vector for the sentence. if you have already an encoding for the vocabulary you can count how many occurrences are there of each word in the sentence. For example the sentence "another dog is chasing my dog" in a vocabulary that looks like

| Word    | Count |
| ------- | ----- |
| another | 1     |
| person  | 0     |
| dog     | 2     |
| banana  | 0     |
| chase   | 1     |
| eat     | 0     |

can be represented by vector $[1,0,2,0,1,0]$. This is already an improvement compared to OHE because it captures some similarity between text, i.e. the occurrences of words, and generates a less sparse matrix. Tho the resulting vectors are nevertheless sparse and have lots of dimensions. Furthermore it does not account for order. "A dog is chasing a person" and "a person is chasing a dog" will have the same representation, which is problematic.

### Embedding

Embedding is a way to encode words as numbers but still preserve meaning. The idea behind embedding is that many words have similarities in terms of what they mean

- King - Queen
- Man - Woman

In this case the relationship that is between King and Queen is kind of the same as Man and Woman. In a sense 

$$
\text{King - Man + Woman = Queen}
$$

which is the same as representing them as vectors 
![alt text](03438.jpg)

If vectors are close then the words have similar meaning, otherwise they will sit far away from each other. each dimension capture a feature of the word. And with many dimensions involved one can capture word relationships far better then in the previously seen cases. Embeddings is a technique of distributed representation in that it distributes the meaning of a word across different dimensions. A potential representatioon of the example that we have just seen could be

- King = $[0.12, 0.6, 0.43, 0.76]$
- Queen = $[0.34, 0.58, 0.46, 0.8]$
- Man = $[0.15, 0.97, 0.21, 0.18]$
- Woman = $[0.15, 0.97, 0.21, 0.18]$

there are different techniques and algorithm for embeddings such as Word2Vec GloVe or FastText