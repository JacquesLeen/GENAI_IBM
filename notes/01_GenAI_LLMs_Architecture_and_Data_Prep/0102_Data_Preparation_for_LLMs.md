# Data Preparation for LLMs

## Tokenization

Tokenization is the procedure that allows LLM models to identify and extract specific keywords by breaking down the prompt. The piece of software that brakes down text into individual tokens is called a tokenizer. There are 3 main strategize to tokenize

* **Word based** : every word of the prompt becomes a token, this preserves semantic meaning, but on the other side increases the dimension of the vocabulary

```python
import spacy
text = "Unicorns are real, I have seen some unicorns"
nlp = spacy.load("en_core_web_sm")
doc= nlp(text)
token_list =[token.text for token in doc]
print(token_list)
```

returns

```bash
['Unicorns', 'are' 'real', ',', 'I', 'have' 'seen' 'some' 'unicorns']
```

does not distinguish between ```Unicorns``` and ```unicorns``` which might be problematic

* **Char Based**: every char is a token, you loose some of the meaning, but the vocabulary is way smaller

* **Subword based**: frequently used word stay unsplit whereas non frequent word are separated. example algorithms for subword base tokenization are the Word Piece, Unigram or Sentence Piece algorithms

```python
#wordpiece example
from transformer import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.tokenize("I learned Tokenization")
```

returns

```bash
['I', 'learned', 'Token', '##ization']
```

```python
#unigram and sentence piece
from transformer import XLNetTokenizer
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
tokenizer.tokenize("I learned Tokenization")
```

returns

```bash
['_I', '_learned', '_Token', 'ization']
```

In ```pytorch``` you can use the ```torchtext``` library to perform tokenization and the ```build_library_from_iterator``` function to create a vocabulary and assign a unique index to each token

```python
dataset =[
    1, "Intro to NLP"),
    2, "Basics of Pytorch")
]

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer("basic_english")
tokenizer(dataset[0][1])
```

returns

```
['introduction', 'to', 'nlp']
```

you can now use an iterator

```python
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

my_iterator = yield_tokens(dataset)

next(my_iterator)
next(my_iterator)
```

returns

```
['introduction', 'to', 'nlp']
['basics', 'of', 'pytorch']
```

now you can call the ```build_vocab_from_iterator``` allows you to build a vocabulary with unique index for the tokens you passed

A sleek way to work with pytorch and tokenizers within is to build a function

```python
from torchtext import vocab

def get_tokenized_sentence_and_indices(iterator):
    tokenized_sentence = next(iterator)
    token_indices =[vocab[token] for token in tokenized_sentence]

    return tokenized_sentence, token_indices

tokenized_sentence, token_indices = get_tokenized_sentence_and_indices(my_iterator)
```

will generate two objects

```bash
['introduction', 'to', 'nlp']
[1,2,3]
```

where the second list are the unique indices

## Data Loaders

A Data Loader is a tool that facilitates the process of prepare and load data for the training of GenAI Models facilitating batching and shuffling of data, memory optimization, integration with other libraries and or training pipelines and so on.

In the Pytorch framework we work with datasets: in general we want to separate training, test and validation sets. By using dataloaders we can load data in batches rather than one at the time, this allows shuffling as well before batching data. Shuffling is particularly important when working with Deep Learning Models as we want to avoid the model to learn patterns based on order.

## Data Quality and Diversity for Training

data quality refers to the accuracy and consistency of the data used to train the model. techniques in that area are:

* Noise reduction: removal of noisy data is useful to have the model learn relevant patterns
* Consistency checks: prevent the model to learn from false or outdated data
* Labeling quality: prevent the model to learn from mislabeled data

Diverse representation is also relevant when training a model, accounting for diversified demographics, balanced data sources and variety is crucial in reducing bias.
