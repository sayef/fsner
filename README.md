<p align="center"> <img src="http://sayef.tech:8082/uploads/FSNER-LOGO-2.png" alt="FSNER LOGO"> </p>

<p align="center">
  Implemented by <a href="https://huggingface.co/sayef"> sayef </a>. 
</p>

> UPDATES
> 1. Training script is now available.
> 2. Pairwise query and support examples are not required anymore. Please look into example usage for details.
> 3. Added [sample dataset](https://github.com/sayef/fsner/blob/master/scripts/sample_dataset.json) and links to converted ontonotes5 training and validation dataset (please see dataset preparation section below).

## Overview

The FSNER model was proposed in [Example-Based Named Entity Recognition](https://arxiv.org/abs/2008.10570) by Morteza
Ziyadi, Yuting Sun, Abhishek Goswami, Jade Huang, Weizhu Chen. To identify entity spans in a new domain, it uses a
train-free few-shot learning approach inspired by question-answering.

## Abstract

> We present a novel approach to named entity recognition (NER) in the presence of scarce data that we call example-based NER. Our train-free few-shot learning approach takes inspiration from question-answering to identify entity spans in a new and unseen domain. In comparison with the current state-of-the-art, the proposed method performs significantly better, especially when using a low number of support examples.

## Model Training Details

| identifier        | epochs |                                            datasets                                             |
| ---------- |:------:|:-----------------------------------------------------------------------------------------------:|
| [sayef/fsner-bert-base-uncased](https://huggingface.co/sayef/fsner-bert-base-uncased)      |   25   |  ontonotes5, conll2003, wnut2017, mit_movie_trivia, mit_restaurant and fin (Alvarado et al.).   |

## Installation and Example Usage

You can use the FSNER model in 3 ways:

1. Install directly from PyPI: `pip install fsner` and import the model as shown in the code example below

   or

2. Install from source: `python install .` and import the model as shown in the code example below

   or

3. Clone [repo](https://github.com/sayef/fsner) and add absolute path of `fsner/src` directory to your PYTHONPATH and
   import the model as shown in the code example below

```python
import json

from fsner import FSNERModel, FSNERTokenizerUtils, pretty_embed

query_texts = [
    "Does Luke's serve lunch?",
    "Chang does not speak Taiwanese very well.",
    "I like Berlin."
]

# Each list in supports are the examples of one entity type
# Wrap entities around with [E] and [/E] in the examples.
# Each sentence should have only one pair of [E] ... [/E]

support_texts = {
    "Restaurant": [
        "What time does [E] Subway [/E] open for breakfast?",
        "Is there a [E] China Garden [/E] restaurant in newark?",
        "Does [E] Le Cirque [/E] have valet parking?",
        "Is there a [E] McDonalds [/E] on main street?",
        "Does [E] Mike's Diner [/E] offer huge portions and outdoor dining?"
    ],
    "Language": [
        "Although I understood no [E] French [/E] in those days , I was prepared to spend the whole day with Chien - chien .",
        "like what the hell 's that called in [E] English [/E] ? I have to register to be here like since I 'm a foreigner .",
        "So , I 'm also working on an [E] English [/E] degree because that 's my real interest .",
        "Al - Jazeera TV station , established in November 1996 in Qatar , is an [E] Arabic - language [/E] news TV station broadcasting global news and reports nonstop around the clock .",
        "They think it 's far better for their children to be here improving their [E] English [/E] than sitting at home in front of a TV . \"",
        "The only solution seemed to be to have her learn [E] French [/E] .",
        "I have to read sixty pages of [E] Russian [/E] today ."
    ]
}

device = 'cpu'

tokenizer = FSNERTokenizerUtils("sayef/fsner-bert-base-uncased")
queries = tokenizer.tokenize(query_texts).to(device)
supports = tokenizer.tokenize(list(support_texts.values())).to(device)

model = FSNERModel("sayef/fsner-bert-base-uncased")
model.to(device)

p_starts, p_ends = model.predict(queries, supports)

# One can prepare supports once and reuse  multiple times with different queries
# ------------------------------------------------------------------------------
# start_token_embeddings, end_token_embeddings = model.prepare_supports(supports)
# p_starts, p_ends = model.predict(queries, start_token_embeddings=start_token_embeddings,
#                                  end_token_embeddings=end_token_embeddings)

output = tokenizer.extract_entity_from_scores(query_texts, queries, p_starts, p_ends,
                                              entity_keys=list(support_texts.keys()), thresh=0.50)

print(json.dumps(output, indent=2))

# install displacy for pretty embed
pretty_embed(query_texts, output, list(support_texts.keys()))
```

<p align="center"> <img src="http://sayef.tech/uploads/FSNER-OUTPUT.png" alt="FSNER OUTPUT"> </p>

## Datasets preparation

1. We need to convert dataset into the following format. Let's say we have a dataset file train.json like following.
2. Each list in supports are the examples of one entity type
3. Wrap entities around with [E] and [/E] in the examples.
4. Each example should have only one pair of [E] ... [/E].

```json
{
  "CARDINAL_NUMBER": [
    "Washington , cloudy , [E] 2 [/E] to 6 degrees .",
    "New Dehli , sunny , [E] 6 [/E] to 19 degrees .",
    "Well this is number [E] two [/E] .",
    "....."
  ],
  "LANGUAGE": [
    "They do n't have the Quicken [E] Dutch [/E] version ?",
    "they learned a lot of [E] German [/E] .",
    "and then [E] Dutch [/E] it 's Mifrau",
    "...."
  ],
  "MONEY": [
    "Per capita personal income ranged from $ [E] 11,116 [/E] in Mississippi to $ 23,059 in Connecticut ... .",
    "The trade surplus was [E] 582 million US dollars [/E] .",
    "It settled with a loss of 4.95 cents at $ [E] 1.3210 [/E] a pound .",
    "...."
  ]
}
```

2. Converted ontonotes5 dataset can be found here:
    1. [train](https://gist.githubusercontent.com/sayef/46deaf7e6c6e1410b430ddc8aff9c557/raw/ea7ae2ae933bfc9c0daac1aa52a9dc093d5b36f4/ontonotes5.train.json)
    2. [dev](https://gist.githubusercontent.com/sayef/46deaf7e6c6e1410b430ddc8aff9c557/raw/ea7ae2ae933bfc9c0daac1aa52a9dc093d5b36f4/ontonotes5.dev.json)

3. Then trainer script can be used to train/evaluate your fsner model.

```bash
fsner trainer --pretrained-model bert-base-uncased --mode train --train-data train.json --val-data val.json \
                --train-batch-size 6 --val-batch-size 6 --n-examples-per-entity 10 --neg-example-batch-ratio 1/3 --max-epochs 25 --device gpu \
                --gpus -1 --strategy ddp
```
