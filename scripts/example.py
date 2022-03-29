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
                                              entity_keys=list(support_texts.keys()), thresh=0.80)

print(json.dumps(output, indent=2))

try:
    import spacy

    with open("result.html", "w") as f:
        f.write(pretty_embed(query_texts, output, list(support_texts.keys())))
except ImportError:
    print("Install spacy to output pretty embedding!")
