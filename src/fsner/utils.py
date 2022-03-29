import json


def start_and_end_tokens_exist(x, tokenizer):
    input_ids = tokenizer.tokenize(x)["input_ids"]
    return tokenizer.entity_start_token_id in input_ids and tokenizer.entity_end_token_id in input_ids


def load_dataset(dataset_path, tokenizer):
    """
    Loads dataset from json file and filters out the texts that are longer than defined max token length in tokenizer
    :param dataset_path: json file path
    :param tokenizer: instance of TokenizerUtils class
    :return:
    """
    with open(dataset_path, encoding="utf-8") as json_file:
        data = json.load(json_file)

    for entity, examples in data.items():
        updated_examples = []
        for x in examples:
            if start_and_end_tokens_exist(x, tokenizer):
                updated_examples.append(x)

        data[entity] = updated_examples

    return data


html_template = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>displaCy</title>
    </head>

    <body style="font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: ltr">
<figure style="margin-bottom: 6rem">
<div class="entities" style="line-height: 2.5; direction: ltr">
%s
 </div>
</figure>
</body>
</html>
"""

colors = {
    "0": "#7aecec",
    "1": "#bfeeb7",
    "2": "#feca74",
    "3": "#ff9561",
    "4": "#aa9cfc",
    "5": "#c887fb",
    "6": "#9cc9cc",
    "7": "#ffeb80",
    "8": "#ff8197",
    "9": "#ff8197",
    "10": "#f0d0ff",
    "11": "#bfe1d9",
    "12": "#bfe1d9",
    "13": "#e4e7d2",
    "14": "#e4e7d2",
    "15": "#e4e7d2",
    "16": "#e4e7d2",
    "17": "#e4e7d2",
}


def pretty_embed(query_texts, output_dict, entity_keys=colors.keys()):
    """
    Generates pretty embedding of the predicted entities using spacy library
    :param query_texts: Original query texts as list
    :param output_dict: List of list of dicts, where each dict corresponds one entity prediction
    :param entity_keys: List of entity keys
    :return: displacy generated html
    """
    new_colors = {}
    for idx, entity_key in enumerate(entity_keys):
        new_colors[entity_key.upper()] = colors[str(idx)]

    from spacy import displacy

    html = ""
    for text, dict_list in zip(query_texts, output_dict):
        html += "\n" + displacy.render({"text": text, "ents": dict_list, "title": None}, manual=True, style="ent",
                                       options={"colors": new_colors})
    return html_template % html
