import torch
import transformers.tokenization_utils_base
from transformers import AutoTokenizer
from typing import Union, List


class FSNERTokenizerUtils(object):
    def __init__(self, pretrained_model_name_or_path, entity_start_token="[E]", entity_end_token="[/E]"):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.entity_start_token = entity_start_token
        self.entity_end_token = entity_end_token
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': [self.entity_start_token, self.entity_end_token]})
        self.entity_start_token_id = self.tokenizer.convert_tokens_to_ids(self.entity_start_token)
        self.entity_end_token_id = self.tokenizer.convert_tokens_to_ids(self.entity_end_token)

    def tokenize(self, x: Union[str, List[str], List[List[str]]]) -> transformers.tokenization_utils_base.BatchEncoding:
        """
        Wrapper function for tokenizing queries and supports
        :param x (`str, List[str] or List[List[str]]`): Single query string or list of strings for queries or list of lists of strings for supports.
        :return `transformers.tokenization_utils_base.BatchEncoding` dict with additional keys and values for start_token_id, end_token_id and sizes of example lists for each entity type
        """
        if isinstance(x, str):
            return self.tokenizer(
                x,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
        elif isinstance(x, list) and all([isinstance(_x, list) for _x in x]):
            d = None
            for l in x:
                t = self.tokenizer(
                    l,
                    padding="max_length",
                    max_length=512,
                    truncation=True,
                    return_tensors="pt",
                )
                t["sizes"] = torch.tensor([len(l)])
                if d is not None:
                    for k in d.keys():
                        d[k] = torch.cat((d[k], t[k]), 0)
                else:
                    d = t

            d["start_token_id"] = torch.tensor(self.entity_start_token_id)
            d["end_token_id"] = torch.tensor(self.entity_end_token_id)

        elif isinstance(x, list) and all([isinstance(_x, str) for _x in x]):
            d = self.tokenizer(
                x,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=True
            )

        else:
            raise Exception(
                "Type of parameter x was not recognized! Only `list of strings` for query or `list of lists of strings` for supports are supported."
            )

        return d

    def _decode(self, offset_map, text, start_idx, end_idx):
        s = offset_map[start_idx:end_idx][0][0].item()
        e = offset_map[start_idx:end_idx][-1][-1].item()
        return s, e, text[s:e]

    def extract_entity_from_scores(self, query_texts, queries, p_starts, p_ends, entity_keys=None, thresh=0.45):
        """ Extracts entities from query and scores given a threshold.
        :param query_texts (`List[str]`): List of query strings
        :param queries (`torch.LongTensor` of shape `(batch_size, sequence_length)`): Indices of query sequence tokens in the vocabulary.
        :param p_starts (`torch.FloatTensor` of shape `(batch_size, sequence_length)`): Scores of each token as being start token of an entity
        :param p_ends (`torch.FloatTensor` of shape `(batch_size, sequence_length)`): Scores of each token as being end token of an entity
        :param entity_keys (`List[str]`): List of entity keys, default numeric indices i.e. 0, 1, 2, ... etc.
        :param thresh (`float [0,1]`): Threshold value for filtering spans
        :return: Non-overlapping predicted entity span (start and end indices) and score for each query
        """
        queries = queries.to("cpu")
        p_starts = p_starts.to("cpu")
        p_ends = p_ends.to("cpu")

        n_queries = len(queries["input_ids"])
        n_entities = len(p_starts)

        if entity_keys is None:
            entity_keys = list(range(n_entities))

        predicted_scores = [[] for _ in range(n_queries)]
        predicted_entities = [[] for _ in range(n_queries)]
        for entity_idx, (p_start, p_end) in enumerate(zip(p_starts, p_ends)):
            for idx in range(n_queries):
                start_indexes = range(len(self.tokenizer.tokenize(query_texts[idx])) + 1)
                end_indexes = range(len(self.tokenizer.tokenize(query_texts[idx])) + 1)
                all_pairs_scores = []
                for start_idx in start_indexes:
                    for end_idx in end_indexes:
                        if start_idx == end_idx == 1:
                            score = (p_start[idx][start_idx].item() + p_end[idx][end_idx].item()) / 2.
                            if score >= thresh:
                                continue
                        if start_idx < end_idx:
                            score = (p_start[idx][start_idx].item() + p_end[idx][end_idx].item()) / 2.
                            if score >= thresh:
                                s = queries["offset_mapping"][idx][start_idx:end_idx][0][0].item()
                                e = queries["offset_mapping"][idx][start_idx:end_idx][-1][-1].item()
                                decoded = query_texts[idx][s:e]
                                predicted_entities[idx].append(dict(start=s, end=e, entity_value=decoded, score=score,
                                                                    label=str(entity_keys[entity_idx])))

        nonoverlapping_predicted_entities = []

        for q in predicted_entities:
            if not len(q):
                nonoverlapping_predicted_entities.append([])
                continue
            q.sort(key=lambda x: x['score'], reverse=True)
            f = [q[0]]
            last = q[0]
            for o in q[1:]:
                if o['start'] < last['end']:
                    continue
                else:
                    f.append(o)
                    last = o

            nonoverlapping_predicted_entities.append(f)

        return nonoverlapping_predicted_entities
