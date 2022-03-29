import random

from pytorch_lightning import (LightningDataModule)
from torch.utils.data import DataLoader, IterableDataset
from typing import Dict, List

from fsner.tokenizer_utils import FSNERTokenizerUtils


class FSNERDataset(IterableDataset):
    def __init__(self, data, n_examples_per_entity: int = 8, negative_examples_ratio: float = 0.5):
        """
        Args:
            data: Dataset as dictionary where examples are grouped by entity. Note: Assuming each example must have start and end token ids.
            n_examples_per_entity: Number of examples per entity in each batch
            negative_examples_ratio: Ratio of negative example and positive example. 0.25 would mean every 4th entity would be a negative example of the query chosen for it
        """
        self.data = data
        self.n_examples_per_entity = n_examples_per_entity
        self.negative_examples_ratio = negative_examples_ratio
        self.counter = 0
        self.entities = list(self.data.keys())
        random.shuffle(self.entities)
        max_n_examples = max([len(self.data[key]) for key in self.data])
        assert 0. < self.negative_examples_ratio < 1., "negative_examples_ratio has to be greater than 0 and less than 1"
        assert 0. < self.n_examples_per_entity < max_n_examples, f"n_examples_per_entity has to be greater than 0 and less than maximum number of examples ({max_n_examples}) for any entity group"

    def _neg(self, ent):
        entities = self.entities.copy()
        entities.remove(ent)
        random.shuffle(entities)
        for entity in entities:
            if len(self.data[entity]) >= self.n_examples_per_entity:
                return random.sample(self.data[entity], self.n_examples_per_entity)
        return []

    def __iter__(self):
        index_pointers = {entity: 0 for entity in self.entities}

        while len(self.entities):
            self.counter += 1
            entity = random.choice(self.entities)
            is_negative = self.counter % int(1. / self.negative_examples_ratio) == 0
            if not is_negative:
                s = index_pointers[entity]
                e = index_pointers[entity] + self.n_examples_per_entity
                if e > len(self.data[entity]):
                    del index_pointers[entity]
                    self.entities.remove(entity)
                    continue
                supports = self.data[entity][s:e]
                query = random.choice(self.data[entity][:s] + self.data[entity][e:])
                index_pointers[entity] = e
            else:
                supports = self._neg(entity)
                if len(supports) == self.n_examples_per_entity and len(self.data[entity]) > 0:
                    query = random.choice(self.data[entity])
                else:
                    continue

            yield [is_negative, supports, query, entity]


class FSNERDataModule(LightningDataModule):

    def __init__(
            self,
            train_data_dict: Dict[str, List[str]],
            val_data_dict: Dict[str, List[str]],
            tokenizer: FSNERTokenizerUtils = None,
            train_batch_size: int = 32,
            val_batch_size: int = 32,
            n_examples_per_entity: int = 8,
            negative_examples_ratio: float = 0.25,
            **kwargs,
    ):
        super().__init__()
        self.train_data_dict = train_data_dict
        self.val_data_dict = val_data_dict
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.tokenizer = tokenizer
        self.n_examples_per_entity = n_examples_per_entity
        self.negative_examples_ratio = negative_examples_ratio

    def _get_labels(self, is_negative, query_texts):
        """Converts text labels to tensor labels.
        For each positive example, the corresponding start and end token labels are 1s.
        For each negative example, the [CLS] token is the start token and next token is the end token and are assigned as 1s.
        """
        input_token_ids = self.tokenizer.tokenize(query_texts)['input_ids'].detach()
        start_token_id = self.tokenizer.entity_start_token_id
        end_token_id = self.tokenizer.entity_end_token_id
        true_starts = (input_token_ids == start_token_id).float()
        true_ends = (input_token_ids == end_token_id).float().roll(-1, 1)
        negative_indices = list(filter(lambda x: is_negative[x], range(len(is_negative))))
        for idx in negative_indices:
            true_starts[idx] *= 0.
            true_ends[idx] *= 0.
            true_starts[idx][0] = 1.
            true_ends[idx][0] = 1.
        return true_starts, true_ends

    def _collate_fn(self, batch):
        is_negative, support_texts, query_texts, entities = list(map(list, zip(*batch)))
        query_texts_replaced = [
            q.replace(self.tokenizer.entity_start_token, '').replace(self.tokenizer.entity_end_token, '') for q
            in query_texts]
        queries = self.tokenizer.tokenize(query_texts_replaced)
        supports = self.tokenizer.tokenize(support_texts)
        entities = [("~ " if is_negative[idx] else "") + entities[idx] for idx in range(len(is_negative))]
        true_starts, true_ends = self._get_labels(is_negative, query_texts)
        return queries, supports, true_starts, true_ends, entities, query_texts_replaced

    def train_dataloader(self):
        return DataLoader(FSNERDataset(self.train_data_dict, n_examples_per_entity=self.n_examples_per_entity,
                                       negative_examples_ratio=self.negative_examples_ratio),
                          batch_size=self.train_batch_size, collate_fn=self._collate_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(FSNERDataset(self.val_data_dict, n_examples_per_entity=self.n_examples_per_entity,
                                       negative_examples_ratio=self.negative_examples_ratio),
                          batch_size=self.val_batch_size, collate_fn=self._collate_fn, drop_last=True)

    @property
    def epoch_steps(self):
        cnt = 0
        for _ in DataLoader(FSNERDataset(self.train_data_dict, n_examples_per_entity=self.n_examples_per_entity,
                                         negative_examples_ratio=self.negative_examples_ratio),
                            batch_size=self.train_batch_size, drop_last=True):
            cnt += 1
        return cnt
