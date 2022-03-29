import torch
from pytorch_lightning import (LightningModule)
from transformers import AutoModel


class FSNERModel(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            learning_rate: float = 5e-5,
            adam_epsilon: float = 1e-8,
            warmup_steps: int = 0,
            weight_decay: float = 0.0,
            epoch_steps: int = 0,
            token_embeddings_size: int = None,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(model_name_or_path, return_dict=True)
        self.cos = torch.nn.CosineSimilarity(3, 1e-08)
        self.softmax = torch.nn.Softmax(dim=1)
        self.entity_start_criterion = torch.nn.BCEWithLogitsLoss()
        self.entity_end_criterion = torch.nn.BCEWithLogitsLoss()
        self.epoch_steps = epoch_steps
        self.model.resize_token_embeddings(
            token_embeddings_size if token_embeddings_size else self.model.config.vocab_size)

    def setup(self, stage):
        if stage != "fit":
            return

    def criterion(self, p_starts, true_starts, p_ends, true_ends):
        return (self.entity_start_criterion(p_starts, true_starts) + self.entity_end_criterion(p_ends, true_ends)) / 2.

    def _bert(self, **inputs):
        return self.model(**inputs).last_hidden_state

    def _vector_sum(self, token_embeddings):
        return token_embeddings.sum(2, keepdim=True)

    def _atten(self, q_rep, S_rep, T=1):
        return self.softmax(T * self.cos(q_rep, S_rep))

    def forward(self, queries, supports):
        support_sizes = supports.pop("sizes")
        start_token_id = supports.pop("start_token_id")
        end_token_id = supports.pop("end_token_id")
        offset_mapping = queries.pop("offset_mapping")

        if len(set(support_sizes.tolist())) > 1:
            raise Exception("All support sizes have to be equal!")

        q = self._bert(**queries)
        s = self._bert(**supports)

        start_token_masks = supports["input_ids"] == start_token_id.item()
        end_token_masks = supports["input_ids"] == end_token_id.item()

        # (batch_size, 384, 784) -> (batch_size, 1, 384, 784)
        q = q.view(q.shape[0], -1, q.shape[1], q.shape[2])
        # (batch_size*n_examples_per_entity, 384, 784) -> (batch_size, n_examples_per_entity, 384, 784)
        s = s.view(q.shape[0], -1, s.shape[1], s.shape[2])

        q_rep = self._vector_sum(q)
        s_rep = self._vector_sum(s)

        start_token_embeddings = s[start_token_masks.view(s.shape[:3])].view(s.shape[0], -1, 1, s.shape[-1])
        end_token_embeddings = s[end_token_masks.view(s.shape[:3])].view(s.shape[0], -1, 1, s.shape[-1])

        atten = self._atten(q_rep, s_rep)

        p_starts = torch.sum(atten * torch.einsum("bitf,bejf->bet", q, start_token_embeddings), dim=1)
        p_ends = torch.sum(atten * torch.einsum("bitf,bejf->bet", q, end_token_embeddings), dim=1)

        supports["sizes"] = support_sizes
        supports["start_token_id"] = start_token_id
        supports["end_token_id"] = end_token_id
        queries["offset_mapping"] = offset_mapping

        return p_starts, p_ends

    def training_step(self, batch, batch_idx):
        queries, supports, true_starts, true_ends, _, _ = batch
        p_starts, p_ends = self(queries, supports)
        loss = self.criterion(p_starts, true_starts, p_ends, true_ends)
        self.log("train_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        queries, supports, true_starts, true_ends, entities, query_texts = batch
        p_starts, p_ends = self(queries, supports)
        loss = self.criterion(p_starts, true_starts, p_ends, true_ends)
        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return {"loss": loss, "p_starts": p_starts, "p_ends": p_ends, "true_starts": true_starts,
                "true_ends": true_ends}

    def validation_step_end(self, batch_parts):
        p_starts = batch_parts["p_starts"]
        p_ends = batch_parts["p_ends"]
        true_starts = batch_parts["true_starts"]
        true_ends = batch_parts["true_ends"]

        start_ids = p_starts.argsort(dim=1)[:, -1:]
        end_ids = p_ends.argsort(dim=1)[:, -1:]
        cnt = 0
        for i in range(len(p_starts)):
            if true_starts[i][start_ids[i]] == 1 and true_ends[i][end_ids[i]] == 1:
                cnt += 1
        acc = cnt / len(p_starts)

        return acc

    def validation_epoch_end(self, accs):
        if len(accs) > 0:
            mean_acc = sum(accs) / len(accs)
            self.log('val_acc_epoch', mean_acc, prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.learning_rate,
                                 eps=self.hparams.adam_epsilon)

    def prepare_supports(self, supports):
        """
        Precomputes supports' start and end token embeddings for reusing multiple times with different query set
        :param supports: Tokenized support examples, generated by TokenizerUtils.tokenize()
        :return: start and end token embeddings for each entity type
        """
        self.model.eval()
        with torch.no_grad():
            support_sizes = supports.pop("sizes")
            start_token_id = supports.pop("start_token_id")
            end_token_id = supports.pop("end_token_id")

            s = self._bert(**supports)

            start_token_masks = supports["input_ids"] == start_token_id.item()
            end_token_masks = supports["input_ids"] == end_token_id.item()

            start_token_embeddings, end_token_embeddings, idx = [], [], 0

            for size in support_sizes:
                start_token_embeddings.append(s[idx: idx + size][start_token_masks[idx: idx + size]])
                end_token_embeddings.append(s[idx: idx + size][end_token_masks[idx: idx + size]])
                idx += size

            supports["sizes"] = support_sizes
            supports["start_token_id"] = start_token_id
            supports["end_token_id"] = end_token_id

        return start_token_embeddings, end_token_embeddings

    def _predict_using_precalculated(self, queries, start_token_embeddings, end_token_embeddings):
        """
        Finds start and end token scores given the precalculated start and end token embeddings and queries
        :param queries: Tokenized queries, generated by TokenizerUtils.tokenize()
        :param start_token_embeddings: precalculated start token embeddings from supports
        :param end_token_embeddings: precalculated end token embeddings from supports
        :return: start and end token scores for each token in the query
        """
        self.model.eval()
        with torch.no_grad():
            offset_mapping = queries.pop("offset_mapping")

            q = self._bert(**queries)

            p_starts, p_ends, idx = [], [], 0

            for ste, ete in zip(start_token_embeddings, end_token_embeddings):
                for q_j in q:
                    # print(q_j.shape, ste.T.shape)
                    p_starts.append(torch.matmul(q_j, ste.T).mean(1).softmax(0))
                    p_ends.append(torch.matmul(q_j, ete.T).mean(1).softmax(0))

            p_starts = torch.vstack(p_starts).view(len(start_token_embeddings), q.shape[0], -1)
            p_ends = torch.vstack(p_ends).view(len(end_token_embeddings), q.shape[0], -1)

            queries["offset_mapping"] = offset_mapping

        return p_starts, p_ends

    def predict(self, queries, supports=None, start_token_embeddings=None, end_token_embeddings=None):
        """
        Find scores of each token being start and end token for an entity.
        :param queries: Tokenized queries, generated by TokenizerUtils.tokenize()
        :param supports: Tokenized support examples, generated by TokenizerUtils.tokenize()
        :param start_token_embeddings: precalculated start token embeddings from supports
        :param end_token_embeddings: precalculated end token embeddings from supports
        :return: start and end token scores for each token in the query
        """

        if start_token_embeddings is not None and end_token_embeddings is not None:
            return self._predict_using_precalculated(queries, start_token_embeddings, end_token_embeddings)

        self.model.eval()
        with torch.no_grad():
            support_sizes = supports.pop("sizes")
            start_token_id = supports.pop("start_token_id")
            end_token_id = supports.pop("end_token_id")
            offset_mapping = queries.pop("offset_mapping")

            q = self._bert(**queries)
            s = self._bert(**supports)

            start_token_masks = supports["input_ids"] == start_token_id.item()
            end_token_masks = supports["input_ids"] == end_token_id.item()

            p_starts, p_ends, idx = [], [], 0

            for size in support_sizes:
                start_token_embeddings = s[idx: idx + size][start_token_masks[idx: idx + size]]
                end_token_embeddings = s[idx: idx + size][end_token_masks[idx: idx + size]]
                for q_j in q:
                    p_starts.append(torch.matmul(q_j, start_token_embeddings.T).sum(1).softmax(0))
                    p_ends.append(torch.matmul(q_j, end_token_embeddings.T).sum(1).softmax(0))
                idx += size

            p_starts = torch.vstack(p_starts).view(len(support_sizes), q.shape[0], -1)
            p_ends = torch.vstack(p_ends).view(len(support_sizes), q.shape[0], -1)

            supports["sizes"] = support_sizes
            supports["start_token_id"] = start_token_id
            supports["end_token_id"] = end_token_id
            queries["offset_mapping"] = offset_mapping

        return p_starts, p_ends
