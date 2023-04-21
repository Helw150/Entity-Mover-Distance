from dataclasses import dataclass

import torch
import torch.nn.functional as F
import transformers
import random

import sys

from geomloss import SamplesLoss
from typing import Dict, List, Optional, Tuple, Any

sys.path.append("../modeling")


@dataclass
class DataCollatorForIndex:
    tokenizer: transformers.PreTrainedTokenizer

    def format_ex(self, example):
        if "context_left" in example:
            return (
                example["context_left"]
                + " [START_ENT] "
                + example["mention"]
                + " [END_ENT] "
                + example["context_right"]
            )
        else:
            return example

    def __call__(self, examples: Dict[str, Any], return_tensors="pt") -> Dict[str, Any]:
        examples = [dict(zip(examples, t)) for t in zip(*examples.values())]
        mention_text = [self.format_ex(example) for example in examples]
        reference_text = [
            self.format_ex(random.choice(example["label"])) for example in examples
        ]

        inputs = self.tokenizer(
            mention_text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        outputs = self.tokenizer(
            reference_text,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_ids=outputs.input_ids,
            output_mask=outputs.attention_mask,
        )


class BiEncoder(torch.nn.Module):
    def __init__(
        self, model_call, start_token, end_token, is_coref, dist="l2", margin=10
    ):
        super(BiEncoder, self).__init__()
        self.context_encoder = model_call()
        if not is_coref:
            self.cand_encoder = model_call()
        self.config = self.context_encoder.config
        self.start_token = start_token
        self.end_token = end_token
        self.is_coref = is_coref
        self.dist = dist
        self.margin = margin
        if dist == "ot":
            self.earth_mover_loss = SamplesLoss(
                loss="sinkhorn", p=2, blur=0.5, backend="online"
            )
        elif dist == "l2":
            self.triplet_loss = lambda ins, outs, negs, margin: F.triplet_margin_loss(
                ins,
                outs,
                negs,
                margin=margin,
                p=2,
                eps=1e-6,
                swap=False,
                reduction="mean",
            )

    def set_index(self, index, labels, tokenizations):
        self.faiss_index = index
        self.labels = labels
        self.tokenizations = tokenizations

    def get_hard_case(self, mention_reps, true_label):
        mention_reps = mention_reps.squeeze(1).detach().cpu().numpy()
        _, similar_sets = self.faiss_index.search(mention_reps, 50)
        hard_negative = []
        for sim_set in similar_sets:
            for case in sim_set:
                if self.labels[case] != true_label:
                    hard_negative.append(self.tokenizations[case])
                    break

        return (
            torch.stack([neg[0] for neg in hard_negative]),
            torch.stack([neg[1] for neg in hard_negative]),
        )

    def get_mention_rep(self, hidden_dim, ids):
        start_pieces = torch.argwhere(ids == self.start_token)[:, 1].flatten()
        end_pieces = torch.argwhere(ids == self.end_token)[:, 1].flatten()
        start_pieces = start_pieces.repeat(1, hidden_dim.shape[-1]).view(
            -1, 1, hidden_dim.shape[-1]
        )
        start_piece_vec = torch.gather(hidden_dim, 1, start_pieces)
        end_piece_vec = torch.gather(
            hidden_dim,
            1,
            end_pieces.repeat(1, hidden_dim.shape[-1]).view(
                -1, 1, hidden_dim.shape[-1]
            ),
        )

        mention_rep = torch.cat([start_piece_vec, end_piece_vec], dim=2)
        return mention_rep.squeeze(1)

    def get_weight(self, mask):
        probs = mask / mask.sum(1).reshape(-1, 1)
        return probs

    def compute_loss(self, ins, outs, negs, in_mask=None, out_mask=None, neg_mask=None):
        if self.dist == "l2":
            loss = self.triplet_loss(ins, outs, negs, self.margin)
        else:
            loss = (
                F.relu(
                    self.earth_mover_loss(
                        self.get_weight(in_mask), ins, self.get_weight(out_mask), outs
                    )
                    - self.earth_mover_loss(
                        self.get_weight(in_mask), ins, self.get_weight(neg_mask), negs
                    )
                    + self.margin
                )
            ).mean()

        return loss

    def forward(
        self,
        input_ids,
        attention_mask,
        output_ids,
        output_mask,
        label_ids=None,
        no_loss=False,
        **kwargs
    ):
        if type(input_ids) == type([]):
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            output_ids = torch.tensor(output_ids)
            output_mask = torch.tensor(output_mask)
        start_index = torch.argwhere(input_ids == self.start_token)[:, 1].flatten()
        end_index = torch.argwhere(input_ids == self.end_token)[:, 1].flatten()
        in_hidden_dim = self.context_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        embedding_ctxt = self.get_mention_rep(
            in_hidden_dim,
            input_ids,
        )

        if self.is_coref:
            out_hidden_dim = self.context_encoder(
                input_ids=output_ids, attention_mask=output_mask
            ).last_hidden_state
            embedding_cands = self.get_mention_rep(out_hidden_dim, output_ids)
        else:
            out_hidden_dim = self.cand_encoder(
                input_ids=output_ids, attention_mask=output_mask
            ).last_hidden_state
            embedding_cands = out_hidden_dim[:, :2]

        if not self.training and no_loss == True:
            return {"input_rep": embedding_ctxt, "output_rep": embedding_cands}
        elif not self.training:
            return {
                "input_rep": embedding_ctxt,
                "output_rep": embedding_cands,
                "loss": torch.zeros_like(embedding_ctxt).sum(),
            }
        else:
            hard_neg_tok, hard_neg_mask = self.get_hard_case(embedding_ctxt, label_ids)
            hard_neg_tok, hard_neg_mask = (
                hard_neg_tok.to(output_ids.device),
                hard_neg_mask.to(output_mask.device),
            )

            if self.is_coref:
                hard_neg_hidden_dim = self.context_encoder(
                    input_ids=hard_neg_tok, attention_mask=hard_neg_mask
                ).last_hidden_state
                embedding_neg = self.get_mention_rep(hard_neg_hidden_dim, hard_neg_tok)
            else:
                hard_neg_hidden_dim = self.cand_encoder(
                    input_ids=hard_neg_tok, attention_mask=hard_neg_mask
                ).last_hidden_state
                embedding_neg = hard_neg_hidden_dim[:, :2]
            if self.dist == "l2":
                loss = self.compute_loss(embedding_ctxt, embedding_cands, embedding_neg)
            else:
                loss = self.compute_loss(
                    in_hidden_dim,
                    out_hidden_dim,
                    hard_neg_hidden_dim,
                    in_mask=attention_mask,
                    out_mask=output_mask,
                    neg_mask=hard_neg_mask,
                )
            return {
                "input_rep": embedding_ctxt,
                "output_rep": embedding_cands,
                "loss": loss,
            }
