from dataclasses import dataclass, field
from datasets import load_dataset
import random
from typing import Dict, List, Optional, Tuple, Any

import torch
import transformers
import sys

sys.path.append("../modeling")
sys.path.append("../eval")

from models import BiEncoder
from eval_biencoder import nn_eval, create_faiss_db


class NeighborTrainer(transformers.Trainer):
    def _wrap_model(self, model, training=True, dataloader=None):
        model = super(NeighborTrainer, self)._wrap_model(model, training, dataloader)
        self.model.eval()
        index, _, labels, tokenizations = create_faiss_db(
            self.train_dataset, self.model, self.tokenizer
        )
        self.model.set_index(index, labels, tokenizations)
        if training:
            self.model.train()
        return model

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        self._memory_tracker.start()
        model = self._wrap_model(self.model, training=False)
        f1, mrr, _, _ = nn_eval(self.eval_dataset, model, self.tokenizer)
        metrics = {metric_key_prefix + "_f1": f1, metric_key_prefix + "_mrr": mrr}
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics


@dataclass
class DataCollatorForAlignment:
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

    def __call__(
        self, examples: List[Dict[str, Any]], return_tensors="pt"
    ) -> Dict[str, Any]:
        mention_text = [self.format_ex(example) for example in examples]
        reference_text = [
            self.format_ex(random.choice(example["label"])) for example in examples
        ]

        inputs = self.tokenizer(mention_text, padding=True, return_tensors="pt")
        outputs = self.tokenizer(reference_text, padding=True, return_tensors="pt")
        return dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_ids=outputs.input_ids,
            output_mask=outputs.attention_mask,
            label_ids=[example["label_id"] for example in examples],
        )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="google/electra-large-discriminator"
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    dataset: str = field(default="ECB")
    dist: str = field(default="l2")


def pretrain():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    training_args.is_coref = training_args.dataset != "ZESHEL"
    model_call = lambda: transformers.AutoModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    num_added_toks = tokenizer.add_tokens(["[START_ENT]", "[END_ENT]"])
    start_token = tokenizer("[START_ENT]").input_ids[1]
    end_token = tokenizer("[END_ENT]").input_ids[1]
    model = BiEncoder(
        model_call,
        start_token,
        end_token,
        is_coref=training_args.is_coref,
        dist=training_args.dist,
    )
    model.context_encoder.resize_token_embeddings(len(tokenizer))
    if not training_args.is_coref:
        model.cand_encoder.resize_token_embeddings(len(tokenizer))
    if "ECB" in training_args.dataset:
        base_url = "../coref_data/ecb/final/"
    elif "CD2CR" in training_args.dataset:
        base_url = "../coref_data/cd2cr/final/"
    elif "ZESHEL" in training_args.dataset:
        base_url = "../blink_data/data/zeshel/blink_format/"

    dataset = load_dataset(
        "json",
        data_files={
            "train": base_url + "train.jsonl",
            "validation": base_url + "valid.jsonl",
        },
    )

    data_module = dict(
        train_dataset=dataset["train"].shuffle(seed=training_args.seed),
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForAlignment(tokenizer=tokenizer),
    )

    model.set_index(None, None, None)

    trainer = NeighborTrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model()
    trainer.save_state()
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    pretrain()
