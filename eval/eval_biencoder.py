import sys
from dataclasses import dataclass, field
from datasets import load_dataset
import random
import faiss
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any

import torch
import transformers

sys.path.append("../modeling")

from models import BiEncoder, DataCollatorForIndex


def mean_reciprocal_rank(rs):
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1.0 / (r[0] + 1) if r.size else 0.0 for r in rs])


def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError("Relevance score length < k")
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.0
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


@torch.no_grad()
def create_faiss_db(eval_data, model, tokenizer, use_label=False):
    hidden = model.context_encoder.config.hidden_size * 2
    collator = DataCollatorForIndex(tokenizer=tokenizer)
    tokenized = eval_data.map(collator, batched=True).with_format(type="torch")
    reps = tokenized.map(
        lambda inputs: model.forward(no_loss=True, **inputs),
        batched=True,
        load_from_cache_file=False,
        batch_size=16,
    )
    if use_label:
        reference_vectors = np.array(
            [rep["output_rep"].cpu().numpy() for rep_batch in reps]
        ).reshape(-1, hidden)
        search_vectors = np.array(
            [rep["input_rep"].cpu().numpy() for rep in reps]
        ).reshape(-1, hidden)
    else:
        search_vectors = reference_vectors = np.array(
            [rep["input_rep"].cpu().numpy() for rep in reps]
        ).reshape(-1, hidden)
    index = faiss.IndexFlatL2(hidden)
    index.add(reference_vectors)
    assert index.ntotal == len(eval_data)
    return (
        index,
        search_vectors,
        [data["label"] for data in eval_data],
        [(data["output_ids"], data["output_mask"]) for data in tokenized],
    )


@torch.no_grad()
def nn_eval(eval_data, model, tokenizer, k=5, use_label=False):
    index, search_vectors, _, _ = create_faiss_db(
        eval_data, model, tokenizer, use_label
    )
    if use_label:
        D, I = index.search(search_vectors, k)
        I = I.tolist()
        matches = [
            [eval_data[ref]["label_id"]] + [eval_data[i]["label_id"] for i in row]
            for ref, row in enumerate(I)
        ]
    else:
        # Add 1 since the first will be identity
        D, I = index.search(search_vectors, k + 1)
        I = I.tolist()
        matches = [[eval_data[i]["label_id"] for i in row] for row in I]
    relevance_matrix = []
    tp = 0
    precision = 0
    singletons = 0
    for match in matches:
        true_label = match[0]
        if type(true_label) == type("") and "Singleton" in true_label:
            singletons += 1
            continue
        cands = match[1:]
        relevance = [label == true_label for label in cands]
        num_correct = np.sum(relevance)
        precision += num_correct / k
        if num_correct >= 1:
            tp += 1
        relevance_matrix.append(relevance)
    return (
        tp / float(len(I) - singletons),
        mean_reciprocal_rank(relevance_matrix),
        mean_average_precision(relevance_matrix),
        precision / float(len(I) - singletons),
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="google/electra-large-discriminator"
    )


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    dataset: str = field(default="ECB")
    load_path: str = field(default=None)


def eval():
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
        model_call, start_token, end_token, is_coref=training_args.is_coref
    )
    model.context_encoder.resize_token_embeddings(len(tokenizer))
    if not training_args.is_coref:
        model.cand_encoder.resize_token_embeddings(len(tokenizer))
    model.eval()
    if training_args.load_path:
        ckpt = torch.load(training_args.load_path)
        model.load_state_dict(ckpt)
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
            "test": base_url + "valid.jsonl",
        },
    )

    print(nn_eval(dataset["validation"], model, tokenizer))


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    eval()
