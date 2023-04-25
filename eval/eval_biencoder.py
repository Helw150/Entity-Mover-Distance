import sys
from dataclasses import dataclass, field
from datasets import load_dataset
import random
import time
import faiss
import numpy as np
import torch
from geomloss import SamplesLoss
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

import torch
import torch.nn.functional as F
import transformers

sys.path.append("../modeling")

from models import BiEncoder, DataCollatorForIndex
from metrics import eval_edges


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
def create_faiss_db(eval_data, model, tokenizer, use_label=False, return_full=False):
    hidden = model.context_encoder.config.hidden_size * 2
    collator = DataCollatorForIndex(tokenizer=tokenizer)
    tokenized = eval_data.map(
        collator, load_from_cache_file=False, batched=True
    ).with_format(type="torch")
    reps = tokenized.map(
        lambda inputs: model.forward(return_full=return_full, no_loss=True, **inputs),
        batched=True,
        load_from_cache_file=False,
        batch_size=16,
    )
    if use_label:
        reference_vectors = np.array(
            [rep["output_rep"].cpu().numpy() for rep in reps]
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
        None if not return_full else reps,
    )


def get_weight(mask):
    probs = mask / mask.sum(1).reshape(-1, 1)
    return probs


@torch.no_grad()
def ot_sort(in_rep, in_mask, out_reps, out_masks):
    out_reps, out_masks = out_reps.cuda(), out_masks.cuda()
    batch_size, tok_size, hidden_dim = out_reps.shape
    start = time.time()
    end_trunc = in_mask.nonzero()[-1].cpu().numpy()[0]
    start_trunc = in_mask.nonzero()[0].cpu().numpy()[0]
    in_reps = in_rep.repeat(out_reps.shape[0], 1, 1)
    in_masks = in_mask.repeat(out_reps.shape[0], 1)
    end_trunc = max(max(out_masks.nonzero()[:, 1].cpu().numpy()), end_trunc) + 1
    start_trunc = min(min(out_masks.nonzero()[:, 1].cpu().numpy()), start_trunc)
    in_reps = (
        in_reps[:, start_trunc:end_trunc, :]
        .reshape(batch_size, -1, hidden_dim)
        .contiguous()
    )
    in_masks = in_masks[:, start_trunc:end_trunc].reshape(batch_size, -1).contiguous()
    out_reps = (
        out_reps[:, start_trunc:end_trunc, :]
        .reshape(batch_size, -1, hidden_dim)
        .contiguous()
    )
    out_masks = out_masks[:, start_trunc:end_trunc].reshape(batch_size, -1).contiguous()
    dists = dist_func(get_weight(in_masks), in_reps, get_weight(out_masks), out_reps)
    end = time.time()
    return dists.argsort(), dists


@torch.no_grad()
def head_sort(in_rep, in_mask, out_reps, out_masks):
    out_reps, out_masks = out_reps.cuda(), out_masks.cuda()
    batch_size, tok_size, hidden_dim = out_reps.shape
    start = time.time()
    head = in_mask.nonzero()[1].cpu().numpy()
    in_rep = in_rep[head, :]
    idx = torch.arange(tok_size, 0, -1)
    val = out_masks * idx
    heads = torch.argmax(val, 1, keepdim=True) + 1
    heads = heads.repeat(1, out_reps.shape[-1]).view(-1, 1, out_reps.shape[-1])
    out_reps = torch.gather(out_reps, 1, heads).squeeze(1)
    dists = F.mse_loss(in_rep.repeat(batch_size, 1), out_reps, reduction="none").mean(1)
    end = time.time()
    return dists.argsort(), dists


@torch.no_grad()
def mean_sort(in_rep, in_mask, out_reps, out_masks):
    out_reps, out_masks = out_reps.cuda(), out_masks.cuda()
    batch_size, tok_size, hidden_dim = out_reps.shape
    start = time.time()
    head = in_mask.nonzero()[1].cpu().numpy()
    in_rep = (in_rep * in_mask.unsqueeze(-1)).mean(0)
    out_reps = (out_reps * out_masks.unsqueeze(-1)).mean(1)
    dists = F.mse_loss(in_rep.repeat(batch_size, 1), out_reps, reduction="none").mean(1)
    end = time.time()
    return dists.argsort(), dists


@torch.no_grad()
def score(
    matches,
    eval_data,
    I,
    D,
    full_data=None,
    use_label=False,
    sort=None,
    verbose=False,
    model=None,
):
    relevance_matrix = []
    tp = 0
    precision = 0
    singletons = 0
    if full_data:
        in_full, in_mask, out_full, out_mask = full_data
    cand_lists, dists = [], []
    for search_i, match in enumerate(tqdm(matches)):
        true_label = match[0]
        cands = match[1:]
        sort_keys = None
        if sort != None:
            out_full_l = out_full[I[search_i][1:]]
            out_mask_l = out_mask[I[search_i][1:]]
        if sort == "ot":
            sort_keys, dist = ot_sort(
                in_full[search_i],
                in_mask[search_i],
                out_full_l,
                out_mask_l,
            )
        elif sort == "head":
            sort_keys, dist = head_sort(
                in_full[search_i],
                in_mask[search_i],
                out_full_l,
                out_mask_l,
            )
        elif sort == "mean":
            sort_keys, dist = mean_sort(
                in_full[search_i],
                in_mask[search_i],
                out_full_l,
                out_mask_l,
            )
        else:
            dist = D[search_i].tolist()[1:]
        idxes = I[search_i][1:]
        if sort_keys != None:
            cands = [cands[i] for i in sort_keys]
            dist = [dist[i] for i in sort_keys]
            idxes = [idxes[i] for i in sort_keys]
        assert dist == sorted(dist)
        cand_lists.append(idxes)
        dists.append(dist)
        if type(true_label) == type("") and "Singleton" in true_label:
            singletons += 1
            continue
        relevance = [label == true_label for label in cands]
        num_correct = np.sum(relevance)
        precision += num_correct / len(cands)
        if num_correct >= 1:
            tp += 1
        relevance_matrix.append(relevance)

    if verbose:
        recalls_at_k(relevance_matrix)
        if not use_label:
            eval_edges(eval_data, cand_lists, dists)
    return (
        tp / float(len(I) - singletons),
        mean_reciprocal_rank(relevance_matrix),
        mean_average_precision(relevance_matrix),
        precision / float(len(I) - singletons),
    )


@torch.no_grad()
def nn_eval(
    eval_data, model, tokenizer, k=10, use_label=False, use_ot_sort=False, verbose=False
):
    print(use_label)
    index, search_vectors, _, tokens, ot_data = create_faiss_db(
        eval_data, model, tokenizer, use_label, use_ot_sort
    )
    if use_label and use_ot_sort:
        out_full = torch.stack([data["output_full"] for data in ot_data])
        out_mask = torch.stack([data["out_ot_mask"] for data in ot_data])
        # out_full = torch.stack([data[0] for data in tokens])
        # out_mask = torch.stack([data[1] for data in tokens])
    else:
        out_full = torch.stack([data["input_full"] for data in ot_data])
        out_mask = torch.stack([data["in_ot_mask"] for data in ot_data])

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

    ot_data_loaded = None
    if use_ot_sort:
        print("NO SORT")
        scores = score(
            matches,
            eval_data,
            I,
            D,
            sort=None,
            verbose=verbose,
            use_label=use_label,
            model=model,
        )
        print(scores)
        full_data_loaded = (
            [data["input_full"] for data in ot_data],
            [data["in_ot_mask"] for data in ot_data],
            out_full,
            out_mask,
        )
        print("HEAD SORT")
        scores = score(
            matches,
            eval_data,
            I,
            D,
            full_data=full_data_loaded,
            sort="head",
            verbose=verbose,
            use_label=use_label,
            model=model,
        )
        print(scores)
        print("MEAN SORT")
        scores = score(
            matches,
            eval_data,
            I,
            D,
            full_data=full_data_loaded,
            sort="mean",
            verbose=verbose,
            use_label=use_label,
            model=model,
        )
        print(scores)
        print("OT SORT")
    sort = "ot" if use_ot_sort else None
    return score(
        matches,
        eval_data,
        I,
        D,
        full_data=full_data_loaded,
        sort=sort,
        verbose=verbose,
        use_label=use_label,
        model=model,
    )


def recalls_at_k(relevance_matrix):
    relevance_matrix = np.array(relevance_matrix)
    print(relevance_matrix.shape)
    for k in [1, 2, 4, 8, 16, 32, 64]:
        recall_at_k = relevance_matrix[:, :k].sum(1)
        recall_at_k = len(recall_at_k.nonzero()[0]) / relevance_matrix.shape[0]
        print("Recall at {}: {}".format(k, recall_at_k))


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="bert-base-uncased")


@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    dataset: str = field(default="ECB")
    load_path: str = field(default=None)
    seed: int = field(default=1)
    use_ot_sort: bool = field(default=True)
    neighbors: int = field(default=64)


def eval():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    model_call = lambda: transformers.AutoModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    num_added_toks = tokenizer.add_tokens(["[START_ENT]", "[END_ENT]"])
    start_token = tokenizer("[START_ENT]").input_ids[1]
    end_token = tokenizer("[END_ENT]").input_ids[1]
    model = BiEncoder(model_call, start_token, end_token)
    new_embeds = model.context_encoder.resize_token_embeddings(len(tokenizer))
    if training_args.load_path:
        ckpt = torch.load(training_args.load_path)
        model.load_state_dict(ckpt)
    model.eval()
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

    print(
        nn_eval(
            dataset["validation"]
            if not "ZESHEL" in training_args.dataset
            else dataset["validation"].filter(
                lambda x: x["world"] == "coronation_street"
            ),
            model,
            tokenizer,
            k=training_args.neighbors,
            use_label="ZESHEL" in training_args.dataset,
            use_ot_sort=training_args.use_ot_sort,
            verbose=True,
        )
    )


if __name__ == "__main__":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    from datasets import disable_caching

    dist_func = SamplesLoss(loss="sinkhorn", p=2, scaling=0.9, backend="tensorized")
    disable_caching()
    eval()
