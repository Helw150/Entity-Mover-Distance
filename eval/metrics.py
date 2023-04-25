import numpy as np
from collections import defaultdict, Counter
from scipy.optimize import linear_sum_assignment
from vizier.service import clients
from vizier.service import pyvizier as vz
from tqdm import tqdm
import random
import string


def r_string(N=10):
    res = "".join(random.choices(string.ascii_uppercase + string.digits, k=N))
    return res


def eval_edges(eval_data, I, D):
    def evaluate(thresh: float) -> float:
        return eval_edges_(eval_data, I, D, thresh=thresh)

    flat_D = [d for ds in D for d in ds]
    # Algorithm, search space, and metrics.
    study_config = vz.StudyConfig(algorithm="GAUSSIAN_PROCESS_BANDIT")
    study_config.search_space.root.add_float_param("thresh", 0, max(flat_D))
    study_config.metric_information.append(
        vz.MetricInformation("metric_name", goal=vz.ObjectiveMetricGoal.MAXIMIZE)
    )

    # Setup client and begin optimization. Vizier Service will be implicitly created.
    study = clients.Study.from_study_config(
        study_config, owner="WillHeld", study_id="example_" + r_string()
    )
    for i in tqdm(range(50)):
        suggestions = study.suggest(count=1)
        for suggestion in suggestions:
            params = suggestion.parameters
            objective = evaluate(params["thresh"])
            print(params)
            suggestion.complete(vz.Measurement({"metric_name": objective}))
    for optimal_trial in study.optimal_trials():
        optimal_trial = optimal_trial.materialize()
        print(
            "Optimal Trial Suggestion and Objective:",
            optimal_trial.parameters,
            optimal_trial.final_measurement,
        )


def eval_edges_(eval_data, I, D, thresh=100000000):
    mention_to_gold = {}
    gold_clusters = defaultdict(list)
    for i, data in enumerate(eval_data):
        mention_to_gold[i] = data["label_id"]
        gold_clusters[data["label_id"]].append(i)
    gold_clusters_dict = gold_clusters
    gold_clusters = list(gold_clusters.values())
    mention_to_clusters = {}
    clusters = defaultdict(list)
    pos_edges = sorted(
        [(i, I[i][j], D[i][j]) for i, _ in enumerate(D) for j, _ in enumerate(D[0])],
        key=lambda x: x[2],
    )
    sanity = set()
    for i, idx, dist in pos_edges:
        if dist < thresh:
            if i in mention_to_clusters and idx in mention_to_clusters:
                cluster_key1 = mention_to_clusters[i]
                cluster_key2 = mention_to_clusters[idx]
                if cluster_key1 != cluster_key2:
                    cluster_key = cluster_key1
                    for member in clusters[cluster_key2]:
                        mention_to_clusters[member] = cluster_key
                        clusters[cluster_key].append(member)
                    del clusters[cluster_key2]
            elif idx in mention_to_clusters:
                cluster_key = mention_to_clusters[idx]
                assert idx in clusters[cluster_key]
                clusters[cluster_key].append(i)
                mention_to_clusters[i] = cluster_key
            elif i in mention_to_clusters:
                cluster_key = mention_to_clusters[i]
                assert i in clusters[cluster_key]
                clusters[cluster_key].append(idx)
                mention_to_clusters[idx] = cluster_key
            else:
                cluster_key = i
                assert clusters[cluster_key] == []
                clusters[cluster_key].extend([i, idx])
                mention_to_clusters[i] = cluster_key
                mention_to_clusters[idx] = cluster_key
        else:
            if i in mention_to_clusters:
                cluster_key = mention_to_clusters[i]
                assert i in clusters[cluster_key]
            else:
                cluster_key = i
                assert clusters[cluster_key] == []
                clusters[cluster_key].append(i)
                mention_to_clusters[i] = cluster_key
        sanity.add(i)
    clusters_dict = clusters
    clusters = list(clusters.values())
    assert sorted(list(sanity)) == sorted(list(mention_to_clusters.keys()))
    assert sorted(list(sanity)) == sorted([i for cluster in clusters for i in cluster])
    metrics = {"lea": lea, "b_cubed": b_cubed, "ceafe": ceafe, "muc": muc}
    for metric in metrics:
        if metric == "ceafe":
            pn, pd, rn, rd = ceafe(clusters, gold_clusters)
        elif metric == "lea":
            pn, pd = lea(clusters, gold_clusters_dict, mention_to_gold)
            rn, rd = lea(gold_clusters, clusters_dict, mention_to_clusters)
            _, _, objective = f1(pn, pd, rn, rd)
        else:
            pn, pd = metrics[metric](clusters, mention_to_gold)
            rn, rd = metrics[metric](gold_clusters, mention_to_clusters)
        print("{}: P={}, R={}, F1={}".format(metric, *f1(pn, pd, rn, rd)))
    return objective


# From https://github.com/clarkkev/deep-coref/blob/master/evaluation.py
def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    f1 = 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)
    return p, r, f1


def b_cubed(clusters, mention_to_gold):
    num, den = 0, 0

    for c in clusters:
        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[mention_to_gold[m]] += 1
        for c2 in gold_counts:
            correct += gold_counts[c2] * gold_counts[c2]

        num += correct / float(len(c))
        den += len(c)

    return num, den


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()
    return similarity, len(clusters), similarity, len(gold_clusters)


def ceafm(clusters, gold_clusters):
    clusters = [c for c in clusters]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi3(gold_clusters[i], clusters[j])
    row_ind, col_ind = linear_sum_assignment(-scores)
    similarity = scores[row_ind, col_ind].sum()
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(input_clusters, output_clusters, mention_to_gold):
    num, den = 0, 0

    for c in input_clusters:
        if len(c) == 1:
            all_links = 1
            if (
                c[0] in mention_to_gold
                and len(output_clusters[mention_to_gold[c[0]]]) == 1
            ):
                common_links = 1
            else:
                common_links = 0
        else:
            common_links = 0
            all_links = len(c) * (len(c) - 1) / 2.0
            for i, m in enumerate(c):
                if m in mention_to_gold:
                    for m2 in c[i + 1 :]:
                        if (
                            m2 in mention_to_gold
                            and mention_to_gold[m] == mention_to_gold[m2]
                        ):
                            common_links += 1

        num += len(c) * common_links / float(all_links)
        den += len(c)

    return num, den
