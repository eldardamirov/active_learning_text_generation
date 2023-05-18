# -*- coding: utf-8 -*-
import subprocess


import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    DataCollatorForSeq2Seq,
)

from datasets import (
    load_dataset,
    load_metric,
    Dataset,
    concatenate_datasets,
    DatasetDict,
)
from munch import Munch
from torch.distributions import Categorical
from tqdm import tqdm
import numpy as np
import ruamel.yaml as yaml
import pandas as pd
from functools import partial
from typing import Tuple
from pathlib import Path
from math import ceil
import argparse

import os
import pickle
import matplotlib.pyplot as plt

from metrics.compute_metrics import compute_metrics

plt.rcParams["figure.figsize"] = (10, 7)
plt.rcParams["axes.facecolor"] = "lightgrey"
plt.rcParams["axes.grid"] = True
plt.rcParams["font.size"] = 16


def construct_input_for_batch(batch):
    """Construct input strings from a batch."""
    source = [" ".join(concepts) for concepts in batch["concepts"]]
    target = batch["target"]
    return source, target


def batch_tokenize(batch, tokenizer, config):
    """Construct the batch (source, target) and run them through a tokenizer."""
    max_length = config['MAX_LENGTH']
    to_pad = config['strategy'] == 'uncertainty'
    source, target = construct_input_for_batch(batch)
    src_tokenized = tokenizer(source,
                              truncation=True,
                              padding='max_length' if to_pad else False,
                              max_length=max_length if to_pad else None)
    with tokenizer.as_target_tokenizer():
        trg_tokenized = tokenizer(target, truncation=True)
    res = {
        "input_ids": src_tokenized["input_ids"],
        "attention_mask": src_tokenized["attention_mask"],
        "labels": trg_tokenized["input_ids"],
    }
    return res


def beam_generate_sentences(
    batch, model, tokenizer, num_beams=4, max_length=32, device="cuda:0"
):
    """Generate outputs from a model with beam search decoding."""
    # Create batch inputs.
    source, _ = construct_input_for_batch(batch)
    # Use the model's tokenizer to create the batch input_ids.
    batch_features = tokenizer(source, padding=True, return_tensors="pt")
    # Move all inputs to the device.
    batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])

    # Generate with beam search.
    generated_ids = model.generate(
        **batch_features, num_beams=num_beams, max_length=max_length,
    )

    # Use model tokenizer to decode to text.
    generated_sentences = [
        tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
        for gen_ids in generated_ids
    ]
    return generated_sentences


def shuffle_and_concatenate(dataset: Dataset, clusters: torch.Tensor, seed: int = 42):
    new_dataset = (
        dataset.add_column("cluster", list(clusters.cpu().numpy()))
        .shuffle(seed=seed)
        .flatten_indices()
    )
    return new_dataset


def load_ds(config) -> Tuple[Dataset, Dataset]:
    dataset = load_dataset(*config["dataset"])
    train_set = dataset["train"]
    test_set = dataset["validation"]
    print(f"Dataset loaded.\nTrain size: {len(train_set)}\nTest size: {len(test_set)}")
    if config["dataset"][0] == 'midas/kp20k':
        train_set = train_set.rename_column('document', 'concepts')
        test_set = test_set.rename_column('document', 'concepts')
        train_set = train_set.rename_column('extractive_keyphrases', 'target')
        test_set = test_set.rename_column('extractive_keyphrases', 'target').select(np.arange(1000))

        def concat_target(example):
            example['target'] = ', '.join(example['target'])
            return example
        train_set = train_set.map(concat_target)
        test_set = test_set.map(concat_target)

    return train_set, test_set


def remove_outliers(dataset: Dataset):
    return dataset.filter(lambda x: x["cluster"] != -1)


def tokenize(df, tokenizer, config):
    data_tokenized = df.map(
        lambda batch: batch_tokenize(batch, tokenizer, config),
        batched=True,
        remove_columns=[x for x in df.column_names if x not in ["id", "cluster"]],
    )
    return data_tokenized


def make_new_set(train_set, n_new_per_cluster, seed: int = 42):
    np.random.seed(seed)
    clusters = np.array(train_set["cluster"])
    unique_clusters = set(clusters.tolist())
    if -1 in unique_clusters:
        unique_clusters.remove(-1)
    n_clusters = len(unique_clusters)
    new_set = []

    for cluster in tqdm(range(n_clusters)):
        cluster_idx = np.argwhere(clusters == cluster).ravel()
        assert len(cluster_idx) != 0, f"no instance of cluster {cluster}"
        assert (
            len(cluster_idx) > n_new_per_cluster
        ), f"new set takes all from the cluster {cluster}"
        sampled_idx = np.random.choice(cluster_idx, n_new_per_cluster, replace=False)
        new_set += sampled_idx.tolist()

    left_idx = np.setdiff1d(range(len(clusters)), new_set)
    train_no_new_set = train_set.select(left_idx)
    new_set = train_set.select(new_set)
    return train_no_new_set, new_set


def filter_builder(strategy, compute_metrics_fn=None, config=None):
    if strategy == "rouge_filter":

        def rouge_filter_(pred):
            return rouge_filter(pred, compute_metrics_fn, config)

        return rouge_filter_
    elif strategy == "dummy":
        return dummy
    else:
        return NotImplementedError


def rouge_filter(pred, compute_metrics_fn, config):
    return compute_metrics_fn(pred, compute_cola=False)[config["rouge_metric"]] > config["rouge_threshold"]


def dummy(pred):
    return False


def get_probs(model, input, device):
    gen = model.generate(
        torch.tensor(input).to(device),
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=30,
        num_beams=1,
    )

    transition_scores = model.compute_transition_scores(
        gen.sequences, gen.scores, normalize_logits=True
    )

    return NotImplementedError


def uncertainty(pred):
    return NotImplementedError


def dump(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def not_special(seq):
    return (seq != 0) * (seq != 1) * (seq != 2)


def uncertainty(transition_scores, seq):
    unc = 1 - torch.exp(transition_scores)
    return (unc * not_special(seq)[:, 1:]).mean(axis=-1)


def mean_by_label(samples, labels):
    labels = labels.long()
    """ select mean(samples), count() from samples group by labels order by labels asc """
    weight = torch.zeros(labels.max().item() + 1, samples.shape[0]).to(
        samples.device
    )  # L, N
    weight[labels, torch.arange(samples.shape[0])] = 1
    label_count = weight.sum(dim=1)
    weight = torch.nn.functional.normalize(
        weight, p=1, dim=1
    ).float()  # l1 normalization
    mean = weight @ samples.float()  # L, F
    index = torch.arange(mean.shape[0]).to(samples.device)[label_count > 0]
    return mean[index]


def uncertainty_filter(pred, dev_clusters, config):
    assert dev_clusters.min() == 0
    assert dev_clusters.max() == len(set(dev_clusters)) - 1
    unc = uncertainty(pred.scores, pred.predictions)
    clusters_unc = mean_by_label(unc, torch.LongTensor(dev_clusters))
    return torch.tensor(torch.topk(
        clusters_unc, int(config["uncertainty_sampling_rate"] * len(clusters_unc))
    ).indices)


def get_sequences_and_scores(model, dev_data_tokenized, config):
    gen = model.generate(
        torch.tensor(dev_data_tokenized["input_ids"]).to(config["device"]),  # TODO
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=config["MAX_LENGTH"],
        num_beams=1,
    )
    transition_scores = model.compute_transition_scores(
        gen.sequences, gen.scores, normalize_logits=True
    )
    return gen.sequences, transition_scores


def top_uncertainty_clusters(trainer, dev_set, config):
    model = trainer.model
    dev_clusters = np.array(dev_set["cluster"])
    seqs, scores = get_sequences_and_scores(model, dev_set, config)
    pred = Munch()
    pred.scores = scores
    pred.predictions = seqs
    return uncertainty_filter(pred, dev_clusters, config)


def worst_rouge_clusters(trainer, dev_set, config):
    worst_clusters = []
    dev_clusters = np.array(dev_set["cluster"])
    dev_pred = trainer.predict(dev_set).predictions

    tokenizer = trainer.tokenizer
    compute_metrics_fn = partial(compute_metrics, tokenizer=tokenizer)
    filter_fn = filter_builder(config["strategy"], compute_metrics_fn, config)

    for cluster in list(set(dev_clusters)):
        pred = Munch()
        is_good = False
        cluster_idx = np.argwhere(dev_clusters == cluster).ravel()
        pred.label_ids = dev_set[cluster_idx]["labels"]
        pred.predictions = (
            dev_pred[cluster_idx]
            if config["dev_size_per_cluster"] != 1
            else dev_pred[cluster_idx][None, :]
        )
        inputs = dev_set[cluster_idx]['input_ids']
        is_good = filter_fn((pred.predictions, pred.label_ids, inputs))

        if not is_good:
            worst_clusters.append(cluster)

    return torch.tensor(worst_clusters)


def return_all_clusters(trainer, dev_set, config):
    return list(set(dev_set["cluster"]))


def build_cluster_getter(config):
    strategy = config["strategy"]
    if strategy == "rouge_filter":
        return worst_rouge_clusters
    elif strategy == "uncertainty":
        return top_uncertainty_clusters
    elif strategy == "dummy":
        return return_all_clusters
    
    
def get_train_args(config: dict, num_labeled_instances: int):
    batch_size = config["batch_size"]
    num_batches = ceil(num_labeled_instances / batch_size)
    max_steps = ((num_batches * 100 + 1000) // 1000) * 1000
    return Seq2SeqTrainingArguments(
        output_dir="output_commongen",
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        logging_steps=100,
        save_total_limit=2,
        # optimization args, the trainer uses the Adam optimizer
        # and has a linear warmup for the learning rate
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=max(1, ceil(128 / batch_size)),
        learning_rate=config["lr"],
        max_steps=max_steps,
        warmup_steps=config["warmup_steps"],
        # misc args
        report_to="none",  # "wandb"
        seed=42,
        disable_tqdm=False,
        load_best_model_at_end=True,
        metric_for_best_model="rouge2",
        # generation
        predict_with_generate=True,
        generation_max_length=256,
        include_inputs_for_metrics=True
    )


def train(
    model_init,
    labeled_set,
    unlabeled_set,
    dev_set,
    test_set,
    tokenizer,
    config,
    compute_metrics_fn,
    n_clusters: int,
    num_al_iterations=50,
    step_size=1024,
    strategy: str = "cluster"
):
    results_path = (Path("results") / strategy)
    results_path.mkdir(exist_ok=True)
    results_file = results_path / f"results_{config['seed']}.csv"
    # Todo: remove hard-code
    generation_kwargs = {
        "max_new_tokens": 30,
        "min_new_tokens": 3,
        "early_stopping": True,
    }
    device = config["device"]
    get_clusters = build_cluster_getter(config)
    train_args = get_train_args(config, len(labeled_set))
    trainer = Seq2SeqTrainer(
        model_init=model_init,
        args=train_args,
        train_dataset=labeled_set.remove_columns(["id", "cluster"]),
        eval_dataset=dev_set.remove_columns(["cluster"]),
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
    )
    print(trainer.evaluate(dev_set.remove_columns(["cluster"])))
    trainer.train()
    print(trainer.evaluate(dev_set.remove_columns(["cluster"])))

    evaluate_df = pd.DataFrame(
        columns=[
          "eval_loss",
          "eval_rouge1",
          "eval_rouge2",
          "eval_rougeL",
          "eval_rougeLsum",
          "eval_cola_score",
          "eval_cola_score_rel",
        ]
    )
    evaluate_results = trainer.evaluate(test_set, **generation_kwargs)
    evaluate_results = [evaluate_results[x] for x in evaluate_df.columns]
    evaluate_df.loc[len(evaluate_df)] = evaluate_results
    evaluate_df.to_csv(results_file, index=0)
    print("=" * 50, "\n", evaluate_df, "\n", "=" * 50, sep="")

    active_clusters = [
        n_clusters,
    ]

    for _ in tqdm(range(num_al_iterations)):
        if strategy != "random":
            probs = torch.zeros(unlabeled_set.shape[0]).to(device)
            unlabeled_clusters = torch.tensor(unlabeled_set["cluster"]).to(device)

            cur_clusters = get_clusters(trainer, dev_set, config).to(device)
            active_clusters.append(len(cur_clusters))

            for cluster in cur_clusters:
                probs[unlabeled_clusters == cluster] = 1

            probs /= probs.sum()
            distr = Categorical(probs)
            new_indices = distr.sample((step_size,))
            new_set = unlabeled_set.select(new_indices.cpu().numpy())
            sampled_id = new_set["id"]
            labeled_set = concatenate_datasets([labeled_set, new_set])
            unlabeled_set = unlabeled_set.filter(lambda x: x not in sampled_id)
        else:
            split = unlabeled_set.train_test_split(test_size=step_size, seed=config["seed"])
            unlabeled_set = split["train"]
            labeled_set = concatenate_datasets([labeled_set, split["test"]])

        # step = trainer.state.log_history[-1]["step"]
        train_args = get_train_args(config, len(labeled_set))
        trainer = Seq2SeqTrainer(
            model_init=model_init,
            args=train_args,
            train_dataset=labeled_set.remove_columns(["id", "cluster"]),
            eval_dataset=dev_set.remove_columns(["cluster"]),
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_fn,
        )
        trainer.train()  # resume_from_checkpoint=f"{train_args.output_dir}/checkpoint-{step}"
        evaluate_results = trainer.evaluate(test_set, **generation_kwargs)
        evaluate_results = [evaluate_results[x] for x in evaluate_df.columns]
        evaluate_df.loc[len(evaluate_df)] = evaluate_results
        evaluate_df.to_csv(results_file, index=0)
        print("=" * 50, "\n" * 3, evaluate_df, "\n", "=" * 50, "\n" * 3, sep="")
    return trainer, active_clusters


def run(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    device = config["device"]
    strategy = config["strategy"]
    seed = config["seed"]

    to_pad = strategy == 'uncertainty'

    train_set, test_set = load_ds(config)
    if "id" not in train_set.column_names:
        train_set = train_set.add_column("id", range(len(train_set)))
    train_data_tokenized = tokenize(train_set, tokenizer, config)
    test_data_tokenized = tokenize(test_set, tokenizer, config)
    compute_metrics_fn = partial(compute_metrics, tokenizer=tokenizer)

    # Initialization
    if strategy != "random":
        data_clusters = torch.tensor(
            np.load(config["clusters_path"])
            if config["np_clusters"]
            else torch.load(config["clusters_path"])
        ).to(device)

        train_set = shuffle_and_concatenate(train_data_tokenized, data_clusters, seed)

        if config["remove_outliers"]:
            train_set = remove_outliers(train_set)
        train_set, dev_set = make_new_set(train_set, config["dev_size_per_cluster"])
        unlabeled_set, labeled_set = make_new_set(
            train_set, config["init_size_per_cluster"]
        )
    else:
        data_clusters = []
        train_data_tokenized = train_data_tokenized.add_column(
            "cluster", [0 for _ in range(len(train_data_tokenized))]
        )
        split = train_data_tokenized.train_test_split(
            test_size=config["dev_size"], seed=seed, shuffle=True
        )
        train_set, dev_set = split["train"], split["test"]
        split = train_set.train_test_split(
            test_size=config["init_size"], seed=seed, shuffle=True
        )
        unlabeled_set, labeled_set = split["train"], split["test"]
    dev_set = dev_set.remove_columns(["id"])

    def model_init():
        return AutoModelForSeq2SeqLM.from_pretrained(config["model"]).to(device)

    trainer, active_clusters = train(
        model_init=model_init,
        labeled_set=labeled_set,
        unlabeled_set=unlabeled_set,
        dev_set=dev_set,
        compute_metrics_fn=compute_metrics_fn,
        config=config,
        test_set=test_data_tokenized,
        tokenizer=tokenizer,
        num_al_iterations=config["num_al_iterations"],
        step_size=config["step_size"],
        n_clusters=len(set(data_clusters)),
        strategy=strategy
    )

    dir_name = config["exp_name"] + "_data"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    dump(trainer.state, dir_name + "/trainer.pkl")

    dump(active_clusters, dir_name + "/active_clusters.pkl")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="seed", required=False)
    parser.add_argument("--strategy", default="cluster", type=str, required=False)
    args = parser.parse_args()
    
    with open("config.yaml") as f:
        config = yaml.load(f)
    config["seed"] = args.seed
    config["strategy"] = args.strategy
    
    run(config)
