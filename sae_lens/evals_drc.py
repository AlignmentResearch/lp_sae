import argparse
import re
from functools import partial
from typing import Any, Mapping

import einops
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens.hook_points import HookedRootModule

from sae_lens.sae import SAE
from sae_lens.training.activations_store import ActivationsStore, DRCActivationsStore
from sae_lens.evals import EvalConfig, get_eval_everything_config
from learned_planners.interp.plot import save_video_sae as save_video_lp
from learned_planners.interp.utils import play_level, run_fn_with_cache, load_policy

from cleanba.environments import BoxobanConfig, EnvpoolBoxobanConfig
import os
import pathlib
import wandb
import concurrent.futures
import multiprocessing

@torch.no_grad()
def run_evals(
    sae: SAE,
    activation_store: ActivationsStore,
    model: HookedRootModule,
    eval_config: EvalConfig = EvalConfig(),
    model_kwargs: Mapping[str, Any] = {},
    num_envs: int = 1,
    envpool: bool = True,
) -> dict[str, Any]:

    hook_name = sae.cfg.hook_name
    actual_batch_size = (
        eval_config.batch_size_prompts or activation_store.store_batch_size_prompts
    )

    # TODO: Come up with a cleaner long term strategy here for SAEs that do reshaping.
    # turn off hook_z reshaping mode if it's on, and restore it after evals
    if "hook_z" in hook_name:
        previous_hook_z_reshaping_mode = sae.hook_z_reshaping_mode
        sae.turn_off_forward_pass_hook_z_reshaping()
    else:
        previous_hook_z_reshaping_mode = None

    metrics = get_recons_loss(
        sae,
        model,
        activation_store,
        compute_kl=eval_config.compute_kl,
        compute_ce_loss=eval_config.compute_ce_loss,
        model_kwargs=model_kwargs,
        num_envs=num_envs,
        envpool=envpool,
    )

    #     assert eval_config.n_eval_reconstruction_batches > 0
    #     metrics |= get_downstream_reconstruction_metrics(
    #         sae,
    #         model,
    #         activation_store,
    #         compute_kl=eval_config.compute_kl,
    #         compute_ce_loss=eval_config.compute_ce_loss,
    #         n_batches=eval_config.n_eval_reconstruction_batches,
    #         eval_batch_size_prompts=actual_batch_size,
    #     )

    #     activation_store.reset_input_dataset()

    # if (
    #     eval_config.compute_l2_norms
    #     or eval_config.compute_sparsity_metrics
    #     or eval_config.compute_variance_metrics
    # ):
    #     assert eval_config.n_eval_sparsity_variance_batches > 0
    #     metrics |= get_sparsity_and_variance_metrics(
    #         sae,
    #         model,
    #         activation_store,
    #         compute_l2_norms=eval_config.compute_l2_norms,
    #         compute_sparsity_metrics=eval_config.compute_sparsity_metrics,
    #         compute_variance_metrics=eval_config.compute_variance_metrics,
    #         n_batches=eval_config.n_eval_sparsity_variance_batches,
    #         eval_batch_size_prompts=actual_batch_size,
    #         model_kwargs=model_kwargs,
    #     )

    if len(metrics) == 0:
        raise ValueError(
            "No metrics were computed, please set at least one metric to True."
        )

    # restore previous hook z reshaping mode if necessary
    if "hook_z" in hook_name:
        if previous_hook_z_reshaping_mode and not sae.hook_z_reshaping_mode:
            sae.turn_on_forward_pass_hook_z_reshaping()
        elif not previous_hook_z_reshaping_mode and sae.hook_z_reshaping_mode:
            sae.turn_off_forward_pass_hook_z_reshaping()

    # total_tokens_evaluated = (
    #     activation_store.context_size
    #     * eval_config.n_eval_reconstruction_batches
    #     * actual_batch_size
    # )

    return metrics


# def get_downstream_reconstruction_metrics(
#     sae: SAE,
#     model: HookedRootModule,
#     activation_store: ActivationsStore,
#     compute_kl: bool,
#     compute_ce_loss: bool,
#     n_batches: int,
#     eval_batch_size_prompts: int,
# ):
#     metrics_dict = {}
#     if compute_kl:
#         metrics_dict["kl_div_with_sae"] = []
#         metrics_dict["kl_div_with_ablation"] = []
#     if compute_ce_loss:
#         metrics_dict["ce_loss_with_sae"] = []
#         metrics_dict["ce_loss_without_sae"] = []
#         metrics_dict["ce_loss_with_ablation"] = []



#     for _ in range(n_batches):
#         batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
#         for metric_name, metric_value in get_recons_loss(
#             sae,
#             model,
#             batch_tokens,
#             activation_store,
#             compute_kl=compute_kl,
#             compute_ce_loss=compute_ce_loss,
#         ).items():
#             metrics_dict[metric_name].append(metric_value)

#     metrics: dict[str, float] = {}
#     for metric_name, metric_values in metrics_dict.items():
#         metrics[f"metrics/{metric_name}"] = torch.stack(metric_values).mean().item()

#     if compute_kl:
#         metrics["metrics/kl_div_score"] = (
#             metrics["metrics/kl_div_with_ablation"] - metrics["metrics/kl_div_with_sae"]
#         ) / metrics["metrics/kl_div_with_ablation"]

#     if compute_ce_loss:
#         metrics["metrics/ce_loss_score"] = (
#             metrics["metrics/ce_loss_with_ablation"]
#             - metrics["metrics/ce_loss_with_sae"]
#         ) / (
#             metrics["metrics/ce_loss_with_ablation"]
#             - metrics["metrics/ce_loss_without_sae"]
#         )

#     return metrics


# def get_sparsity_and_variance_metrics(
#     sae: SAE,
#     model: HookedRootModule,
#     activation_store: ActivationsStore,
#     n_batches: int,
#     compute_l2_norms: bool,
#     compute_sparsity_metrics: bool,
#     compute_variance_metrics: bool,
#     eval_batch_size_prompts: int,
#     model_kwargs: Mapping[str, Any],
# ):

#     hook_name = sae.cfg.hook_name
#     hook_head_index = sae.cfg.hook_head_index

#     metric_dict = {}
#     if compute_l2_norms:
#         metric_dict["l2_norm_in"] = []
#         metric_dict["l2_norm_out"] = []
#         metric_dict["l2_ratio"] = []
#     if compute_sparsity_metrics:
#         metric_dict["l0"] = []
#         metric_dict["l1"] = []
#     if compute_variance_metrics:
#         metric_dict["explained_variance"] = []
#         metric_dict["mse"] = []

#     for _ in range(n_batches):
#         batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)

#         # get cache
#         _, cache = model.run_with_cache(
#             batch_tokens,
#             prepend_bos=False,
#             names_filter=[hook_name],
#             **model_kwargs,
#         )

#         # we would include hook z, except that we now have base SAE's
#         # which will do their own reshaping for hook z.
#         has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
#         if hook_head_index is not None:
#             original_act = cache[hook_name][:, :, hook_head_index]
#         elif any(substring in hook_name for substring in has_head_dim_key_substrings):
#             original_act = cache[hook_name].flatten(-2, -1)
#         else:
#             original_act = cache[hook_name]

#         # normalise if necessary (necessary in training only, otherwise we should fold the scaling in)
#         if activation_store.normalize_activations == "expected_average_only_in":
#             original_act = activation_store.apply_norm_scaling_factor(original_act)

#         # send the (maybe normalised) activations into the SAE
#         sae_feature_activations = sae.encode(original_act.to(sae.device))
#         sae_out = sae.decode(sae_feature_activations).to(original_act.device)
#         del cache

#         if activation_store.normalize_activations == "expected_average_only_in":
#             sae_out = activation_store.unscale(sae_out)

#         flattened_sae_input = einops.rearrange(original_act, "b ctx d -> (b ctx) d")
#         flattened_sae_feature_acts = einops.rearrange(
#             sae_feature_activations, "b ctx d -> (b ctx) d"
#         )
#         flattened_sae_out = einops.rearrange(sae_out, "b ctx d -> (b ctx) d")

#         if compute_l2_norms:
#             l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
#             l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
#             l2_norm_in_for_div = l2_norm_in.clone()
#             l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
#             l2_norm_ratio = l2_norm_out / l2_norm_in_for_div
#             metric_dict["l2_norm_in"].append(l2_norm_in)
#             metric_dict["l2_norm_out"].append(l2_norm_out)
#             metric_dict["l2_ratio"].append(l2_norm_ratio)

#         if compute_sparsity_metrics:
#             l0 = (flattened_sae_feature_acts > 0).sum(dim=-1).float()
#             l1 = flattened_sae_feature_acts.sum(dim=-1)
#             metric_dict["l0"].append(l0)
#             metric_dict["l1"].append(l1)

#         if compute_variance_metrics:
#             resid_sum_of_squares = (
#                 (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)
#             )
#             total_sum_of_squares = (
#                 (flattened_sae_input - flattened_sae_input.mean(dim=0)).pow(2).sum(-1)
#             )
#             explained_variance = 1 - resid_sum_of_squares / total_sum_of_squares
#             metric_dict["explained_variance"].append(explained_variance)
#             metric_dict["mse"].append(resid_sum_of_squares)

#     metrics: dict[str, float] = {}
#     for metric_name, metric_values in metric_dict.items():
#         metrics[f"metrics/{metric_name}"] = torch.stack(metric_values).mean().item()

#     return metrics


# TODO(tomMcGrath): the rescaling below is a bit of a hack and could probably be tidied up
def sae_replacement_hook(activations: torch.Tensor, hook: Any, sae, activation_store):
    original_device = activations.device
    activations = activations.to(sae.device)
    if activation_store.grid_wise:
        original_shape = activations.shape
        d_in_idx = activations.shape.index(sae.cfg.d_in)
        # take d_in_idx as the last dimension
        activations = activations.permute(*([i for i in range(activations.ndim) if i != d_in_idx] + [d_in_idx]))
    else:
        raise NotImplementedError("Only grid_wise activations are supported for now")

    # Handle rescaling if SAE expects it
    if activation_store.normalize_activations == "expected_average_only_in":
        activations = activation_store.apply_norm_scaling_factor(activations)

    # SAE class agnost forward forward pass.
    activations = sae.decode(sae.encode(activations)).to(activations.dtype)

    # Unscale if activations were scaled prior to going into the SAE
    if activation_store.normalize_activations == "expected_average_only_in":
        activations = activation_store.unscale(activations)

    # reverse the permutation
    if activation_store.grid_wise:
        new_permutation = list(range(activations.ndim - 1))
        new_permutation.insert(d_in_idx, activations.ndim - 1)
        activations = activations.permute(*new_permutation)
        assert activations.shape == original_shape, f"Expected {original_shape}, got {activations.shape}"
    return activations.to(original_device)


def save_video(sae_feature_activations, original_obs, sae_cfg, num_envs, lengths, num_features_to_show=45):
    act_freq = (sae_feature_activations > 0).sum(dim=0)
    assert act_freq.shape == (sae_cfg.d_sae,)
    assert sae_feature_activations.shape == (len(original_obs) * num_envs * 10 * 10, sae_cfg.d_sae)
    sae_feature_reshaped = sae_feature_activations.reshape(len(original_obs), num_envs, 10, 10, -1)
    sae_acts = sae_feature_reshaped[:lengths[0], 0].permute(3, 0, 1, 2)
    top_activating_features = torch.argsort(act_freq, descending=True)
    topkfeatures = 15
    step = wandb.run.step
    videos_dict = {}

    multiprocessing.set_start_method("spawn", force=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(4, num_features_to_show // topkfeatures)) as executor:
        futures = [
            executor.submit(
                save_video_lp,
                f"top_activating_features_step-{step}_start-{feature_start_idx}.mp4",
                original_obs[:lengths[0], 0],
                sae_acts[top_activating_features[feature_start_idx : feature_start_idx + topkfeatures]],
                base_dir=wandb.run.dir + "/local-files/videos",
                sae_feature_indices=top_activating_features[feature_start_idx : feature_start_idx + topkfeatures],
            )
            for feature_start_idx in range(0, num_features_to_show, topkfeatures)
        ]
        for future in concurrent.futures.as_completed(futures):
            video_path = future.result()
            feature_start_idx = re.search(r"start-(\d+)", video_path).group(1)
            videos_dict[f"videos/top_activating_features_start_{feature_start_idx}"] = wandb.Video(video_path, format="mp4")
    return videos_dict


@torch.no_grad()
def get_recons_loss(
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    compute_kl: bool,
    compute_ce_loss: bool,
    model_kwargs: Mapping[str, Any] = {},
    num_envs: int = 1,
    envpool: bool = False,
) -> dict[str, Any]:
    hook_name = sae.cfg.hook_name
    head_index = sae.cfg.hook_head_index

    on_cluster = os.path.exists("/training")
    LP_DIR = pathlib.Path(__file__).parent.parent.parent.parent
    if on_cluster:
        BOXOBAN_CACHE = pathlib.Path("/training/.sokoban_cache/")
    else:
        BOXOBAN_CACHE = pathlib.Path(__file__).parent.parent.parent / "training/.sokoban_cache/"

    env_kwargs = dict(
        cache_path=BOXOBAN_CACHE,
        num_envs=num_envs,
        max_episode_steps=120,
        min_episode_steps=120,
        difficulty="medium",
        split="valid",
    )
    if envpool:
        env_cls = EnvpoolBoxobanConfig
        env_kwargs.update(load_sequentially=True)
    else:
        env_cls = BoxobanConfig
        env_kwargs.update(asynchronous=False, tinyworld_obs=True)

    boxo_cfg = env_cls(**env_kwargs)
    boxo_env = boxo_cfg.make()

    hook_name_last_int_pos = hook_name + ".0.2"
    out = play_level(
        boxo_env,
        model,
        max_steps=100,
        fwd_hooks=None,
        names_filter=[hook_name_last_int_pos],
    )
    original_obs, original_logits, cache, lengths = out.obs, out.logits, out.cache, out.lengths
    # we would include hook z, except that we now have base SAE's
    # which will do their own reshaping for hook z.
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if head_index is not None:
        original_act = cache[hook_name_last_int_pos][:, :, head_index]
    elif any(substring in hook_name for substring in has_head_dim_key_substrings):
        original_act = cache[hook_name_last_int_pos].flatten(-2, -1)
    else:
        original_act = cache[hook_name_last_int_pos]
    original_act = activation_store.acts_ds.process_cache_for_sae(original_act, grid_wise=activation_store.grid_wise)

    # normalise if necessary (necessary in training only, otherwise we should fold the scaling in)
    if activation_store.normalize_activations == "expected_average_only_in":
        original_act = activation_store.apply_norm_scaling_factor(original_act)

    # send the (maybe normalised) activations into the SAE
    sae_feature_activations = sae.encode(original_act.to(sae.device))

    video_dict = save_video(sae_feature_activations.cpu(), original_obs.cpu(), sae.cfg, num_envs, lengths)

    sae_out = sae.decode(sae_feature_activations).to(original_act.device)
    del cache

    if activation_store.normalize_activations == "expected_average_only_in":
        sae_out = activation_store.unscale(sae_out)
    if len(original_act.shape) == 3:
        flattened_sae_input = einops.rearrange(original_act, "b ctx d -> (b ctx) d")
        flattened_sae_feature_acts = einops.rearrange(
            sae_feature_activations, "b ctx d -> (b ctx) d"
        )
        flattened_sae_out = einops.rearrange(sae_out, "b ctx d -> (b ctx) d")
    else:
        flattened_sae_input = original_act
        flattened_sae_feature_acts = sae_feature_activations
        flattened_sae_out = sae_out

    compute_l2_norms, compute_sparsity_metrics, compute_variance_metrics = True, True, True
    metric_dict = {k: [] for k in ["l2_norm_in", "l2_norm_out", "l2_ratio", "l0", "l1", "explained_variance", "mse"]}
    if compute_l2_norms:
        l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
        l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
        l2_norm_in_for_div = l2_norm_in.clone()
        l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
        l2_norm_ratio = l2_norm_out / l2_norm_in_for_div
        metric_dict["l2_norm_in"].append(l2_norm_in)
        metric_dict["l2_norm_out"].append(l2_norm_out)
        metric_dict["l2_ratio"].append(l2_norm_ratio)

    if compute_sparsity_metrics:
        l0 = (flattened_sae_feature_acts > 0).sum(dim=-1).float()
        l1 = flattened_sae_feature_acts.sum(dim=-1)
        metric_dict["l0"].append(l0)
        metric_dict["l1"].append(l1)

    if compute_variance_metrics:
        resid_sum_of_squares = (
            (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)
        )
        total_sum_of_squares = (
            (flattened_sae_input - flattened_sae_input.mean(dim=0)).pow(2).sum(-1)
        )
        explained_variance = 1 - resid_sum_of_squares / total_sum_of_squares
        metric_dict["explained_variance"].append(explained_variance)
        metric_dict["mse"].append(resid_sum_of_squares)

    metrics: dict[str, float] = {**video_dict}
    for metric_name, metric_values in metric_dict.items():
        metrics[f"metrics/{metric_name}"] = torch.stack(metric_values).mean().item()

    standard_replacement_hook = partial(sae_replacement_hook, sae=sae, activation_store=activation_store)

    def standard_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations = torch.zeros_like(activations)
        return activations.to(original_device)

    def single_head_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations[:, :, head_index] = torch.zeros_like(activations[:, :, head_index])
        return activations.to(original_device)

    # we would include hook z, except that we now have base SAE's
    # which will do their own reshaping for hook z.
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if any(substring in hook_name for substring in has_head_dim_key_substrings):
        if head_index is None:
            replacement_hook = all_head_replacement_hook
            zero_ablate_hook = standard_zero_ablate_hook
        else:
            replacement_hook = single_head_replacement_hook
            zero_ablate_hook = single_head_zero_ablate_hook
    else:
        replacement_hook = standard_replacement_hook
        zero_ablate_hook = standard_zero_ablate_hook

    N = original_obs.shape[1]
    state = model.recurrent_initial_state(N, device=model.device)
    eps_start = torch.zeros(original_obs.shape[:2], dtype=torch.bool, device=model.device)
    all_hook_names = [hook_name + f".{pos}.{int_pos}" for pos in range(len(original_obs)) for int_pos in range(3)]

    fwd_hooks = [(pos_hook_name, replacement_hook) for pos_hook_name in all_hook_names]
    (distribution, _), _ = run_fn_with_cache(
        model,
        "get_distribution",
        original_obs,
        state,
        eps_start,
        # return_repeats=False,
        fwd_hooks=fwd_hooks,
    )
    recons_logits = distribution.distribution.logits

    fwd_hooks = [(pos_hook_name, zero_ablate_hook) for pos_hook_name in all_hook_names]
    (distribution, _), _ = run_fn_with_cache(
        model,
        "get_distribution",
        original_obs,
        state,
        eps_start,
        # return_repeats=False,
        fwd_hooks=fwd_hooks,
    )
    zero_abl_logits = distribution.distribution.logits

    def kl(original_logits: torch.Tensor, new_logits: torch.Tensor):
        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        log_original_probs = torch.log(original_probs)
        # new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
        # log_new_probs = torch.log(new_probs)
        log_new_probs = torch.nn.functional.log_softmax(new_logits, dim=-1)
        kl_div = original_probs * (log_original_probs - log_new_probs)
        kl_div = kl_div.sum(dim=-1)
        return kl_div

    if compute_kl:
        recons_kl_div = kl(original_logits, recons_logits)
        zero_abl_kl_div = kl(original_logits, zero_abl_logits)
        metrics["metrics/kl_div_with_sae"] = recons_kl_div.mean().item()
        metrics["metrics/kl_div_with_ablation"] = zero_abl_kl_div.mean().item()
        metrics["metrics/kl_div_score"] = ((zero_abl_kl_div - recons_kl_div) / zero_abl_kl_div).mean().item()

    metrics["metrics/total_tokens_evaluated"] = boxo_cfg.num_envs * boxo_cfg.max_episode_steps
    return metrics


# def all_loadable_saes() -> list[tuple[str, str, float, float]]:
#     all_loadable_saes = []
#     saes_directory = get_pretrained_saes_directory()
#     for release, lookup in tqdm(saes_directory.items()):
#         for sae_name in lookup.saes_map.keys():
#             expected_var_explained = lookup.expected_var_explained[sae_name]
#             expected_l0 = lookup.expected_l0[sae_name]
#             all_loadable_saes.append(
#                 (release, sae_name, expected_var_explained, expected_l0)
#             )

#     return all_loadable_saes

import glob

def all_loadable_saes() -> list[str]:
    base_path = "/training/TrainSAEConfig/01-train-sae-on-hard-set/wandb/" # run-20240830_024028-x2jh4zdf/local-files/checkpoint/final_30007296
    run_dirs = glob.glob(base_path + "run-*")
    all_loadable_saes = []
    for run_dir in run_dirs:
        checkpoint_path = run_dir + "/local-files/checkpoint/*"
        checkpoints = glob.glob(checkpoint_path)
        for checkpoint in checkpoints:
            all_loadable_saes.append(checkpoint)


def multiple_evals() -> pd.DataFrame:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sae_regex_compiled = re.compile(sae_regex_pattern)
    sae_block_compiled = re.compile(sae_block_pattern)
    all_saes = all_loadable_saes()
    filtered_saes = [
        sae
        for sae in all_saes
        if sae_regex_compiled.fullmatch(sae[0]) and sae_block_compiled.fullmatch(sae[1])
    ]

    assert len(filtered_saes) > 0, "No SAEs matched the given regex patterns"

    eval_results = []

    eval_config = get_eval_everything_config(
        batch_size_prompts=eval_batch_size_prompts,
        n_eval_reconstruction_batches=num_eval_batches,
        n_eval_sparsity_variance_batches=num_eval_batches,
    )

    current_model = load_policy()
    print(filtered_saes)
    for path in tqdm(filtered_saes):

        sae = SAE.load_from_pretrained(path, device=device)

        activation_store = DRCActivationsStore.from_config(current_model, sae.cfg)

        eval_metrics = run_evals(
            sae=sae,
            activation_store=activation_store,
            model=current_model,
            eval_config=eval_config,
        )
        eval_metrics["sae_path"] = path
        eval_results.append(eval_metrics)

    return pd.DataFrame(eval_results)


if __name__ == "__main__":

    # Example commands:
    # python sae_lens/evals.py "gpt2-small-res-jb.*" "blocks.8.hook_resid_pre.*" --save_path "gpt2_small_jb_layer8_resid_pre_eval_results.csv"
    # python sae_lens/evals.py "gpt2-small.*" "blocks.8.hook_resid_pre.*" --save_path "gpt2_small_layer8_resid_pre_eval_results.csv"
    # python sae_lens/evals.py "gpt2-small.*" ".*" --save_path "gpt2_small_eval_results.csv"
    # python sae_lens/evals.py "mistral.*" ".*" --save_path "mistral_eval_results.csv"

    arg_parser = argparse.ArgumentParser(description="Run evaluations on SAEs")
    arg_parser.add_argument(
        "sae_regex_pattern",
        type=str,
        help="Regex pattern to match SAE names. Can be an entire SAE name to match a specific SAE.",
    )
    arg_parser.add_argument(
        "sae_block_pattern",
        type=str,
        help="Regex pattern to match SAE block names. Can be an entire block name to match a specific block.",
    )
    arg_parser.add_argument(
        "--num_eval_batches",
        type=int,
        default=10,
        help="Number of evaluation batches to run.",
    )
    arg_parser.add_argument(
        "--eval_batch_size_prompts",
        type=int,
        default=8,
        help="Batch size for evaluation prompts.",
    )
    arg_parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Skylion007/openwebtext", "lighteval/MATH"],
        help="Datasets to evaluate on.",
    )
    arg_parser.add_argument(
        "--ctx_lens",
        nargs="+",
        default=[64, 128, 256, 512],
        help="Context lengths to evaluate on.",
    )
    arg_parser.add_argument(
        "--save_path",
        type=str,
        default="eval_results.csv",
        help="Path to save evaluation results to.",
    )

    args = arg_parser.parse_args()

    eval_results = multiple_evals(
        sae_regex_pattern=args.sae_regex_pattern,
        sae_block_pattern=args.sae_block_pattern,
        num_eval_batches=args.num_eval_batches,
        eval_batch_size_prompts=args.eval_batch_size_prompts,
        datasets=args.datasets,
        ctx_lens=args.ctx_lens,
    )

    eval_results.to_csv(args.save_path, index=False)
