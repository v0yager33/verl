# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def _compute_vcpo_loss_in_ppo(
    vcpo_model: torch.nn.Module,
    config: "ActorConfig",
    data: TensorDict,
    metric_aggregation,
) -> tuple:
    """Compute VCPO loss inline during PPO training.

    Performs gradient attribution on the current model to compute visual causal
    alignment loss. Supports both SA (Self-Alignment) and SD (Self-Distillation) modes.

    Returns:
        Tuple of (vcpo_loss tensor or None, metrics dict).
    """
    from verl.workers.utils.vcpo import (
        compute_causal_probability,
        compute_gradient_attribution,
        compute_vcpo_kl_loss,
        find_answer_token_positions,
        resolve_image_token_id,
    )

    metrics = {}
    tokenizer = tu.get(data, key="tokenizer", default=None)
    if tokenizer is None:
        logger.warning("VCPO: tokenizer not found in data, skipping VCPO loss")
        return None, metrics

    vcpo_mode = config.vcpo_mode
    vcpo_temperature = config.vcpo_temperature
    vcpo_answer_tag = config.vcpo_answer_tag

    # Unwrap FSDP to get the raw model for get_input_embeddings()
    raw_model = getattr(vcpo_model, "_fsdp_wrapped_module", vcpo_model)
    raw_model = getattr(raw_model, "module", raw_model)

    image_token_id = resolve_image_token_id(raw_model, tokenizer, config.vcpo_image_token)

    # Use padded tensors saved before left_right_2_no_padding conversion
    input_ids = data.get("vcpo_input_ids", None)
    attention_mask = data.get("vcpo_attention_mask", None)
    position_ids = data.get("vcpo_position_ids", None)
    responses = data.get("responses", None)

    if input_ids is None or attention_mask is None or position_ids is None or responses is None:
        logger.warning("VCPO: required padded tensors not found in data, skipping")
        return None, metrics

    batch_size = input_ids.shape[0]
    response_length = responses.shape[1]

    image_token_mask = (input_ids == image_token_id)
    num_image_tokens = image_token_mask.sum().item()
    metrics["actor/vcpo_num_image_tokens"] = Metric(
        value=float(num_image_tokens), aggregation=AggregationType.MEAN
    )

    if not image_token_mask.any():
        metrics["actor/vcpo_skip_reason"] = Metric(value=1.0, aggregation=AggregationType.MEAN)
        return None, metrics

    answer_positions_list = find_answer_token_positions(responses, tokenizer, vcpo_answer_tag)

    # Get multi-modal inputs from data
    multi_modal_inputs_array = tu.get(data, key="multi_modal_inputs", default=None)

    def get_sample_multi_modal_inputs(idx):
        if multi_modal_inputs_array is not None:
            try:
                item = multi_modal_inputs_array[idx]
                if hasattr(item, "data"):
                    item = item.data
                return item if isinstance(item, dict) else {}
            except (IndexError, TypeError):
                pass
        return {}

    if vcpo_mode == "sa":
        vcpo_loss_val, mode_metrics = _compute_vcpo_sa_loss_inline(
            model=vcpo_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            responses=responses,
            response_length=response_length,
            answer_positions_list=answer_positions_list,
            image_token_mask=image_token_mask,
            batch_size=batch_size,
            vcpo_temperature=vcpo_temperature,
            data=data,
            get_multi_modal_inputs_fn=get_sample_multi_modal_inputs,
        )
        metrics.update(mode_metrics)
    elif vcpo_mode == "sd":
        vcpo_loss_val, mode_metrics = _compute_vcpo_sd_loss_inline(
            model=vcpo_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            responses=responses,
            response_length=response_length,
            answer_positions_list=answer_positions_list,
            image_token_mask=image_token_mask,
            batch_size=batch_size,
            vcpo_temperature=vcpo_temperature,
            tokenizer=tokenizer,
            vcpo_answer_tag=vcpo_answer_tag,
            image_token_id=image_token_id,
            data=data,
            get_multi_modal_inputs_fn=get_sample_multi_modal_inputs,
        )
        metrics.update(mode_metrics)
    else:
        logger.warning("VCPO: unknown mode '%s', skipping", vcpo_mode)
        return None, metrics

    if vcpo_loss_val is not None:
        vcpo_alpha = config.vcpo_alpha
        metrics["actor/vcpo_loss"] = Metric(value=vcpo_loss_val.detach(), aggregation=metric_aggregation)
        metrics["actor/vcpo_alpha"] = Metric(value=float(vcpo_alpha), aggregation=AggregationType.MEAN)

    return vcpo_loss_val, metrics


def _compute_vcpo_sa_loss_inline(
    model, input_ids, attention_mask, position_ids, responses, response_length,
    answer_positions_list, image_token_mask, batch_size, vcpo_temperature,
    data, get_multi_modal_inputs_fn,
) -> tuple:
    """Compute VCPO-SA loss using Self-Alignment (group consensus of correct samples)."""
    from verl.workers.utils.vcpo import (
        compute_causal_probability,
        compute_gradient_attribution,
        compute_vcpo_kl_loss,
    )

    metrics = {}
    vcpo_rewards = data.get("vcpo_rewards", None)
    uids = tu.get(data, key="uid", default=None)

    sample_attributions = {}
    for i in range(batch_size):
        if answer_positions_list[i] is None:
            continue
        grad_attr = compute_gradient_attribution(
            model=model,
            input_ids=input_ids[i:i+1],
            attention_mask=attention_mask[i:i+1],
            position_ids=(position_ids[i:i+1] if position_ids.dim() == 2
                          else position_ids[:, i:i+1, :]),
            response_length=response_length,
            answer_token_positions=answer_positions_list[i],
            image_token_mask=image_token_mask[i:i+1],
            multi_modal_inputs=get_multi_modal_inputs_fn(i),
            temperature=1.0,
        )
        if grad_attr is not None:
            sample_attributions[i] = grad_attr

    metrics["actor/vcpo_num_valid_samples"] = Metric(
        value=float(len(sample_attributions)), aggregation=AggregationType.MEAN
    )

    if len(sample_attributions) == 0:
        metrics["actor/vcpo_skip_reason"] = Metric(value=2.0, aggregation=AggregationType.MEAN)
        return None, metrics

    if uids is not None and vcpo_rewards is not None:
        uid_to_indices = {}
        for i in range(batch_size):
            uid_val = _extract_uid(uids, i)
            uid_to_indices.setdefault(uid_val, []).append(i)

        group_targets = {}
        for uid_val, indices in uid_to_indices.items():
            correct_attrs = []
            for idx in indices:
                if idx in sample_attributions and float(vcpo_rewards[idx]) == 1.0:
                    prob = compute_causal_probability(sample_attributions[idx], vcpo_temperature)
                    correct_attrs.append(prob)
            if correct_attrs:
                stacked = torch.stack(correct_attrs, dim=0)
                mean_attr = stacked.mean(dim=0)
                group_targets[uid_val] = compute_causal_probability(mean_attr, vcpo_temperature)

        kl_losses = []
        for i in range(batch_size):
            if i not in sample_attributions:
                continue
            uid_val = _extract_uid(uids, i)
            if uid_val not in group_targets:
                continue
            kl_loss = compute_vcpo_kl_loss(
                current_attribution=sample_attributions[i],
                target_attribution=group_targets[uid_val],
                temperature=vcpo_temperature,
            )
            kl_losses.append(kl_loss)

        metrics["actor/vcpo_num_kl_pairs"] = Metric(
            value=float(len(kl_losses)), aggregation=AggregationType.MEAN
        )
        if not kl_losses:
            metrics["actor/vcpo_skip_reason"] = Metric(value=3.0, aggregation=AggregationType.MEAN)
            return None, metrics
        return torch.stack(kl_losses).mean(), metrics
    else:
        all_attrs = [compute_causal_probability(attr, vcpo_temperature)
                     for attr in sample_attributions.values()]
        if len(all_attrs) < 2:
            return None, metrics
        target = torch.stack(all_attrs, dim=0).mean(dim=0)
        target = compute_causal_probability(target, vcpo_temperature)
        kl_losses = [compute_vcpo_kl_loss(attr, target, vcpo_temperature)
                     for attr in sample_attributions.values()]
        return torch.stack(kl_losses).mean(), metrics


def _compute_vcpo_sd_loss_inline(
    model, input_ids, attention_mask, position_ids, responses, response_length,
    answer_positions_list, image_token_mask, batch_size, vcpo_temperature,
    tokenizer, vcpo_answer_tag, image_token_id, data, get_multi_modal_inputs_fn,
) -> tuple:
    """Compute VCPO-SD loss using Self-Distillation (ground truth answer)."""
    from verl.workers.utils.vcpo import (
        compute_causal_probability,
        compute_gradient_attribution,
        compute_vcpo_kl_loss,
        find_answer_token_positions,
    )

    metrics = {}
    gt_input_ids = data.get("gt_input_ids", None)
    gt_attention_mask = data.get("gt_attention_mask", None)
    gt_position_ids = data.get("gt_position_ids", None)
    gt_response_length = tu.get(data, key="gt_response_length", default=None)

    if gt_input_ids is None:
        metrics["actor/vcpo_skip_reason"] = Metric(value=4.0, aggregation=AggregationType.MEAN)
        return None, metrics

    gt_image_token_mask = (gt_input_ids == image_token_id)
    gt_response_ids = gt_input_ids[:, -gt_response_length:]
    gt_answer_positions = find_answer_token_positions(gt_response_ids, tokenizer, vcpo_answer_tag)

    if gt_answer_positions[0] is None:
        metrics["actor/vcpo_skip_reason"] = Metric(value=5.0, aggregation=AggregationType.MEAN)
        return None, metrics

    gt_multi_modal_inputs = tu.get(data, key="gt_multi_modal_inputs", default=None)
    if gt_multi_modal_inputs is not None and hasattr(gt_multi_modal_inputs, "data"):
        gt_multi_modal_inputs = gt_multi_modal_inputs.data
    if gt_multi_modal_inputs is None:
        gt_multi_modal_inputs = get_multi_modal_inputs_fn(0)

    gt_grad_attr = compute_gradient_attribution(
        model=model,
        input_ids=gt_input_ids[0:1],
        attention_mask=gt_attention_mask[0:1],
        position_ids=(gt_position_ids[0:1] if gt_position_ids.dim() == 2
                      else gt_position_ids[:, 0:1, :]),
        response_length=gt_response_length,
        answer_token_positions=gt_answer_positions[0],
        image_token_mask=gt_image_token_mask[0:1],
        multi_modal_inputs=gt_multi_modal_inputs if isinstance(gt_multi_modal_inputs, dict) else {},
        temperature=1.0,
    )

    if gt_grad_attr is None:
        metrics["actor/vcpo_skip_reason"] = Metric(value=6.0, aggregation=AggregationType.MEAN)
        return None, metrics

    target_prob = compute_causal_probability(gt_grad_attr, vcpo_temperature)

    kl_losses = []
    for i in range(batch_size):
        if answer_positions_list[i] is None:
            continue
        grad_attr = compute_gradient_attribution(
            model=model,
            input_ids=input_ids[i:i+1],
            attention_mask=attention_mask[i:i+1],
            position_ids=(position_ids[i:i+1] if position_ids.dim() == 2
                          else position_ids[:, i:i+1, :]),
            response_length=response_length,
            answer_token_positions=answer_positions_list[i],
            image_token_mask=image_token_mask[i:i+1],
            multi_modal_inputs=get_multi_modal_inputs_fn(i),
            temperature=1.0,
        )
        if grad_attr is not None:
            kl_loss = compute_vcpo_kl_loss(grad_attr, target_prob, vcpo_temperature)
            kl_losses.append(kl_loss)

    metrics["actor/vcpo_num_valid_samples"] = Metric(
        value=float(len(kl_losses)), aggregation=AggregationType.MEAN
    )
    metrics["actor/vcpo_num_kl_pairs"] = Metric(
        value=float(len(kl_losses)), aggregation=AggregationType.MEAN
    )

    if not kl_losses:
        metrics["actor/vcpo_skip_reason"] = Metric(value=7.0, aggregation=AggregationType.MEAN)
        return None, metrics

    return torch.stack(kl_losses).mean(), metrics


def _extract_uid(uids, index: int) -> str:
    """Extract uid string from various uid container types."""
    if isinstance(uids, (list, tuple, np.ndarray)):
        val = uids[index]
    else:
        val = uids
    if hasattr(val, "data"):
        val = val.data
    return str(val)


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None, vcpo_model=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc.) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    # ============================================================================
    # add vcpo loss (computed inline via gradient attribution)
    if config.use_vcpo_loss and vcpo_model is not None:
        vcpo_loss_value, vcpo_metrics = _compute_vcpo_loss_in_ppo(
            vcpo_model=vcpo_model,
            config=config,
            data=data,
            metric_aggregation=metric_aggregation,
        )
        metrics.update(vcpo_metrics)
        if vcpo_loss_value is not None:
            vcpo_alpha = config.vcpo_alpha
            policy_loss = policy_loss + vcpo_alpha * vcpo_loss_value
    elif config.use_vcpo_loss and vcpo_model is None:
        logger.warning("VCPO: use_vcpo_loss=True but vcpo_model is None, skipping VCPO loss")
    # ============================================================================

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
