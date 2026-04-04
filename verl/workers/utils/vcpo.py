"""
VCPO (Visual Causal Policy Optimization) - Core Gradient Attribution Module

This module implements the deterministic gradient attribution protocol for
visual causal alignment in VLM reinforcement learning.
"""

import re
from typing import Optional

import torch
import torch.nn.functional as F


def find_answer_token_positions(
    response_ids: torch.Tensor,
    tokenizer,
    answer_tag: str = "boxed",
) -> list[Optional[list[int]]]:
    """Find the positions of answer tokens within each response sequence.

    For each response in the batch, decode the tokens, locate the answer
    region (e.g., content inside \\boxed{...} or <answer>...</answer>),
    and return the token-level positions corresponding to that region.

    Args:
        response_ids: (batch_size, response_length) token IDs of responses.
        tokenizer: HuggingFace tokenizer instance.
        answer_tag: Tag name to locate the answer. Supports:
            - "boxed": matches \\boxed{...} (LaTeX style)
            - other: matches <tag>...</tag> (XML style)

    Returns:
        List of length batch_size. Each element is either:
            - A list of integer positions (relative to response) for answer tokens
            - None if no answer is found in that response
    """
    batch_size = response_ids.shape[0]
    results = []

    for i in range(batch_size):
        token_ids = response_ids[i]
        # Decode the full response
        text = tokenizer.decode(token_ids, skip_special_tokens=False)

        # Find answer span in text
        if answer_tag == "boxed":
            # Match \boxed{...} with nested braces
            match = _find_boxed_content(text)
        else:
            # Match <tag>...</tag>
            pattern = f"<{answer_tag}>(.*?)</{answer_tag}>"
            m = re.search(pattern, text, re.DOTALL)
            match = m.group(1) if m else None

        if match is None:
            results.append(None)
            continue

        # Find the character-level start position of the answer content
        if answer_tag == "boxed":
            # Find the position of the matched content
            content_start = text.find(match)
        else:
            tag_match = re.search(f"<{answer_tag}>(.*?)</{answer_tag}>", text, re.DOTALL)
            content_start = tag_match.start(1)

        if content_start == -1:
            results.append(None)
            continue

        content_end = content_start + len(match)

        # Map character positions back to token positions
        answer_positions = _char_span_to_token_positions(
            token_ids, tokenizer, content_start, content_end
        )

        if len(answer_positions) == 0:
            results.append(None)
        else:
            results.append(answer_positions)

    return results


def _find_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    idx += len("\\boxed{")
    depth = 1
    start = idx
    while idx < len(text) and depth > 0:
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
        idx += 1
    if depth != 0:
        return None
    return text[start : idx - 1]


def _char_span_to_token_positions(
    token_ids: torch.Tensor,
    tokenizer,
    char_start: int,
    char_end: int,
) -> list[int]:
    """Map a character-level span to token-level positions.

    Decodes tokens one by one, tracking cumulative character offset,
    to find which tokens fall within [char_start, char_end).

    Args:
        token_ids: 1D tensor of token IDs.
        tokenizer: HuggingFace tokenizer.
        char_start: Start character index (inclusive).
        char_end: End character index (exclusive).

    Returns:
        List of token position indices that overlap with the character span.
    """
    positions = []
    current_char = 0

    for pos in range(token_ids.shape[0]):
        token_text = tokenizer.decode(token_ids[pos : pos + 1], skip_special_tokens=False)
        token_start = current_char
        token_end = current_char + len(token_text)

        # Check overlap between [token_start, token_end) and [char_start, char_end)
        if token_end > char_start and token_start < char_end:
            positions.append(pos)

        current_char = token_end

    return positions


def compute_gradient_attribution(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_length: int,
    answer_token_positions: list[int],
    image_token_mask: torch.Tensor,
    multi_modal_inputs: dict,
    temperature: float = 1.0,
) -> Optional[torch.Tensor]:
    """Compute gradient attribution of answer tokens w.r.t. visual token embeddings.

    Performs a forward pass, extracts the logits at answer token positions,
    and computes the L2 norm of gradients of those logits w.r.t. the visual
    token embeddings in the input layer.

    Mathematical formulation:
        g = || ∂Logit(y) / ∂X_v ||_2

    where X_v are the visual token embeddings and y is the answer token.
    If multiple answer tokens exist, their gradient norms are averaged.

    Args:
        model: The VLM model (must have get_input_embeddings()).
        input_ids: (1, seq_len) input token IDs for a single sample.
        attention_mask: (1, seq_len) attention mask.
        position_ids: (1, seq_len) or (num_dims, 1, seq_len) position IDs.
        response_length: Length of the response portion.
        answer_token_positions: List of positions (relative to response start)
            where answer tokens are located.
        image_token_mask: (1, seq_len) boolean mask, True for visual tokens.
        multi_modal_inputs: Dict of multi-modal inputs (pixel_values, etc.).
        temperature: Temperature for logit scaling.

    Returns:
        Gradient attribution vector of shape (num_visual_tokens,), or None if
        computation fails. Each element represents the causal sensitivity of
        the corresponding visual token.
    """
    model.eval()

    # Get the embedding layer
    embedding_layer = model.get_input_embeddings()

    # Compute input embeddings with gradient tracking
    with torch.enable_grad():
        input_embeds = embedding_layer(input_ids)
        input_embeds.requires_grad_(True)
        input_embeds.retain_grad()

        # Forward pass using embeddings instead of input_ids
        if position_ids.dim() == 3:
            pos_ids = position_ids.transpose(0, 1)
        else:
            pos_ids = position_ids

        output = model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=pos_ids,
            **multi_modal_inputs,
            use_cache=False,
        )

        logits = output.logits  # (1, seq_len, vocab_size)
        logits = logits / temperature

        # Extract logits at answer token positions (absolute positions in full sequence)
        prompt_length = input_ids.shape[1] - response_length
        absolute_positions = [prompt_length + p for p in answer_token_positions]

        # Get the logit of the actual next token at each answer position
        # For autoregressive models, logit at position t predicts token at t+1
        # So we use the logit at position (answer_pos - 1) to get the prediction for answer_pos
        target_logits = []
        for pos in absolute_positions:
            if pos > 0 and pos < input_ids.shape[1]:
                # logit at pos-1 predicts token at pos
                token_id = input_ids[0, pos]
                logit_value = logits[0, pos - 1, token_id]
                target_logits.append(logit_value)

        if len(target_logits) == 0:
            return None

        # Average the target logits across answer tokens
        mean_logit = torch.stack(target_logits).mean()

        # Backward to get gradients w.r.t. input embeddings
        mean_logit.backward(retain_graph=False)

        if input_embeds.grad is None:
            return None

        # Extract gradients for visual tokens only
        # input_embeds.grad shape: (1, seq_len, hidden_dim)
        grad = input_embeds.grad[0]  # (seq_len, hidden_dim)

        # Get visual token positions
        visual_mask = image_token_mask[0]  # (seq_len,)
        visual_grad = grad[visual_mask]  # (num_visual_tokens, hidden_dim)

        # Compute L2 norm along hidden_dim → (num_visual_tokens,)
        gradient_attribution = torch.norm(visual_grad, p=2, dim=-1)

        # Detach and clean up
        gradient_attribution = gradient_attribution.detach()
        input_embeds.requires_grad_(False)
        model.zero_grad()

    return gradient_attribution


def compute_causal_probability(
    gradient_attribution: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Normalize gradient attribution into a causal probability distribution via Softmax.

    P(g) = Softmax(g / τ)

    This ensures the sum of visual contributions equals 1 regardless of
    sequence length, eliminating scale bias across different rollout lengths.

    Args:
        gradient_attribution: (num_visual_tokens,) raw gradient norms.
        temperature: τ, temperature coefficient for softmax.

    Returns:
        Causal probability distribution of shape (num_visual_tokens,).
    """
    return F.softmax(gradient_attribution / temperature, dim=0)


def compute_vcpo_sd_target(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_ids: torch.Tensor,
    gt_input_ids: torch.Tensor,
    gt_attention_mask: torch.Tensor,
    gt_position_ids: torch.Tensor,
    gt_response_length: int,
    image_token_mask: torch.Tensor,
    gt_image_token_mask: torch.Tensor,
    multi_modal_inputs: dict,
    gt_multi_modal_inputs: dict,
    tokenizer,
    answer_tag: str = "boxed",
    temperature: float = 1.0,
) -> Optional[torch.Tensor]:
    """Compute the VCPO-SD (Self-Distillation) target attribution.

    Uses the ground truth answer A_GT to perform a forward pass and extract
    the "god's eye view" attribution map G_SD.

    Process:
        1. Construct input with ground truth answer
        2. Find answer token positions in GT response
        3. Compute gradient attribution using GT forward pass
        4. Normalize to probability distribution

    Args:
        model: The VLM model.
        input_ids: Original input_ids (prompt + rollout response).
        attention_mask: Original attention mask.
        position_ids: Original position IDs.
        response_ids: Response token IDs from rollout.
        gt_input_ids: Input IDs with ground truth response (prompt + GT answer).
        gt_attention_mask: Attention mask for GT input.
        gt_position_ids: Position IDs for GT input.
        gt_response_length: Length of the GT response.
        image_token_mask: Visual token mask for original input.
        gt_image_token_mask: Visual token mask for GT input.
        multi_modal_inputs: Multi-modal inputs for original.
        gt_multi_modal_inputs: Multi-modal inputs for GT.
        tokenizer: Tokenizer instance.
        answer_tag: Tag to locate answer tokens.
        temperature: Softmax temperature.

    Returns:
        P_SD: Causal probability distribution (num_visual_tokens,), or None.
    """
    # Find answer positions in GT response
    gt_response_ids = gt_input_ids[:, -gt_response_length:]
    answer_positions_list = find_answer_token_positions(
        gt_response_ids, tokenizer, answer_tag
    )

    if answer_positions_list[0] is None:
        return None

    # Compute gradient attribution using GT
    grad_attr = compute_gradient_attribution(
        model=model,
        input_ids=gt_input_ids,
        attention_mask=gt_attention_mask,
        position_ids=gt_position_ids,
        response_length=gt_response_length,
        answer_token_positions=answer_positions_list[0],
        image_token_mask=gt_image_token_mask,
        multi_modal_inputs=gt_multi_modal_inputs,
        temperature=1.0,
    )

    if grad_attr is None:
        return None

    return compute_causal_probability(grad_attr, temperature)


def compute_vcpo_sa_target(
    model: torch.nn.Module,
    batch_input_ids: list[torch.Tensor],
    batch_attention_mask: list[torch.Tensor],
    batch_position_ids: list[torch.Tensor],
    batch_response_ids: list[torch.Tensor],
    batch_response_lengths: list[int],
    batch_rewards: list[float],
    batch_image_token_masks: list[torch.Tensor],
    batch_multi_modal_inputs: list[dict],
    tokenizer,
    answer_tag: str = "boxed",
    temperature: float = 1.0,
) -> Optional[torch.Tensor]:
    """Compute the VCPO-SA (Self-Alignment) target attribution.

    From G parallel rollouts, select all correct samples (reward == 1),
    compute their individual attribution maps, and average them to get
    the "group consensus" attribution.

    G_SA = Mean({G_i | Reward(A_i) = 1})

    Args:
        model: The VLM model.
        batch_input_ids: List of input_ids for each rollout in the group.
        batch_attention_mask: List of attention masks.
        batch_position_ids: List of position IDs.
        batch_response_ids: List of response token IDs.
        batch_response_lengths: List of response lengths.
        batch_rewards: List of reward values for each rollout.
        batch_image_token_masks: List of visual token masks.
        batch_multi_modal_inputs: List of multi-modal inputs.
        tokenizer: Tokenizer instance.
        answer_tag: Tag to locate answer tokens.
        temperature: Softmax temperature.

    Returns:
        P_SA: Averaged causal probability distribution (num_visual_tokens,), or None.
    """
    correct_attributions = []

    for idx in range(len(batch_rewards)):
        # Only use correct samples (reward == 1)
        if batch_rewards[idx] != 1.0:
            continue

        response_ids = batch_response_ids[idx]
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)

        # Find answer positions
        answer_positions_list = find_answer_token_positions(
            response_ids, tokenizer, answer_tag
        )

        if answer_positions_list[0] is None:
            continue

        # Compute gradient attribution
        input_ids = batch_input_ids[idx]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        attention_mask = batch_attention_mask[idx]
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        position_ids = batch_position_ids[idx]
        image_token_mask = batch_image_token_masks[idx]
        if image_token_mask.dim() == 1:
            image_token_mask = image_token_mask.unsqueeze(0)

        grad_attr = compute_gradient_attribution(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_length=batch_response_lengths[idx],
            answer_token_positions=answer_positions_list[0],
            image_token_mask=image_token_mask,
            multi_modal_inputs=batch_multi_modal_inputs[idx],
            temperature=1.0,
        )

        if grad_attr is not None:
            prob = compute_causal_probability(grad_attr, temperature)
            correct_attributions.append(prob)

    if len(correct_attributions) == 0:
        return None

    # Average across all correct samples
    stacked = torch.stack(correct_attributions, dim=0)
    mean_attribution = stacked.mean(dim=0)

    # Re-normalize after averaging
    return compute_causal_probability(mean_attribution, temperature)


def compute_vcpo_kl_loss(
    current_attribution: torch.Tensor,
    target_attribution: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the KL divergence loss for visual causal alignment.

    KL(P_current || P_target)

    where P_current is the causal probability from the current rollout,
    and P_target is from either SD or SA variant.

    Args:
        current_attribution: (num_visual_tokens,) gradient attribution from current rollout.
        target_attribution: (num_visual_tokens,) target attribution (P_SD or P_SA).
        temperature: Softmax temperature for normalizing current attribution.

    Returns:
        Scalar KL divergence loss.
    """
    current_prob = compute_causal_probability(current_attribution, temperature)
    # KL(P || P_target) = sum(P * log(P / P_target))
    kl_loss = F.kl_div(
        target_attribution.log(),  # log(P_target)
        current_prob,              # P
        reduction="sum",
        log_target=True,
    )
    # F.kl_div with log_target=True computes: sum(P * (log(P) - log_target))
    # which is KL(P || P_target)
    return kl_loss