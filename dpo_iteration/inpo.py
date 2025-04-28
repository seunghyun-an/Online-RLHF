import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Tuple, Optional, List, Literal, Any
from copy import deepcopy
from contextlib import nullcontext

# Assume trl library and transformers are installed
from trl import DPOTrainer
from trl.trainer.utils import DPODataCollatorWithPadding # Assuming this collator is used
from transformers import PreTrainedModel, TrainingArguments, PreTrainedTokenizerBase
import warnings

# Assume compute_inpo_loss is defined as previously:
# def compute_inpo_loss(policy_chosen_logps, policy_rejected_logps, ..., eta, tau, ...): -> torch.Tensor
# And compute_log_probs (or equivalent logic within concatenated_forward) exists

class MyINPOTrainer(DPOTrainer):
    """
    An adaptation of DPOTrainer to implement the INPO loss,
    using the recomputation strategy for pi_t log probabilities.
    """
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        pi_t_model: Optional[Union[PreTrainedModel, nn.Module]] = None, # ADDED: Frozen model from iteration t
        beta: float = 0.1, # REPURPOSED: Use DPOTrainer's beta as INPO's tau
        eta: float = 0.05, # ADDED: INPO's OMD parameter
        label_smoothing: float = 0, # Keep for potential use or ignore if INPO doesn't use it
        loss_type: Literal["inpo"] = "inpo", # Fixed for INPO
        args: Optional[TrainingArguments] = None, # Use standard TrainingArguments
        data_collator: Optional[Any] = None, # Should be DPODataCollatorWithPadding or similar
        label_pad_token_id: int = -100,
        padding_value: Optional[int] = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Union[Any, Dict[str, Any]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[Any]] = None, # Use TrainerCallback type if available
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[Any], Dict]] = None, # Use EvalLoopOutput type if available
        precompute_ref_log_probs: bool = False, # DPOTrainer specific, might interact
        dataset_num_proc: Optional[int] = None,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
        model_adapter_name: Optional[str] = None,
        ref_adapter_name: Optional[str] = None,
        reference_free: bool = False,
        force_use_ref_model: bool = False, # DPOTrainer specific
    ):
        # We can leverage DPOTrainer's __init__ for model loading, setup etc.
        # We just need to store the extra pi_t_model and eta.
        super().__init__(
            model=model,
            ref_model=ref_model,
            beta=beta, # Passing beta (tau for us)
            label_smoothing=label_smoothing,
            loss_type=loss_type, # Set to 'inpo' - DPO's dpo_loss won't be called if we override compute_loss/get_batch_loss_metrics
            args=args,
            data_collator=data_collator,
            label_pad_token_id=label_pad_token_id,
            padding_value=padding_value,
            truncation_mode=truncation_mode,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
            max_target_length=max_target_length,
            peft_config=peft_config,
            is_encoder_decoder=is_encoder_decoder,
            disable_dropout=disable_dropout,
            generate_during_eval=generate_during_eval,
            compute_metrics=compute_metrics,
            precompute_ref_log_probs=precompute_ref_log_probs,
            dataset_num_proc=dataset_num_proc,
            model_init_kwargs=model_init_kwargs,
            ref_model_init_kwargs=ref_model_init_kwargs,
            model_adapter_name=model_adapter_name,
            ref_adapter_name=ref_adapter_name,
            reference_free=reference_free,
            force_use_ref_model=force_use_ref_model,
        )

        # --- Store INPO specific models and parameters ---
        self.eta = eta
        self.tau = self.beta # Use beta passed to super() as tau

        if pi_t_model is None:
             raise ValueError("pi_t_model must be provided for MyINPOTrainer.")
        self.pi_t_model = pi_t_model

        # Ensure pi_t_model is on the correct device and frozen if it's a nn.Module
        if isinstance(self.pi_t_model, nn.Module):
             self.pi_t_model.to(self.args.device)
             self.pi_t_model.eval()
             self.pi_t_model.requires_grad_(False)

        # Verify loss_type is set to inpo, otherwise behavior is undefined
        if self.loss_type != "inpo":
             warnings.warn("MyINPOTrainer initialized with loss_type != 'inpo'. Loss calculation will use INPO logic.", UserWarning)
             self.loss_type = "inpo" # Force it for clarity

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """
        Compute the INPO loss and metrics for the given batch.
        Overrides the DPOTrainer method.
        """
        metrics = {}

        # 1. Get logps for current policy (pi_{t+1}) and reference policy (pi_ref)
        # Leverages DPOTrainer's existing logic which uses concatenated_forward internally
        # Note: DPOTrainer's get_batch_loss_metrics calls concatenated_forward for policy
        # and potentially ref_model, then calls dpo_loss. We intercept before dpo_loss.

        # Call concatenated_forward for the policy model (requires gradients)
        # We need chosen_logps, rejected_logps from this.
        # concatenated_forward also returns logits etc., which we might ignore here
        # unless needed for metrics.
        policy_forward_output = self.concatenated_forward(model, batch)
        policy_chosen_logps = policy_forward_output[0]
        policy_rejected_logps = policy_forward_output[1]
        # Keep original logits if needed for metrics later
        policy_chosen_logits = policy_forward_output[2].detach()
        policy_rejected_logits = policy_forward_output[3].detach()


        # Get reference model logps (no gradients needed)
        if self.ref_model is None or self.reference_free:
             # Handle case without a reference model if necessary (e.g., set logratios to 0)
             # DPOTrainer's concatenated_forward has logic for self.null_ref_context
             # but easier to just create zero tensors if truly reference free.
             reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
             reference_rejected_logps = torch.zeros_like(policy_rejected_logps)
        elif self.precompute_ref_log_probs and train_eval == "train":
             # Use precomputed logps if available and enabled
             reference_chosen_logps = batch["reference_chosen_logps"]
             reference_rejected_logps = batch["reference_rejected_logps"]
        else:
             # Compute reference logps using ref_model
             with torch.no_grad():
                  ref_forward_output = self.concatenated_forward(self.ref_model, batch)
                  reference_chosen_logps = ref_forward_output[0]
                  reference_rejected_logps = ref_forward_output[1]

        # 2. *** Recompute logps for the previous policy (pi_t) ***
        # This requires an extra forward pass using the frozen pi_t_model
        with torch.no_grad():
             # Ensure pi_t_model is on the correct device (should be done in __init__)
             # self.pi_t_model.to(self.args.device) # Already done in init
             pi_t_forward_output = self.concatenated_forward(self.pi_t_model, batch)
             pi_t_chosen_logps = pi_t_forward_output[0]
             pi_t_rejected_logps = pi_t_forward_output[1]


        # 3. Compute the INPO loss
        # Use the external compute_inpo_loss function or inline the logic:
        # --- Inlining INPO loss calculation ---
        policy_logratios_pi = policy_chosen_logps - policy_rejected_logps
        reference_logratios_ref = reference_chosen_logps - reference_rejected_logps
        pi_t_logratios_pit = pi_t_chosen_logps - pi_t_rejected_logps

        reference_term = (self.tau / self.eta) * reference_logratios_ref
        pi_t_term = ((self.eta - self.tau) / self.eta) * pi_t_logratios_pit

        h_t = policy_logratios_pi - reference_term - pi_t_term

        if (2 * self.eta) <= 0:
            raise ValueError("2 * eta must be positive for the target offset.")
        target = 1.0 / (2.0 * self.eta)
        losses = (h_t - target) ** 2
        loss = losses.mean()
        # --- End Inlining ---

        # 4. Calculate metrics (optional, can reuse DPO's reward calculation logic)
        # DPO calculates implicit rewards based on log ratios. We can do the same for monitoring.
        with torch.no_grad():
             chosen_rewards = self.tau * (policy_chosen_logps - reference_chosen_logps).detach()
             rejected_rewards = self.tau * (policy_rejected_logps - reference_rejected_logps).detach()
             reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}loss"] = loss.item() # Store computed INPO loss
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/pi_t_rejected"] = pi_t_rejected_logps.detach().mean().cpu() # Add pi_t logps
        metrics[f"{prefix}logps/pi_t_chosen"] = pi_t_chosen_logps.detach().mean().cpu() # Add pi_t logps
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().cpu()
        # Add eta and tau for tracking
        metrics[f"{prefix}inpo/eta"] = self.eta
        metrics[f"{prefix}inpo/tau"] = self.tau


        return loss, metrics

    # Override compute_loss to use the modified get_batch_loss_metrics
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        # Ensure the correct collator is used, otherwise inputs might be wrong format
        if not isinstance(self.data_collator, DPODataCollatorWithPadding):
             warnings.warn(
                  "MyINPOTrainer expects inputs formatted by DPODataCollatorWithPadding or similar."
                  " Unexpected behavior may occur with different collators."
             )

        # Use appropriate context manager for mixed precision
        compute_loss_context_manager = self.accelerator.autocast() if self.accelerator.state.mixed_precision != "no" else nullcontext()

        with compute_loss_context_manager:
             loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Store metrics for logging
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
             # Note: The original DPO trainer might return different outputs here.
             # Adjust if specific outputs beyond loss/metrics are needed downstream.
             return (loss, metrics)
        return loss

    # prediction_step needs to be overridden as well if evaluation is desired,
    # as it also calls get_batch_loss_metrics and interprets the results.
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
         # Ensure the correct collator is used
         if not isinstance(self.data_collator, DPODataCollatorWithPadding):
             warnings.warn(
                 "MyINPOTrainer expects inputs formatted by DPODataCollatorWithPadding or similar."
                 " Unexpected behavior may occur with different collators during prediction."
             )

         if ignore_keys is None:
             if hasattr(model, "config"):
                 ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
             else:
                 ignore_keys = []

         # Use appropriate context manager for mixed precision
         prediction_context_manager = self.accelerator.autocast() if self.accelerator.state.mixed_precision != "no" else nullcontext()

         with torch.no_grad(), prediction_context_manager:
             # Get loss and metrics using the overridden method
             loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

         # Move loss to CPU if necessary (Trainer handles device placement generally)
         loss = loss.detach()

         # Store metrics for logging
         self.store_metrics(metrics, train_eval="eval")

         if prediction_loss_only:
             return (loss, None, None)

         # The original DPOTrainer returns dummy logits/labels here based on metrics.
         # We can replicate or modify if needed.
         # Returning None for logits/labels as INPO doesn't directly produce comparable eval logits/labels easily.
         # You might want to add actual generation/evaluation logic here if needed.
         # For just computing eval loss, this is sufficient.
         logits = None
         labels = None

         return (loss, logits, labels)


# --- Remember ---
# The outer loop managing INPO iterations (t), creating the INPOTrainer
# instance for each iteration with the correct pi_t_model, and preparing
# the corresponding dataset (D_t) needs to be implemented in your main script.