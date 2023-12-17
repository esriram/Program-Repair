import sys
import os
sys.path.insert(0, os.getcwd() + '/')

import random
import warnings
import numpy as np
import torch
from huggingface_hub.utils import logging
from transformers.trainer import (
    Trainer,
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    SCALER_NAME,
    TRAINER_STATE_NAME
)
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    ShardedDDPOption
)
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

logger = logging.get_logger(__name__)

class SemCodeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        original_labels = inputs.pop("labels_original")
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # calculate contrastive loss
        cos_emb_loss = torch.nn.CosineEmbeddingLoss()
        input_encoder = model.encoder(input_ids=inputs['input_ids'])['last_hidden_state']
        output_encoder = model.encoder(input_ids=original_labels)['last_hidden_state']

        sec_loss = 0.0
        for i in range(input_encoder.size()[0]):
            sec_loss = cos_emb_loss(input_encoder[i], output_encoder[i], torch.ones(input_encoder.size()[1])).item()

        loss = (loss + sec_loss) / 2
        
        return (loss, outputs) if return_outputs else loss


    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return
        local_rank = self.args.local_rank
        if local_rank != -1:
            rng_file = os.path.join(checkpoint, f"rng_state_{local_rank}.pth")
            if not os.path.isfile(os.path.join(checkpoint, rng_file)):
                logger.info(
                    f"Didn't find an RNG file for process {local_rank}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(os.path.join(checkpoint, rng_file)):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return
        checkpoint_rng_state = torch.load(rng_file)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if torch.cuda.is_available():
            try:
                if self.args.local_rank != -1:
                    torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
                else:
                    torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            except Exception as ex:
                logger.info(
                    f"""Error encountered while loading the states, you may have used different numbers of GPUs
                    Error Message {ex}
                    """
                )

    def save_checkpoint(self):
        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self.args.output_dir
        self.store_flos()
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir)
        if self.deepspeed:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_fp16_weights_on_model_save` is True
            self.deepspeed.save_checkpoint(output_dir)
        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()
        # if is_torch_tpu_available():
        #     xm.rendezvous("saving_optimizer_states")
        #     xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        #     with warnings.catch_warnings(record=True) as caught_warnings:
        #         xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #         reissue_pt_warnings(caught_warnings)
        # elif is_sagemaker_mp_enabled():
        #     if smp.dp_rank() == 0:
        #         # Consolidate the state dict on all processed of dp_rank 0
        #         opt_state_dict = self.optimizer.state_dict()
        #         # Save it and the scheduler on the main process
        #         if self.args.should_save:
        #             torch.save(opt_state_dict, os.path.join(output_dir, OPTIMIZER_NAME))
        #             with warnings.catch_warnings(record=True) as caught_warnings:
        #                 torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        #             reissue_pt_warnings(caught_warnings)
        #             if self.use_amp:
        #                 torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif not self.deepspeed:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.use_amp:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.local_rank == -1:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model,
        # in which case output_dir may not yet exist.
        os.makedirs(output_dir, exist_ok=True)
        local_rank = self.args.local_rank
        if local_rank == -1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{local_rank}.pth"))
        self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
