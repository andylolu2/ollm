import torch
import wandb
from absl import logging
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState


class GenerateSamplesCallback(TrainerCallback):
    def __init__(self, num_samples: int, response_template: list[int]):
        super().__init__()
        self.num_samples = num_samples
        self.response_template = torch.tensor(response_template, device="cpu")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        model = kwargs["model"]
        eval_loader = kwargs["eval_dataloader"]
        tokenizer = kwargs["tokenizer"]

        prompts: list[dict[str, torch.Tensor]] = []
        for batch in eval_loader:
            if len(prompts) >= self.num_samples:
                break

            device = batch["input_ids"].device
            for input_ids in batch["input_ids"].cpu():
                if len(prompts) >= self.num_samples:
                    break

                # search for the response template
                for start in range(len(input_ids) - len(self.response_template)):
                    end = start + len(self.response_template)
                    if (input_ids[start:end] == self.response_template).all():
                        prompts.append(
                            {
                                "input_ids": input_ids[:end].to(device),
                                "target_ids": input_ids[end:].to(device),
                            }
                        )
                        break

        samples = []
        for i, prompt in enumerate(prompts):
            [sample] = model.generate(
                inputs=prompt["input_ids"].unsqueeze(0),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                max_new_tokens=1024,
                use_cache=True,
            )
            sample = {
                "prompt": tokenizer.decode(prompt["input_ids"]),
                "completion": tokenizer.decode(sample[len(prompt["input_ids"]) :]),
                "target": tokenizer.decode(
                    prompt["target_ids"],
                    skip_special_tokens=True,  # Remove pad tokens
                ),
            }
            logging.info("Sample %d: %s", i, sample)
            samples.append(sample)

        if len(samples) > 0 and wandb.run is not None:
            table = wandb.Table(
                columns=list(samples[0].keys()),
                data=[list(s.values()) for s in samples],
            )
            wandb.log({"eval/samples": table}, step=state.global_step + 1)
