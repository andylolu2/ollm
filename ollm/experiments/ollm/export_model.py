"""Merge the LoRA adapters into a single model"""

from pathlib import Path

from absl import app, flags, logging
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", None, "Checkpoint directory", required=True)


def main(_):
    logging.info("Loading model from %s", FLAGS.checkpoint_dir)
    peft_config = PeftConfig.from_pretrained(FLAGS.checkpoint_dir)
    logging.info("Peft config: %s", peft_config)
    peft_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,  # type: ignore
            device_map="auto",
            torch_dtype="auto",
        ),
        FLAGS.checkpoint_dir,
    ).eval()
    base_model = peft_model.merge_and_unload(progressbar=True, safe_merge=True)
    base_model.save_pretrained(Path(FLAGS.checkpoint_dir) / "merged")

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.checkpoint_dir)
    tokenizer.save_pretrained(Path(FLAGS.checkpoint_dir) / "merged")
    logging.info(
        "Model and tokenizer saved to %s", Path(FLAGS.checkpoint_dir) / "merged"
    )


if __name__ == "__main__":
    app.run(main)
