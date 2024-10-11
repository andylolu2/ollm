from ml_collections import config_dict


def get_config():
    config = config_dict.ConfigDict()

    config.seed = 0
    config.output_dir = config_dict.placeholder(str)
    config.wandb = dict(
        project="llm-ol",
        notes="",
    )

    config.model = dict(
        name="mistralai/Mistral-7B-Instruct-v0.2",
        response_template=[733, 28748, 16289, 28793],  # _[/INST]
        instruction_template=[733, 16289, 28793],  # _[INST]
    )

    config.data = dict(
        train_file=config_dict.placeholder(str),
        eval_file=config_dict.placeholder(str),
        train_size=config_dict.placeholder(int),
        eval_size=1024,
    )

    config.train = dict(
        epochs=2.0,
        warmup_steps=100,
        learning_rate=1e-5,
        logging_steps=50,
        grad_acc_steps=1,
        batch_size=16,
        max_seq_length=2048,
        group_by_length=True,
        lora=dict(
            rank=32,
            alpha=16,
            dropout=0,
        ),
    )

    config.eval = dict(
        eval_steps=500,
        batch_size=32,
        num_generate_samples=5,
    )

    return config
