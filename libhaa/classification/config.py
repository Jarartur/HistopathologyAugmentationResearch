CLASS_CONFIG = {
    "seed": 2021,
    "trainer": {
        "max_epochs": 1000,
        "accumulate_grad_batches": 1,
        "log_every_n_steps": 50,
    },
    "data": {
        "dataset_name": "haav2",
        "batch_size": 256,
        "img_size": [224, 224],
        "num_workers": 14,
        "train_data_csv": None,
        "val_data_csv": None,
        "test_data_csv": None,
    },
    "model": {
        "optimizer_init": {
            "class_path": "torch.optim.AdamW",
            "init_args": {
                "lr": 0.01,
            },
        },
        "lr_scheduler_init": {
            "class_path": "torch.optim.lr_scheduler.ExponentialLR",
            "init_args": {"gamma": 0.97},
            "step": "epoch",
        },
    },
}

ARTIFACT_NAMES = {
    0: "air",
    1: "dust",
    2: "tissue",
    3: "ink",
    4: "marker",
    5: "focus",
}
