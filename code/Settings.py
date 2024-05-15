LVSRFIT_SETTINGS_DEFAULT = {
    "scale": 4,
    "num_frames": 7,
    "batch_size": 4,
    "num_epochs": 80,
    "max_iters_per_epoch": 10000,
    "training_save_interval": 1000,
    "training": {
        "learning_rate": 0.0002,
        "beta1": 0.9,
        "beta2": 0.99,
        "warmup_iter": -1, # 5000  # -1: no warm up
        "T_period": [2000, 4000, 6000, 8000],
        "restarts": [2000, 4000, 6000],
        "restart_weights": [0.5, 0.5, 0.5],
        "weight_decay": 0,
        "eta_min": 1e-7
    }
}