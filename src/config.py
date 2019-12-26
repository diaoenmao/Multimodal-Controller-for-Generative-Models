def init():
    global PARAM
    PARAM = {
        'data_name': 'Omniglot',
        'subset': 'label',
        'model_name': 'vae',
        'control': {'normalization': 'none', 'activation': 'relu', 'hidden_size': '1000', 'latent_size': '200',
                    'num_layers': '2', 'mode_data_size': '100', 'mode_size': '1'},
        'optimizer_name': 'Adam',
        'lr': 1e-3,
        'momentum': 0,
        'weight_decay': 0,
        'scheduler_name': 'MultiStepLR',
        'step_size': 1,
        'milestones': [100],
        'patience': 5,
        'threshold': 1e-3,
        'factor': 0.1,
        'batch_size': {'train': 200, 'test': 500},
        'shuffle': {'train': True, 'test': False},
        'num_workers': 0,
        'device': 'cuda',
        'num_epochs': 20,
        'save_mode': 0,
        'world_size': 1,
        'metric_names': {'train': ['Loss', 'NLL'], 'test': ['Loss', 'NLL']},
        'init_seed': 0,
        'num_Experiments': 1,
        'log_interval': 0.25,
        'log_overwrite': False,
        'resume_mode': 0
    }
