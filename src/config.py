def init():
    global PARAM
    PARAM = {
        'data_name': 'CIFAR10',
        'subset': 'label',
        'model_name': 'mcvae',
        'ae_name': 'vqvae',
        'control': {'controller_rate': '0.5'},
        'optimizer_name': 'Adam',
        'lr': 1e-3,
        'momentum': 0,
        'weight_decay': 0,
        'scheduler_name': 'ExponentialLR',
        'step_size': 1,
        'milestones': [100, 150],
        'patience': 10,
        'threshold': 1e-3,
        'factor': 0.5,
        'batch_size': {'train': 128, 'test': 512},
        'shuffle': {'train': True, 'test': False},
        'num_workers': 0,
        'device': 'cuda',
        'num_epochs': 200,
        'save_mode': 0,
        'world_size': 1,
        'metric_names': {'train': ['Loss','NLL'], 'test': ['Loss','NLL']},
        'init_seed': 0,
        'num_experiments': 1,
        'log_interval': 0.25,
        'log_overwrite': False,
        'resume_mode': 0
    }
