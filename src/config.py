def init():
    global PARAM
    PARAM = {
        'data_name': 'MNIST',
        'model_name': 'fae',
        'control_name': '32_8_2_32_0_100_0_0',
        'optimizer_name': 'Adam',
        'lr': 1e-3,
        'momentum': 0,
        'weight_decay': 0,
        'scheduler_name': 'None',
        'step_size': 1,
        'milestones': [100],
        'patience': 5,
        'threshold': 1e-3,
        'factor': 0.1,
        'batch_size': {'train': 200, 'test': 200},
        'shuffle': {'train': True, 'test': False},
        'num_workers': 0,
        'device': 'cuda',
        'num_epochs': 200,
        'save_mode': 0,
        'world_size': 1,
        'metric_names': {'train':['Loss', 'PSNR'],'test':['Loss','PSNR']},
        'init_seed': 0,
        'num_Experiments': 1,
        'log_interval': 0.25,
        'normalization': 'none',
        'activation': 'relu',
        'resume_mode': 0
    }
