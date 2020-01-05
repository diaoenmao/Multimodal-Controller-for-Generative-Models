def init():
    global PARAM
    PARAM = {
        'data_name': 'MNIST',
        'subset': 'label',
        'model_name': 'dccgan',
        'control': {'mode_data_size': '0'},
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
        'batch_size': {'train': 64, 'test': 512},
        'shuffle': {'train': True, 'test': False},
        'num_workers': 0,
        'device': 'cuda',
        'num_epochs': 20,
        'save_mode': 0,
        'world_size': 1,
        'metric_names': {'train': ['Loss','NLL'], 'test': ['Loss','NLL']},
        'init_seed': 0,
        'num_Experiments': 1,
        'log_interval': 0.25,
        'log_overwrite': False,
        'resume_mode': 0
    }
    if PARAM['model_name'] in ['vae', 'cvae', 'dcvae', 'dccvae']:
        PARAM['lr'] = 1e-3
        PARAM['batch_size']['train'] = 256
        PARAM['metric_names'] = {'train': ['Loss','NLL'], 'test': ['Loss','NLL']}
    elif PARAM['model_name'] in ['gan', 'cgan', 'dcgan', 'dccgan']:
        PARAM['lr'] = 2e-4
        PARAM['batch_size']['train'] = 64
        PARAM['metric_names'] = {'train': ['Loss', 'Loss_D', 'Loss_G'], 'test': ['Loss', 'Loss_D', 'Loss_G']}
    else:
        raise ValueError('Not valid config')
