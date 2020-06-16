import yaml

# if __name__ == '__main__':
#     f = open('config.yml', 'r')
#     config = yaml.load(f, Loader=yaml.FullLoader)
#     print(config)


from config import cfg

if __name__ == '__main__':
    print(cfg)
    cfg['a'] = 123456
    print(cfg)

