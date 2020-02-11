import click
from utils import get_config, set_print_precision, seed_random
from data import get_data_loader
from starGAN_v2 import StarGAN

"""
TODO :
1. 1x1 에서 activation 들어갈까??
1. resblock 에서 skip connected 되는애 norm 이랑 activation
"""


@click.command()
@click.option('--config', type=str, default='./config/celeba_HQ.yaml', help='Path to the config file.')
@click.option('--dataset_root', default='/mnt/disks/sdb/datasets', type=str)
@click.option('--resume', default=False, help='whether resume and train')
def main(config, dataset_root, resume):
    set_print_precision()
    seed_random()

    config = get_config(config)
    train_loader = get_data_loader(config, dataset_root, is_train=True)
    test_loader = get_data_loader(config, dataset_root, is_train=False)

    # model
    model = StarGAN(config, train_loader, test_loader)

    if not resume:
        model.train_starGAN(init_epoch=0)
    else:
        model.resume_train()  # resume train after the last saved epoch model


if __name__ == '__main__':
    main()
