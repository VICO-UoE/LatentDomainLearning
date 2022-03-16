import argparse


def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expecting a bool: provide one of yes/no, true/false, t/f, y/n, 1/0.')


def get_default_parser():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=120)

    parser.add_argument('--ckpt_path', type=str, default="ckpt")
    parser.add_argument('--data_path', type=str, default="data")
    parser.add_argument('--dataset', type=str, default="office_home", choices=["office_home", "pacs"])

    parser.add_argument('--milestones', type=str, default="80,100")
    parser.add_argument('--gamma', type=float, default=.1)
    parser.add_argument('--optim', type=str, default="sgd")
    parser.add_argument('--lr_sgd', type=float, default=.1)
    parser.add_argument('--weight_decay', type=float, default=1.e-4)
    parser.add_argument('--momentum_sgd', type=float, default=.9)
    parser.add_argument('--num_adapters', type=int, default=2)

    return parser
