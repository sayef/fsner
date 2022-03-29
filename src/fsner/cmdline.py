import argparse

from fsner import __version__
from fsner.trainer import init_trainer_parser, trainer_main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--version', action='version',
                        version='fsner-{version}'.format(version=__version__))

    sub_parsers = parser.add_subparsers()
    trainer_parser = sub_parsers.add_parser('trainer')
    trainer_parser = init_trainer_parser(trainer_parser)
    trainer_parser.set_defaults(func=trainer_main)

    args = parser.parse_args()
    if 'trainer' in args:
        print("Parameters:")
        print("=" * 50)
        for k, v in vars(args).items():
            v = str(v)
            if str(k) == 'func': continue
            print(f"{k:<30}{v:>20}")
        print("=" * 50)

    try:
        args.func(args)
    except AttributeError:
        parser.print_help()
        parser.exit()


if __name__ == "__main__":
    main()
