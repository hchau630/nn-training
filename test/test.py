import argparse

def subparser_a():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('a')
    group.add_argument('-m', type=int)
    group = parser.add_argument_group('b')
    group.add_argument('-n', type=str)
    return parser

def subparser_b():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('a')
    group.add_argument('-m', type=int)
    group = parser.add_argument_group('b')
    group.add_argument('-o', type=str)
    return parser

def parser():
    parser = argparse.ArgumentParser()
    subparsers_action = parser.add_subparsers()
    subparsers = []
    
    d = {'subparser_a': subparser_a(), 'subparser_b': subparser_b()}
    for k, v in d.items():
        subparser = subparsers_action.add_parser(k, parents=[v], add_help=False)
        subparsers.append(subparser)
        
    parser.add_argument('-k', type=str)
        
    return parser, subparsers_action, subparsers

def main(args):
    print(args)

if __name__ == '__main__':
    p, spa, sps = parser()
    args = p.parse_args()
    main(args)