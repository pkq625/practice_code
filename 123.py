import argparse
parser = argparse.ArgumentParser()
parser.add_argument('square', type=int)
parser.add_argument('--verbosity','-v', action='count', default = 0)
args = parser.parse_args()
answer = args.square**2
print(args.verbosity)
if args.verbosity == 2:
    print(f'the square of {args.square} equals {answer}')
elif args.verbosity == 1:
    print(f'{args.square}^2 is {answer}')
else:
    print('?')
