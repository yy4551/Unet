import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output', help='Path to the output file')

args = parser.parse_args(['--output', 'output.txt'])
print(args.output)
print(f"Output file: {args.output}")