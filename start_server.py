from serve.controller import Controller
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--speculate", action="store_true", help="Enable speculation for successor callsites")
args = parser.parse_args()
server = Controller(speculate=args.speculate)