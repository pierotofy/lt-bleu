#!/usr/bin/env python
import argparse
import random
import os
parser = argparse.ArgumentParser(description='Sample a subset of translations from a parallel corpus')
parser.add_argument('source',
    type=str,
    default=None,
    help='Source input .txt file')
parser.add_argument('target',
    type=str,
    default=None,
    help='Target input .txt file')
parser.add_argument('samples',
    type=int,
    default=None,
    help='Number of samples')


args = parser.parse_args()

source_dst = args.source + ".sampled.%s" % args.samples
target_dst = args.target + ".sampled.%s" % args.samples

if os.path.isfile(source_dst):
    print("File exists: %s exiting..." % source_dst)
    exit(1)
if os.path.isfile(target_dst):
    pritn("File exists: %s exiting..." % target_dst)
    exit(1)

print("Reading %s" % args.source)
with open(args.source) as f:
    source_lines = [l.strip() for l in f.read().split("\n")]

print("Reading %s" % args.target)
with open(args.target) as f:
    target_lines = [l.strip() for l in f.read().split("\n")]

if len(source_lines) != len(target_lines):
    print("Files have different number of lines (%s vs. %s)" % (len(source_lines), len(target_lines)))
    exit(1)



sampled = {}
with open(source_dst, "w") as fs:
    with open(target_dst, "w") as ft:
        for i in range(args.samples):
            r = random.randint(0, len(target_lines))
            if not r in sampled:
                fs.write(source_lines[r] + "\n")
                ft.write(target_lines[r] + "\n")
                sampled[r] = True

print("Wrote %s" % source_dst)
print("Wrote %s" % target_dst)
