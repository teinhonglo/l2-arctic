import os
import argparse

parser = argparse.ArgumentParser()
   
# Training configuration.
parser.add_argument('--text', type=str, default="data/train/text")
parser.add_argument('--dict', type=str, default="data/local/dict")

args = parser.parse_args()

lexicon = set()
oov = set()

with open(os.path.join(args.dict, "lexicon.txt"), "r") as fn:
    for line in fn.readlines():
        info = line.split()
        lexicon.add(info[0])
        
with open(args.text, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        for p in info[1:]:
            if not p in lexicon:
                oov.add(p)

print(oov)
