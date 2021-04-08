import argparse
from tqdm import tqdm
import re

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str, default="decode/hyp.trn")
parser.add_argument('--dest', type=str, default="decode/capt/hyp.txt")

args = parser.parse_args()

new_hyp = {}

with open(args.src, "r") as fn:
    for line in fn.readlines():
        info = line.split()
        spkid_uttid = info[-1]
        # NOTE: hardcore (e.g., (NJS-NJS_arctic_a0023) ->  NJS_arctic_a0023)
        uttid = '-'.join(re.sub("\(|\)", "", spkid_uttid).split("-")[1:])
        content = ' '.join(info[:-1])
        new_hyp[uttid] = content
        print(len(list(new_hyp.keys())))

with open(args.dest, "w") as fn:
    for utt_id, content in new_hyp.items():
        fn.write(utt_id + " " + content + "\n")
