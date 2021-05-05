import argparse
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import pprint
pp = pprint.PrettyPrinter(indent=4)

parser = argparse.ArgumentParser()
   
# configuration.
parser.add_argument('--anno', type=str, default="capt/annotation.txt")
parser.add_argument('--text', type=str, default="data/train_tr/text")

args = parser.parse_args()

sent_anno = {}
sent_pred = {}

def process_fn(info_fn, sent_dict):
    # print("process_fn:", info_fn)
    with open(info_fn, "r") as fn:
        for line1, line2, line3, line4 in itertools.zip_longest(*[fn]*4):
            ref_info = line1.split()
            utt_id = ref_info[0]
            
            ref_info = ref_info[2:]
            hyp_info = line2.split()[2:]
            op_info = line3.split()[2:]
            # ignore 4rd line (non-sense)
            assert len(ref_info) == len(hyp_info) and len(hyp_info) == len(op_info)
            
            sent_dict[utt_id] = {"ref":[], "hyp":[], "op":[]}
            # extract alignment information from the file
            for i in range(len(ref_info)):
                if op_info[i] == "D": continue
                sent_dict[utt_id]["ref"].append(ref_info[i])
                sent_dict[utt_id]["hyp"].append(hyp_info[i])
                sent_dict[utt_id]["op"].append(op_info[i])
    return sent_dict


if __name__ == "__main__":
    sent_anno = process_fn(args.anno, sent_anno)
    text = {}
    with open(args.text, "r") as fn:
        for line in fn.readlines():
            info = line.split()
            text[info[0]] = info[1:]

    for utt_id in list(text.keys()):
        p_tag = []
        if utt_id in sent_anno:
            utt_info = sent_anno[utt_id]
            for o in utt_info["op"]:
                if o == "C":
                    p_tag.append("C")
                else:
                    p_tag.append("MP")
        else:
            p_tag = ["C" for t in text[utt_id]]
        utt_content = ' '.join(p_tag)
        print(utt_id + " " + utt_content)
        
