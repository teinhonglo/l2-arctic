import argparse
import itertools
from sklearn.metrics import confusion_matrix, classification_report

parser = argparse.ArgumentParser()
   
# configuration.
parser.add_argument('--anno', type=str, default="capt/annotation.txt")
parser.add_argument('--pred', type=str, default="capt/prediction.txt")

args = parser.parse_args()

sent_anno = {}
sent_pred = {}

def process_fn(info_fn, sent_dict):
    print("process_fn:", info_fn)
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
                if op_info[i] == "I": continue
                sent_dict[utt_id]["ref"].append(ref_info[i])
                sent_dict[utt_id]["hyp"].append(hyp_info[i])
                sent_dict[utt_id]["op"].append(op_info[i])
    return sent_dict

def comp_metric(sent_anno, sent_pred):
    print("comp_metric:")
    # Detection
    anno_list = []
    pred_list = []
    # Diagnose
    diag_report = {"ce":0, "de":0, "de_ori": []}
    utt_ids = list(sent_anno.keys())
    
    for utt_id in utt_ids:
        anno_info = sent_anno[utt_id]
        pred_info = sent_pred[utt_id]
        assert len(anno_info["op"]) == len(pred_info["op"])
        
        for i in range(len(anno_info["op"])):
            # annotation
            if anno_info["op"][i] == "C":
                anno = 1
            else:
                anno = 0
            # prediction
            if pred_info["op"][i] == "C":
                pred = 1
            else:
                pred = 0
            anno_list.append(anno)
            pred_list.append(pred)
            # Diagose
            if pred == 0:
                if anno_info["hyp"][i] == pred_info["hyp"][i]:                
                    diag_report["ce"] += 1
                else:
                    diag_report["de"] += 1
                    diag_report["de_ori"].append([anno_info["hyp"][i],  pred_info["hyp"][i]])
            
    return anno_list, pred_list, diag_report

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

if __name__ == "__main__":
    sent_anno = process_fn(args.anno, sent_anno)
    sent_pred = process_fn(args.pred, sent_pred)
    anno_list, pred_list, diag_report = comp_metric(sent_anno, sent_pred)
    
    tn, fp, fn, tp = confusion_matrix(anno_list, pred_list).ravel()
    print("tn, fp, fn, tp")
    print(tn, fp, fn, tp)
    print(classification_report(anno_list, pred_list))
    print(diag_report["ce"], diag_report["de"], diag_report["ce"] / (diag_report["ce"] + diag_report["de"]))
    # print(diag_report["de_ori"])
