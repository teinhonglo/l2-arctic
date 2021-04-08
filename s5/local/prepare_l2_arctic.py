import textgrid
from tqdm import tqdm
import os
import re
import argparse

parser = argparse.ArgumentParser()
   
# configuration.
parser.add_argument('--ignore_chars', type=str, default="sp,sil,,spn")

args = parser.parse_args()
spk_ids=["ABA", "BWC", "ERMS", "HKK", "MBMPS", "NJS", 
         "SKA", "THV", "TNI", "YBAA", "YKWK", "ASI", 
         "EBVS", "HJK", "HQTV", "LXC", "NCC", "PNV", 
         "RRBI", "SVBI", "TLV", "TXHC", "YDCK", "ZHAA",]

l2_arctic_corpus="/share/corpus/l2arctic_release_v4.0"
ignore_chars = [ic for ic in args.ignore_chars.split(",")]
print(ignore_chars)

should_say_word = {"wav":{}, "text":{}, "spk":{}}
should_say_phone = {"wav":{}, "text":{}, "spk":{}}
actual_say_phone = {"wav":{}, "text":{}, "spk":{}}
actual_say_phone_wb = {"wav":{}, "text":{}, "spk":{}}
mis_utt_ids = {}

def text_normalize(text):
    text = text.lower()
    text_new = ""

    for t in text.split():
        t_n = re.sub("[0-9]", "", t)
        text_new += t_n + " "

    return text_new[:-1]

def write_kaldi_format(write_dict, data_dir):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    with open(data_dir + "/wav.scp", "w") as wav_fn:
        with open(data_dir + "/utt2spk", "w") as spk_fn:
            with open(data_dir + "/text", "w") as text_fn:
                for utt_id in list(write_dict["wav"].keys()):
                    wav_fn.write(utt_id + " " + write_dict["wav"][utt_id] + "\n")
                    spk_fn.write(utt_id + " " + write_dict["spk"][utt_id] + "\n")
                    text_fn.write(utt_id + " " + text_normalize(write_dict["text"][utt_id]) + "\n")
                

for spk in os.listdir(l2_arctic_corpus):
    if spk in spk_ids:
        # /share/corpus/l2arctic_release_v4.0/MBMPS/annotation
        spk_dir = l2_arctic_corpus + "/" + spk + "/annotation"
        wav_dir = l2_arctic_corpus + "/" + spk + "/wav"
        for tg_fn in tqdm(os.listdir(spk_dir)):
            utt_id = spk + "_" + tg_fn.split(".")[0]
            should_say_word["text"][utt_id] = ""
            should_say_phone["text"][utt_id] = ""
            actual_say_phone["text"][utt_id] = ""
            actual_say_phone_wb["text"][utt_id] = ""
            
            textgrid_fn = spk_dir + "/" + tg_fn
            wav_fn = wav_dir + "/" + tg_fn.split(".")[0] + ".wav"
            should_say_word["wav"][utt_id] = wav_fn
            should_say_phone["wav"][utt_id] = wav_fn
            actual_say_phone["wav"][utt_id] = wav_fn
            actual_say_phone_wb["wav"][utt_id] = wav_fn
            
            should_say_word["spk"][utt_id] = spk
            should_say_phone["spk"][utt_id] = spk
            actual_say_phone["spk"][utt_id] = spk
            actual_say_phone_wb["spk"][utt_id] = spk

            tg = textgrid.TextGrid()
            tg.read(textgrid_fn)
            word_history = []
            word_info = tg.tiers[0]

            # word annotation
            for i in range(len(word_info[:])):
                if word_info[i].mark == "":
                    continue
                anno_text = word_info[i].mark.split(",")
                if len(anno_text) != 3:
                    should_say_word["text"][utt_id] += " " + anno_text[0].upper()
                    word_history.append([anno_text[0].upper(), word_info[i].minTime, word_info[i].maxTime])
                else:
                    print("Something went wrong.")
                    exit(0)
            w_idx = 0
            # phone annotation
            phn_info = tg.tiers[1]
            for i in range(len(phn_info[:])):
                if phn_info[i].mark == "":
                    continue
                
                anno_text = phn_info[i].mark.replace(" ","").split(",")
                # correct phoneme
                if len(anno_text) != 3:
                    if anno_text[0] in ignore_chars:
                        continue
                    should_say_phone["text"][utt_id] += " " + anno_text[0]
                    actual_say_phone["text"][utt_id] += " " + anno_text[0]
                    
                    if phn_info[i].minTime >= word_history[w_idx][2]:
                        actual_say_phone_wb["text"][utt_id] += " "
                        w_idx += 1
                    actual_say_phone_wb["text"][utt_id] += "{" + anno_text[0] + "}"
                # mis. phoneme
                else:
                    # skip phonemes sil, sp and "" (should say phone)
                    if not (anno_text[0] in ignore_chars):
                        should_say_phone["text"][utt_id] += " " + anno_text[0]
                    
                    if anno_text[1] in ignore_chars:
                        continue
                    
                    actual_say_phone["text"][utt_id] += " " + anno_text[1]
                    
                    if phn_info[i].minTime >= word_history[w_idx][2]:
                        actual_say_phone_wb["text"][utt_id] += " "
                        w_idx += 1
                    if utt_id not in mis_utt_ids:
                        mis_utt_ids[utt_id] = len(list(mis_utt_ids.keys()))
                    actual_say_phone_wb["text"][utt_id] += "{" + anno_text[1] + "}"
                    
            actual_say_phone_wb["text"][utt_id] = " " + actual_say_phone_wb["text"][utt_id]
    else:
        print(spk)

write_kaldi_format(should_say_word, "data/l2_arctic/should_say_word")
write_kaldi_format(should_say_phone, "data/l2_arctic/should_say_phone")
write_kaldi_format(actual_say_phone, "data/l2_arctic/actual_say_phone")
write_kaldi_format(actual_say_phone_wb, "data/l2_arctic/actual_say_phone_wb")

with open("data/l2_arctic/mis_utt_ids", "w") as fn:
    for utt_id in list(mis_utt_ids.keys()):
        fn.write(utt_id)
        fn.write("\n")
