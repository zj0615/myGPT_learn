import json

def SFT_bonding(datafile, outputfile):
    with open(datafile, 'r') as f, open(outputfile, 'w', encoding='utf8') as output_file:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)

            inst = line['instruction']
            input = line['input']
            out = line['output']
            sys = line['system']
            hist = line['history']

            if not sys:
                text = "<bos> "
            else:
                text = "<bos> " + sys + " "
            
            if hist:
                for hist_f in hist:
                    text += "<INST> " + hist_f[0] + " </ISNT>" + " <SYS> " + hist_f[1] + " </SYS> "
            if inst:
                text += "<INST> " + inst + (input or "") + " </INST>"
            if out:
                text += " <SYS> " + out + " </SYS> <eos>"

            new_data = {"text": text, "meta": line["meta"]}
            
            json.dump(new_data, output_file, ensure_ascii=False)
            output_file.write("\n") 


if __name__ == "__main__":
    SFT_bonding('./data/sft/train.txt', './data/sft/sft_train.txt')
    SFT_bonding('./data/sft/val.txt', './data/sft/sft_val.txt')