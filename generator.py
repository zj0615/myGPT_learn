import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import time
from data import  DataLoader, s2t
from mygpt import myGPT
from mytokenizer import Tokenizer

def top_p_sampling(logits, k, p, temperature=1.0):
    logits = logits / temperature
    ps, idx = torch.topk(logits, k=k)
    for i in range(k):
        if torch.sum(ps[:i]) > p:
            return ps[:i], idx[:i]
    return ps, idx

def generate(lm_model, lm_vocab, device, instruction, top_k, top_p, max_new_tokens, temperature):
    s = [[w for w in instruction]]
    incremental_state = None  
    x, m = s2t(s, lm_vocab)
    inputs = x
    x = x.to(device)
    res = []
    logits_list = []

    for l in range(max_new_tokens):
        probs, pred, incremental_state = lm_model.work_incremental(x, incremental_state=incremental_state)
        next_tk = []
        step_logits = []

        if l == max_new_tokens - 1:
            logits_list = probs / temperature

        for i in range(len(s)):
            if l == 0:
                logits = probs[len(s[i]) - 1, i]
                ps, idx = top_p_sampling(logits, k=top_k, p=top_p, temperature=temperature)
                ps = ps / torch.sum(ps)
            else:
                logits = probs[0, i]
                ps, idx = top_p_sampling(logits, k=top_k, p=top_p, temperature=temperature)
                ps = ps / torch.sum(ps)
            sampled = torch.multinomial(ps, num_samples = 1)
            sampled_idx = idx[sampled]
            next_tk.append(lm_vocab.idx2token(sampled_idx.item()))
        
        s_ = []
        bidx = [1] * len(s)
        for idx, (sent, t) in enumerate(zip(s, next_tk)):
            if t == "<eos>":
                res.append(sent)
                bidx[idx] = 0
            else:
                s_.append(sent + [t])
        if not s_:
            break
        s = s_
        x, m = s2t(s, lm_vocab)
        x = x.to(device)
        bidx = torch.BoolTensor(bidx).to(device)
        incremental_state["bidx"] = bidx

    res += s_
    r = ''.join(res[0])
    if "<bos>" in r:
        response =  r.split("<bos>")[1]
    else:
        response = r

    # logits_tensor = torch.stack(logits_list, dim=0) if logits_list else torch.empty(0)
    
    return inputs, [response], logits_list





def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Tokenizer(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = myGPT(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.to(device)
    lm_model.eval()
    return lm_model, lm_vocab, lm_args

if __name__ == "__main__":
    device = 0
    print("loading...")
    m_path = "./ckpt/sft/epoch6_batch_39999"
    v_path = "./model/sft/vocab.txt"
    lm_model, lm_vocab, lm_args = init_model(m_path, device, v_path)
    print("done.")

    qs = "### INST:\n 介绍⼀下南京航空航天⼤学，\n\n### SYS:\n"


    inputs, response, logits = generate(lm_model, lm_vocab, device, qs, 50, 0.9, 20, 0.9)

    print(inputs.shape)
    print(response[0])
    print(logits.shape)

    logits = logits[0][0]
    logits = logits.float().cpu().detach()
    print(logits.argmax(-1).item())
    response = lm_vocab.decode([logits.argmax(-1).item()])
    print(response)