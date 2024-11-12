import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy 
import time

from mygpt import myGPT
from mytokenizer import Tokenizer
from data import  DataLoader, s2t

mstime = lambda: int(round(time.time() * 1000))    

def init_model(m_path, device, vocab):
    ckpt= torch.load(m_path, map_location='cpu')
    lm_args = ckpt['args']
    lm_vocab = Tokenizer(vocab, min_occur_cnt=lm_args.min_occur_cnt, specials=[])
    lm_model = myGPT(device, lm_vocab, lm_args.embed_dim, lm_args.ff_embed_dim, lm_args.num_heads, lm_args.dropout, lm_args.layers)
    lm_model.load_state_dict(ckpt['model'])
    lm_model = lm_model.to(device)
    lm_model.eval()
    return lm_model, lm_vocab, lm_args

def greedy(lm_model, lm_vocab, device, s, max_len):
    x, m = s2t(s, lm_vocab)
    x = x.to(device)
    res = []
    for l in range(max_len):
        probs, pred = lm_model.work(x)
        next_tk = []
        for i in range(len(s)):
            next_tk.append(lm_vocab.idx2token(pred[len(s[i]) - 1, i].item()))

        s_ = []
        for idx, (sent, t) in enumerate(zip(s, next_tk)):
            if t == "<eos>":
                res.append(sent)
            else:
                s_.append(sent + [t])

        if not s_:
            break
        s = s_ 
        x, m = s2t(s, lm_vocab)
        x = x.to(device)
    res += s_   

    r = ''.join(res[0])
    if "<bos>" in r:
        return r.split("<bos>")[1]
    else:
        return r

@torch.no_grad()
def top_k_inc(lm_model, lm_vocab, device, s, k, max_len):
    start = time.time()
    incremental_state = None    
    x, m = s2t(s, lm_vocab)
    x = x.to(device)
    res = []
    for l in range(max_len):
        probs, pred, incremental_state = lm_model.work_incremental(x, incremental_state=incremental_state)
        next_tk = []
        for i in range(len(s)):
            if l == 0:
                logits = probs[len(s[i]) - 1, i]
                ps, idx = torch.topk(logits, k=k)
                ps = ps / torch.sum(ps)
            else:
                logits = probs[0, i]
                ps, idx = torch.topk(logits, k=k)
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
        return r.split("<bos>")[1]
    else:
        return r

def top_p_sampling(logits, k, p):
    ps, idx = torch.topk(logits, k=k)
    for i in range(k):
        if torch.sum(ps[:i]) > p:
            return ps[:i], idx[:i]
    return ps, idx

def top_p_inc(lm_model, lm_vocab, device, s, k, p, max_len):
    start = time.time()
    incremental_state = None
    x, m = s2t(s, lm_vocab)
    x = x.to(device)
    res = []
    for l in range(max_len):
        probs, pred, incremental_state = lm_model.work_incremental(x, incremental_state=incremental_state)
        next_tk = []
        for i in range(len(s)):
            if l == 0:
                logits = probs[len(s[i]) - 1, i]
                ps, idx = top_p_sampling(logits, k=k, p=p)
                ps = ps / torch.sum(ps)
            else:
                logits = probs[0, i]
                ps, idx = top_p_sampling(logits, k=k, p=p)
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
        return r.split("<bos>")[1]
    else:
        return r


if __name__ == "__main__":
    stage = "pretain"

    device = 0
    print("loading...")
    m_path = {"pretain" : "./ckpt/pretain/epoch0_batch_109999",
              "sft" :"./ckpt/sft/epoch6_batch_39999"}
    m_path =  m_path[stage]
    v_path = {"pretain" : "./model/pretain/vocab.txt",
              "sft" :"./model/sft/vocab.txt"}
    v_path =  v_path[stage]
    lm_model, lm_vocab, lm_args = init_model(m_path, device, v_path)
    print("done.")
    
    max_len  = 100
    i = 0

    val_set_pretrain = ["介绍⼀下南京航空航天⼤学", 
                        "Please introduce Nanjing University of Aeronautics and Astronautics",
                        "⽩⽇依⼭尽，", 
                        "君不⻅，⻩河之⽔天上来，奔流到海不复回。君不⻅，", 
                        "秦孝公据崤函之固，拥雍州之地，君⾂固守以窥周室，有席卷天下，包举宇内，囊括四海之意，并吞⼋荒之⼼。", 
                        "古之学者必有师。师者，所以传道受业解惑也。⼈⾮⽣⽽知之者，孰能⽆惑？", 
                        "当我醒来时，我发现⾃⼰在⼀个完全陌⽣的地⽅。我看到周围没有⼈，只有⼀张纸条。", 
                        "这是⼀个⽃⽓决定⼀切的⼤陆。在加玛帝国乌坦城，有个天才少年萧炎打破了所有族⼈的修炼纪录，⼀时间万⼈敬仰，众⼈艳羡。但不知为何，", 
                        "⼈⼯智能技术在图像识别领域取得了很⼤的进展，然⽽在复杂场景下仍然存在⼀些问题，例如", 
                        "In recent years, there has been increasing interest in the use of machine learning to", 
                        "已知三个数分别为1, 2, 3，则它们的平均数是", 
                        "⼩明总共有15个苹果，他分别给了3个⼈两个苹果，然后⾃⼰⼜吃了⼀个苹果，那么它还剩⼏个苹果？", 
                        "根据⽜顿第⼆定律，物体的加速度等于", 
                        "碳纳⽶管是⼀种新型的材料，具有⾮常独特的电学和光学性质。在过去的⼏年中，我们对碳纳", 
                        "下⾯是⼀段⽤python写的快速排序的代码:", 
                        "The quantum many-body problem is a fundamental problem in condensed matter physics. Despite decades of research, there is still no exact solution to this problem for large systems. In this paper, we propose a novel approach based on", 
                        "下⾯是⼀个使⽤ PyTorch 和 Transformer 的示例代码，⽤于训练⼀个⽂本分类模型：import torch\nimport torch.nn as nn\nfrom torch.utils.data import DataLoader, Dataset", 
                    ] 

    val_set_sft = [ "### INST:\n 介绍⼀下南京航空航天⼤学，\n\n### SYS:\n", 
                    "### INST:\n ⽩⽇依⼭尽，\n\n### SYS:\n", 
                    "### INST:\n 已知三个数分别为1, 2, 3，则它们的平均数是\n\n### SYS:\n", 
                    "### INST:\n ⼩明总共有15个苹果，他分别给了3个⼈两个苹果，然后⾃⼰⼜吃了⼀个苹果，那么它还剩⼏个苹果？\n\n### SYS:\n", 
                    "### INST:\n 根据⽜顿第⼆定律，物体的加速度等于\n\n### SYS:\n", 
                    "### INST:\n 碳纳⽶管是⼀种新型的材料，具有⾮常独特的电学和光学性质。在过去的几年中，我们对碳纳\n\n### SYS:\n", 
                    "### INST:\n 下⾯是⼀段⽤python写的快速排序的代码:\n\n### SYS:\n", 
                    "### INST:\n 下⾯是⼀个使⽤ PyTorch 和 Transformer 的示例代码，⽤于训练⼀个⽂本分类模型：import torch\nimport torch.nn as nn\nfrom torch.utils.data import DataLoader, Dataset\n\n### SYS:\n", 
]

    qs = val_set_sft if stage == "sft" else val_set_pretrain

    for q in qs:
        start = mstime()
        i += 1
        s = [[w for w in q]]

        r1 = greedy(lm_model, lm_vocab, device, s, max_len)

        # r2 = beam_search(lm_model, lm_vocab, device, s, max_len)

        r3 = top_k_inc(lm_model, lm_vocab, device, s, 5, max_len)

        r4 = top_k_inc(lm_model, lm_vocab, device, s, 10, max_len)

        r5 = top_k_inc(lm_model, lm_vocab, device, s, 20, max_len)

        r6 = top_k_inc(lm_model, lm_vocab, device, s, 50, max_len)

        r7 = top_k_inc(lm_model, lm_vocab, device, s, 500, max_len)

        r8 = top_p_inc(lm_model, lm_vocab, device, s, 20, 0.95, max_len)
    
        print(i)
        print("q: ", q)
        print("greedy: ", r1)
        # print("bm5: ", q+r2)
        print("tk5: ", r3)
        print("tk10: ", r4)
        print("tk20: ", r5)
        print("tk50: ", r6)
        print("tk500: ", r7)
        print("tp0.95: ", r8)
        print(mstime()-start)
    
