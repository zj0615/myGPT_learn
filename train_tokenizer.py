import json
from multiprocessing import Pool
from collections import Counter

BUFSIZE = 10000

ttype = 'bpe'

def process(doc):
    res = [w for w in doc]
    return res

def save(cnt, docs, nprocessors):
    res = pool.map(process, docs, len(docs)//nprocessors)
    all_lines = []
    for xs in res:
        all_lines.extend(xs)
    for x in all_lines:
        cnt.update(x)


if ttype == 'char':
    cnt = Counter()
    nprocessors = 20
    pool = Pool(nprocessors)
    docs = []
    with open('./data/sft/sft_train.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = json.loads(line)['text']
            if not line:
                continue
            docs.append(line)

            if len(docs) == BUFSIZE:
                save(cnt, docs, nprocessors)
                docs = []
                print(BUFSIZE, ' lines processed')
        if len(docs) > 0:
            save(cnt, docs, nprocessors)
            print(len(docs), ' lines processed')

    print("vocab size: ", len(cnt))
    with open('./model/sft/vocab.txt', 'w', encoding='utf8') as f:
        for x, y in cnt.most_common():
            f.write(x + '\t' + str(y) + '\n')
    print("done")

elif ttype == 'bpe':

    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input='./data/sft/sft_train.txt', model_prefix='m', vocab_size=20000, \
        character_coverage=1.0, model_type='bpe', \
        user_defined_symbols=['<pad>', '<bos>', '<eos>', '<mask>', '<INST>', '</INST>', '<SYS>', '</SYS>']
    )

else:
    pass