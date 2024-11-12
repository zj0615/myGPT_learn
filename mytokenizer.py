import numpy as np
import sentencepiece as spm

# 定义一些特殊的标记符号
PAD, UNK, BOS, EOS = '<pad>', '<unk>', '<bos>', '<eos>'
LS, RS, SP = '<s>', '</s>', ' '
BINST, EINST = '<INST>', '</INST>'
BSYS, ESYS = '<SYS>', '</SYS>'

class Tokenizer(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        # 初始化特殊标记符号列表
        idx2token = [PAD, UNK, BOS, EOS] + [LS, RS, SP, BINST, EINST, BSYS, ESYS] + (specials if specials is not None else [])
        # 读取词汇文件，统计词频
        for line in open(filename, encoding='utf8').readlines():
            try:
                token, cnt = line.strip().split()
            except:
                continue
            # 只保留出现次数大于等于min_occur_cnt的词汇
            if int(cnt) >= min_occur_cnt:
                idx2token.append(token)
        # 构建token到idx的映射和idx到token的映射
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        # 获取padding和unk的idx
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    @property
    def size(self):
        # 返回词汇表的大小
        return len(self._idx2token)
    
    @property
    def unk_idx(self):
        # 返回unk标记的idx
        return self._unk_idx
    
    @property
    def padding_idx(self):
        # 返回padding标记的idx
        return self._padding_idx
 
    def random_token(self):
        # 随机返回一个token
        return self.idx2token(1 + np.random.randint(self.size - 1))

    def idx2token(self, x):
        # 将idx转换为token，支持列表输入
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        # 将token转换为idx，支持列表输入
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

    def encode(self, x):
        # 将文本转换为token idx列表
        return self.token2idx([w for w in x])

    def decode(self, x):
        # 将token idx列表转换为文本
        return ''.join(self.idx2token(x))


if __name__ == '__main__':
    # 测试文本
    text = '南京航空航天大学是一所坐落在南京的双一流大学, Nanjing University of Aeronautics and Astronautics is a double-first-class university located in Nanjing.'
    # 初始化Tokenizer对象
    tokenizer = Tokenizer('./model/sft/vocab.txt', min_occur_cnt=50)

    # 编码文本
    tks = tokenizer.encode(text)
    print(tks)

    # 解码文本
    dtext = tokenizer.decode(tks)
    print(dtext)
    print(text == dtext)

    # 使用SentencePiece模型编码和解码文本
    sp = spm.SentencePieceProcessor(model_file='./model/sft/m.model')
    tks = sp.encode(text, out_type=int)
    print(tks)

    dtext = sp.decode(tks)
    print(dtext)
    print(text == dtext)

    print(sp.encode(text, out_type=str))