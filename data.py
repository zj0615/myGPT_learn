# data.py

import random
import torch
import numpy as np 
import json

BUFSIZE = 4096000  # 定义缓冲区大小

# 将列表转换为张量的函数，如果提供了分词器，则使用分词器的token2idx方法，否则直接填充0
def ListsToTensor(xs, tknizer=None):
    max_len = max(len(x) for x in xs)  # 找到列表中最长的元素长度
    ys = []  # 初始化结果列表
    for x in xs:
        if tknizer is not None:
            y = tknizer.token2idx(x) + [tknizer.padding_idx] * (max_len - len(x))  # 使用分词器转换并填充
        else:
            y = x + [0] * (max_len - len(x))  # 直接填充0
        ys.append(y)  # 添加到结果列表
    return ys  # 返回结果列表

# 将数据批处理的函数，返回真实值、输入值和掩码
def batchify(data, tknizer):
    truth, inp, msk = [], [], []  # 初始化三个列表
    for x in data:
        inp.append(x[:-1])  # 输入值为除最后一个元素外的所有元素
        truth.append(x[1:])  # 真实值为除第一个元素外的所有元素
        msk.append([1 for i in range(len(x) - 1)])  # 掩码为长度等于真实值的列表，元素全为1
    truth = torch.LongTensor(ListsToTensor(truth, tknizer)).t_().contiguous()  # 转换为张量并转置
    inp = torch.LongTensor(ListsToTensor(inp, tknizer)).t_().contiguous()  # 转换为张量并转置
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()  # 转换为张量并转置
    return truth, inp, msk  # 返回三个张量

# 将字符串列表转换为输入和掩码的函数
def s2t(strs, tknizer):
    inp, msk = [], []  # 初始化两个列表
    for x in strs:
        inp.append([w for w in x])  # 输入值为字符串中的每个字符
        msk.append([1 for i in range(len(x))])  # 掩码为长度等于字符串长度的列表，元素全为1
    inp = torch.LongTensor(ListsToTensor(inp, tknizer)).t_().contiguous()  # 转换为张量并转置
    msk = torch.FloatTensor(ListsToTensor(msk)).t_().contiguous()  # 转换为张量并转置
    return inp, msk  # 返回两个张量

# 将列表分割为多个子列表的函数
def chunks(l, n):
    n = max(1, n)  # 确保n至少为1
    return [l[i:i+n] for i in range(0, len(l), n)]  # 分割列表

# 解析文件行的函数，返回处理后的数据列表
def parse_lines(lines, max_len, min_len):
    data = []  # 初始化数据列表
    for line in lines:
        line = line.strip()  # 去除行首尾空白字符
        if not line:  # 如果行为空，则跳过
            continue
        line = json.loads(line)['text'].strip()  # 解析json并去除文本首尾空白字符
        if not line:  # 如果文本为空，则跳过
            continue
        line = [w for w in line]  # 将文本转换为字符列表
        sents = chunks(line, max_len)  # 分割文本为多个子列表
        if len(sents[-1]) < min_len:  # 如果最后一个子列表长度小于最小长度，则去除
            sents = sents[:-1]
        data.extend(sents)  # 将子列表添加到数据列表
    return data  # 返回数据列表

# 数据加载器类
class DataLoader(object):
    def __init__(self, tknizer, filename, batch_size, max_len, min_len):
        self.batch_size = batch_size  # 批处理大小
        self.tknizer = tknizer  # 分词器
        self.max_len = max_len  # 最大长度
        self.min_len = min_len  # 最小长度
        self.filename = filename  # 文件名
        self.stream = open(self.filename, encoding='utf8')  # 打开文件流
        self.epoch_id = 0  # 初始化epoch计数

    # 迭代器方法
    def __iter__(self):
        lines = self.stream.readlines(BUFSIZE)  # 读取文件行到缓冲区

        if not lines:  # 如果没有读到行，则关闭当前文件流，重新打开并读取
            self.epoch_id += 1
            self.stream.close()
            self.stream = open(self.filename, encoding='utf8')
            lines = self.stream.readlines(BUFSIZE)
        
        data = parse_lines(lines[:-1], self.max_len, self.min_len)  # 解析文件行
        random.shuffle(data)  # 打乱数据顺序

        idx = 0  # 初始化索引
        while idx < len(data):  # 当索引小于数据长度时
            yield batchify(data[idx : idx + self.batch_size], self.tknizer)  # 生成批处理数据
            idx += self.batch_size  # 更新索引