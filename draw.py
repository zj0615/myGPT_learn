import re
import matplotlib.pyplot as plt

# 定义正则表达式来提取参数
pattern = r"batch_acm (\d+), loss ([\d.]+), acc ([\d.]+), nll ([\d.]+), ppl ([\d.]+), x_acm (\d+), lr ([\d.eE\-]+)"

# 初始化存储每个参数的列表
batch_acm, loss, acc, nll, ppl, x_acm, lr,  = [], [], [], [], [], [], []

# 读取文件并解析每一行
with open("./ckpt/sft/train1.log", "r") as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            batch_acm.append(int(match.group(1)))
            loss.append(float(match.group(2)))
            acc.append(float(match.group(3)))
            nll.append(float(match.group(4)))
            ppl.append(float(match.group(5)))
            x_acm.append(int(match.group(6)))
            lr.append(float(match.group(7)))


# 绘制每个参数的变化曲线
plt.figure(figsize=(12, 10))

# 绘制各参数变化
plt.subplot(3, 3, 1)
plt.plot(batch_acm, loss, label="Loss", color='b')
plt.xlabel("Batch ACM")
plt.ylabel("Loss")
plt.title("Loss vs Batch ACM")

plt.subplot(3, 3, 2)
plt.plot(batch_acm, acc, label="Accuracy", color='g')
plt.xlabel("Batch ACM")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Batch ACM")

plt.subplot(3, 3, 3)
plt.plot(batch_acm, nll, label="NLL", color='r')
plt.xlabel("Batch ACM")
plt.ylabel("NLL")
plt.title("NLL vs Batch ACM")

plt.subplot(3, 3, 4)
plt.plot(batch_acm[10:], ppl[10:], label="PPL", color='purple')
plt.xlabel("Batch ACM")
plt.ylabel("PPL")
plt.title("PPL vs Batch ACM")

plt.subplot(3, 3, 5)
plt.plot(batch_acm, x_acm, label="X_ACM", color='orange')
plt.xlabel("Batch ACM")
plt.ylabel("X_ACM")
plt.title("X_ACM vs Batch ACM")

plt.subplot(3, 3, 6)
plt.plot(batch_acm, lr, label="Learning Rate", color='brown')
plt.xlabel("Batch ACM")
plt.ylabel("Learning Rate")
plt.title("Learning Rate vs Batch ACM")

plt.tight_layout()

# 保存为图像文件
plt.savefig("./ckpt/sft/output.png", dpi=300)  # dpi=300用于提高图像分辨率
