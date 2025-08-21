import numpy as np

def softmax(logits, temperature=None):
    logits = np.array(logits) / temperature if temperature else np.array(logits)
    m = max(logits)
    logits = logits-m
    exp_vals = np.exp(logits)
    return exp_vals / np.sum(exp_vals)


def only_s(logits, temperature):
    return softmax(logits, temperature)

def c_and_s(logits, temperature):
    logits_centered = logits - np.mean(logits)
    return softmax(logits_centered, temperature)

# 원래 로짓
logits = [7, 5, 2]

print(logits)
print("그냥 소프트맥스  :", softmax(logits))
print("sharpening   :", only_s(logits, 0.04))
print("둘다          :", c_and_s(logits, 0.04))
print()

logits = [1.1, 0.7, 0.9]
print(logits)
print("그냥 소프트맥스  :", softmax(logits))
print("sharpening   :", only_s(logits, 0.04))
print("둘다          :", c_and_s(logits, 0.04))


logits = [70, 50, 2]

print(logits)
print("그냥 소프트맥스  :", softmax(logits))
print("sharpening   :", only_s(logits, 0.04))
print("둘다          :", c_and_s(logits, 0.04))
print()