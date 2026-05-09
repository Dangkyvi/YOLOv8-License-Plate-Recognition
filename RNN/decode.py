def decode(pred, chars):
    pred = pred.argmax(dim=2)
    pred = pred.squeeze(1).tolist()

    res = ""
    prev = -1
    for p in pred:
        if p != prev and p != 0:
            res += chars[p-1]
        prev = p
    return res