import torch


def accuracy(output, target, topk=(1,), condition_on=None):
    """Computes the precision@k for the specified values of k.
    args:
        output:         torch.Variable
        target:         torch.Variable
        topk:           tuple
    returns:
        res:            torch.Variable or [torch.Variable, ..]
    """
    maxk = max(topk)

    if condition_on is not None:
        assert condition_on.shape[0] == output.shape[0]

        output = output[condition_on]
        target = target[condition_on]

    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.div_(batch_size))

    if len(topk) == 1:
        return res[0]
    else:
        return res
