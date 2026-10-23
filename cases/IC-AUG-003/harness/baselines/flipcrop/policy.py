"""IC-AUG-003 baseline run by the evaluator: the standard CIFAR-10 augmentation, a random horizontal flip and a random
translation by up to 4 pixels."""


def augment(x, ops, gen, epoch, n_epochs):
    return ops.translate(ops.flip_lr(x, gen), 4, gen)
