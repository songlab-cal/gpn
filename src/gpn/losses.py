"""Loss functions shared by GPN-SS and GPN-MSA."""

from torch.nn import CrossEntropyLoss


def masked_lm_loss(logits, labels, output_probs, loss_weight, vocab_size):
    """Compute hard-label or soft-label masked language-model loss."""

    loss = None
    if labels is not None and loss_weight is None:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
    elif labels is not None and loss_weight is not None:
        loss_fct = CrossEntropyLoss(reduction="none")
        labels = labels.view(-1)
        loss = loss_fct(logits.view(-1, vocab_size), labels)
        loss_weight = loss_weight.view(-1)
        loss_weight[labels == -100] = 0.0
        loss = (loss * loss_weight / loss_weight.sum()).sum()
    elif output_probs is not None:
        loss_fct = CrossEntropyLoss(reduction="none")
        output_probs = output_probs.view(-1, vocab_size)
        exclude = (output_probs == 0.0).all(dim=-1)
        output_probs = output_probs[~exclude]
        logits = logits.view(-1, vocab_size)[~exclude]
        loss_weight = loss_weight.view(-1)[~exclude]
        loss = loss_fct(logits, output_probs)
        loss = (loss * loss_weight / loss_weight.sum()).sum()
    return loss
