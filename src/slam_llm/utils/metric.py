import torch
import numpy as np

def compute_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (LongTensor): Prediction tensors (B, Lmax).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_outputs.masked_select(mask) == pad_targets.masked_select(mask)
    )
    denominator = torch.sum(mask)
    return numerator.float() / denominator.float() #(FIX:MZY):return torch.Tensor type

def compute_bleu(pad_outputs, pad_targets,tokenizer,metric):

    preds, labels = pad_outputs.cpu(), pad_targets.cpu()
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    preds = np.where(labels != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,tokenize="13a")


    print(decoded_preds)
    print(decoded_labels)
    print(result)

    return result["score"]