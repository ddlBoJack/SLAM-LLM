import torch
import numpy as np
import sacrebleu
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

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

def compute_bleu(pad_outputs, pad_targets,tokenizer,text_lan):

    pred_ids, label_ids = pad_outputs.cpu(), pad_targets.cpu()
    # In case the model returns more than the prediction logits
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    # pred_ids = np.where(label_ids != -100, pred_ids, tokenizer.pad_token_id)
    # label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    pred_ids[label_ids == -100] = tokenizer.pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]



    if text_lan == "ja":
        text_lan = "ja-mecab"
    elif text_lan == "zh-CN":
        text_lan = "zh"
    else:
        text_lan = "13a"

    result = sacrebleu.corpus_bleu(decoded_preds,[decoded_labels], tokenize=text_lan)


    print(decoded_preds)
    print(decoded_labels)
    print(result)

    return result.score

def compute_wer(pad_outputs, pad_targets,tokenizer,text_lan):

    pred_ids, label_ids = pad_outputs.cpu(), pad_targets.cpu()
    # In case the model returns more than the prediction logits
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]

    # pred_ids = np.where(label_ids != -100, pred_ids, tokenizer.pad_token_id)
    # label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
    pred_ids[label_ids == -100] = tokenizer.pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    normalizer = BasicTextNormalizer()

    import evaluate

    metric = evaluate.load("wer")
    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # compute orthographic wer
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # compute normalised WER
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    # filtering step to only evaluate the samples that correspond to non-zero references:
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)
    print(pred_str_norm)
    print(label_str_norm)
    print(wer)

    # from evaluate import load
    # cer = load("cer")
    # cer_score = cer.compute(predictions=pred_str_norm, references=label_str_norm)
    # print(cer_score)

    return wer

