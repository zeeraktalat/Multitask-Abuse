import torch
from .batching import Batcher, prepare
from sklearn.metrics import mean_absolute_error, f1_score
from scipy.stats import spearmanr


def predict_model(model, data, task_id=0, batch_size=64):
    batcher = Batcher(len(data), batch_size)
    predicted = []
    for size, start, end in batcher:
        d = prepare(data[start:end])
        model.eval()
        pred = model(d, input_task_id=task_id).cpu()
        predicted.extend(pred)
    predicted = torch.stack(predicted).data.numpy().reshape([-1])
    if model.binary:
        predicted = predicted >= 0
    return predicted


def eval_model(model, X, y_true, task_id=0, batch_size=64):
    if model.binary:
        return eval_model_binary(model, X, y_true, task_id=task_id,
                                 batch_size=batch_size)
    else:
        return eval_model_regression(model, X, y_true, task_id=task_id,
                                     batch_size=batch_size)


def eval_model_regression(model, X, y_true, task_id=0, batch_size=64):
    predicted = predict_model(model, X, task_id, batch_size)
    mae, rank_corr = 0, float('nan')
    mae = mean_absolute_error(y_true, predicted)
    if predicted.sum() > 0:
        rank_corr = spearmanr(y_true, predicted)[0]
    return mae, rank_corr, predicted


def eval_model_binary(model, X, y_true, task_id=0, batch_size=64):
    predicted = predict_model(model, X, task_id, batch_size)
    f1 = f1_score(y_true, predicted)
    if predicted.sum() > 0:
        rank_corr = spearmanr(y_true, predicted)[0]
    else:
        rank_corr = float('nan')
    return f1, rank_corr, predicted
