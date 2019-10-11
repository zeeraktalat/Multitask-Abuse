from collections import OrderedDict
from typing import List, Callable, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix


def select_metrics(metrics: List[str], lib: str) -> Dict[str, Callable]:
    """Select metrics for computation based on a list of metric names.
    :param metrics: List of metric names.
    :param lib: Library used (options: ['sklearn', 'pytorch'])
    :return out: Dictionary containing name and methods.
    """
    out = OrderedDict()
    for m in metrics:
        m = m.lower()
        if 'accuracy' in m and 'accuracy' not in out:
            out['accuracy'] = get_accuracy_func(lib)
        elif 'precision' in m and 'precision' not in out:
            out['precision'] = get_precision_func(lib)
        elif 'recall' in m and 'recall' not in out:
            out['recall'] = get_recall_func(lib)
        elif 'auc' in m and 'auc' not in out:
            out['auc'] = get_auc_func(lib)
        elif 'confusion' in m and 'confusion' not in out:
            out['confusion'] = get_confusion_func(lib)

    return out


def get_accuracy_func(lib: str) -> Callable:
    """Select the accuracy computation function depending on the library.
    :param lib: Library used.
    :return func: Accuracy function.
    """

    if lib.lower == 'sklearn':
        func = accuracy_score
    else:
        raise NotImplementedError

    return func


def get_precision_func(lib: str) -> Callable:
    """Select the precision computation function depending on the library.
    :param lib: Library used.
    :return func: precision function.
    """

    if lib.lower == 'sklearn':
        func = precision_score
    else:
        raise NotImplementedError

    return func


def get_recall_func(lib: str) -> Callable:
    """Select the recall computation function depending on the library.
    :param lib: Library used.
    :return func: recall function.
    """

    if lib.lower == 'sklearn':
        func = recall_score
    else:
        raise NotImplementedError

    return func


def get_auc_func(lib: str) -> Callable:
    """Select the auc computation function depending on the library.
    :param lib: Library used.
    :return func: auc function.
    """

    if lib.lower == 'sklearn':
        func = roc_auc_score
    else:
        raise NotImplementedError

    return func


def get_confusion_func(lib: str) -> Callable:
    """Select the confusion_matrix computation function depending on the library.
    :param lib: Library used.
    :return func: confusion_matrix function.
    """

    if lib.lower == 'sklearn':
        func = confusion_matrix
    else:
        raise NotImplementedError

    return func
