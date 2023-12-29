from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def getAUC(y_true, y_score):
    y_score = y_score[:,-1]
    return roc_auc_score(y_true, y_score)



def getACC(y_true, y_score, threshold=0.5):
    y_pre = np.zeros_like(y_true)
    for i in range(y_score.shape[0]):
        y_pre[i] = (y_score[i][-1] > threshold)
    return accuracy_score(y_true, y_pre)


def save_results(y_true, y_score, outputpath):
    idx = []

    idx.append('id')

    for i in range(y_true.shape[1]):
        idx.append('true_%s' % (i))
    for i in range(y_score.shape[1]):
        idx.append('score_%s' % (i))

    df = pd.DataFrame(columns=idx)
    for id in range(y_score.shape[0]):
        dic = {}
        dic['id'] = id
        for i in range(y_true.shape[1]):
            dic['true_%s' % (i)] = y_true[id][i]
        for i in range(y_score.shape[1]):
            dic['score_%s' % (i)] = y_score[id][i]

        df_insert = pd.DataFrame(dic, index = [0])
        df = pd.concat([df.dropna(axis=1, how='all'), df_insert], ignore_index=True)
    df.to_csv(outputpath, sep=',', index=False, header=True, encoding="utf_8_sig")
