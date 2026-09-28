"""Проверки устойчивости (не нужны для получения submission.csv). Модель в проверках — LightGBM на
финальном наборе признаков: он в разы быстрее ансамбля, а выводы об устойчивости у них общие.

1. Adversarial validation: насколько test отличим от train по признакам финальной модели.
2. Кластеры ботов (прокси «сервисов сбора») + leave-one-cluster-out: ловит ли модель
   ботов типа, которого не было в обучении.
3. Лёгкий тюнинг LightGBM на той же временной CV.

Запуск: python validate.py [--skip-tune] [--loco-only]
Результаты: results/adversarial.csv, results/bot_clusters.csv, results/loco.csv, results/tuning.csv
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

from features import feature_group
from train import SEED, precision_at_recall, prepare, time_cv

PROFILE = ['v_n_events', 't_dt_q50', 't_dt_q90', 'p_share_missing_web', 'p_step_mean', 'p_share_border',
           's_page_max', 'c_n_loc', 'x_item_pop_mean', 'h_share_night', 'a_age_days']


def adversarial(Xtr, Xte, cols):
    X = pd.concat([Xtr[cols], Xte[cols]], ignore_index=True)
    t = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    oof, imp = np.zeros(len(X)), 0
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, t):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                               random_state=SEED, verbose=-1).fit(X.iloc[tr], t[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        imp = imp + pd.Series(m.booster_.feature_importance('gain'), index=cols)
    auc = roc_auc_score(t, oof)
    top = (imp / imp.sum()).sort_values(ascending=False).head(10).round(3)
    print(f'\n=== Adversarial validation: ROC-AUC train vs test = {auc:.4f} (0.5 = неотличимы)')
    print('признаки, по которым test отличается сильнее всего:\n' + top.to_string())
    top.to_csv('results/adversarial.csv', header=['gain_share'])


def cluster_bots(Xtr, y, cols, k_range=range(2, 7)):
    """KMeans по ботам train в пространстве квантильно нормализованных признаков (+PCA)."""
    B = Xtr.loc[y == 1, cols].select_dtypes('number')   # только числовые признаки (категории UA не нужны)
    B = B.fillna(B.median())
    Z = QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=SEED).fit_transform(B)
    Z = PCA(n_components=10, random_state=SEED).fit_transform(Z)
    scores = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)) for k in k_range}
    k = max(scores, key=scores.get)
    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    print('\n=== Кластеры ботов: silhouette по k:', {kk: round(v, 3) for kk, v in scores.items()}, '→ k =', k)
    clusters = pd.Series(-1, index=Xtr.index)   # -1 = люди
    clusters[y == 1] = lab
    prof = Xtr[PROFILE].groupby(clusters).median().round(2)
    prof.insert(0, 'size', clusters.value_counts())
    prof.index = ['humans' if i == -1 else f'bots_{i}' for i in prof.index]
    print('медианы ключевых признаков по кластерам:\n' + prof.T.to_string())
    prof.to_csv('results/bot_clusters.csv')
    return clusters


def loco(Xtr, y, dates, cols, clusters):
    """Для каждого кластера: P@R70 на (люди + боты кластера) в валидации, когда кластер был в обучении
    и когда его боты из обучения исключены. Падение = зависимость от «знакомого» сервиса."""
    _, base = time_cv(Xtr, y, dates, cols)
    rows = []
    for c in sorted(set(clusters) - {-1}):
        _, held = time_cv(Xtr, y, dates, cols, train_mask=(clusters != c).to_numpy())
        ev = (base.notna() & clusters.isin([-1, c])).to_numpy()
        r = {'cluster': f'bots_{c}', 'n_bots_val': int(((clusters == c) & base.notna()).sum()),
             'p_at_r70_seen': precision_at_recall(y[ev], base[ev]),
             'p_at_r70_unseen': precision_at_recall(y[ev], held[ev]),
             'roc_auc_seen': roc_auc_score(y[ev], base[ev]), 'roc_auc_unseen': roc_auc_score(y[ev], held[ev])}
        rows.append(r)
    res = pd.DataFrame(rows).round(4)
    print('\n=== Leave-one-cluster-out\n' + res.to_string(index=False))
    res.to_csv('results/loco.csv', index=False)


def tune(Xtr, y, dates, cols):
    grid = {
        'base': {},
        'leaves7': {'num_leaves': 7},
        'leaves31': {'num_leaves': 31},
        'min_child60': {'min_child_samples': 60},
        'colsample0.3': {'colsample_bytree': 0.3},
        'reg_lambda20': {'reg_lambda': 20.0},
        'trees800': {'n_estimators': 800},
        'lr0.015_trees800': {'learning_rate': 0.015, 'n_estimators': 800},
        'scale_pos_weight3': {'scale_pos_weight': 3.0},
    }
    rows = []
    for name, params in grid.items():
        r, _ = time_cv(Xtr, y, dates, cols, params=params)
        rows.append({'config': name, **{k: v for k, v in r.items() if k != 'p_at_r70_folds'},
                     'folds': r['p_at_r70_folds']})
        print(f"{name:20s} P@R70 OOF={r['p_at_r70_oof']:.4f} [{r['p_at_r70_folds']}] "
              f"PR-AUC={r['pr_auc_oof']:.4f} ROC-AUC={r['roc_auc_oof']:.4f}", flush=True)
    pd.DataFrame(rows).round(4).to_csv('results/tuning.csv', index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--skip-tune', action='store_true')
    ap.add_argument('--loco-only', action='store_true', help='только кластеры + LOCO')
    args = ap.parse_args()
    Path('results').mkdir(exist_ok=True)
    _, _, Xtr, Xte, y, dates, _, cols = prepare(args.data)
    if not args.loco_only:
        adversarial(Xtr, Xte, cols)
    # кластеры строим по поведенческим признакам (без UA), чтобы группы описывали поведение сервисов
    clusters = cluster_bots(Xtr, y, [c for c in cols if feature_group(c) != 'u'])
    loco(Xtr, y, dates, cols, clusters)
    if not (args.skip_tune or args.loco_only):
        print('\n=== Тюнинг (временная CV, OOF)')
        tune(Xtr, y, dates, cols)


if __name__ == '__main__':
    main()