"""Валидация, baseline vs финальная модель, ablation по группам признаков, сабмит.

Запуск:  python train.py                 # всё: эксперименты + submission.csv
         python train.py --no-ablation   # быстрее, без поочерёдного удаления групп
Результаты: results/experiments.csv, results/feature_importance.csv,
            results/univariate_auc.csv, results/oof_final.csv, submission.csv
"""
import argparse
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from features import build_features, feature_group, load_data

try:  # официальная реализация метрики из архива
    from metric import precision_at_recall
except ImportError:  # запасной вариант с той же семантикой: равные score отмечаются группой
    def precision_at_recall(y_true, y_score, min_recall=0.7):
        df = pd.DataFrame({'y': y_true, 's': y_score}).groupby('s', sort=True)['y'].agg(['sum', 'size'])[::-1]
        tp, n = df['sum'].cumsum(), df['size'].cumsum()
        prec, rec = tp / n, tp / df['sum'].sum()
        return float(prec[rec >= min_recall].max())

warnings.filterwarnings('ignore', message='.*eval_set.*')

SEED = 42
CV_SEEDS = 3          # сидов на фолд в CV (снимает шум самой модели)
FINAL_SEEDS = 5       # сидов в финальной модели
N_BOOT = 300          # бутстреп для доверительного интервала OOF-метрики
# Скользящая временная валидация: обучение на всех днях до начала блока (минимум 6 дней),
# валидация — следующие 2 дня. Test (20–26.04) идёт сразу после train, как и валидация здесь.
TIME_FOLDS = [('2026-04-12', '2026-04-14'), ('2026-04-14', '2026-04-16'),
              ('2026-04-16', '2026-04-18'), ('2026-04-18', '2026-04-20')]
# Число деревьев фиксировано: early stopping по валидационному фолду подглядывал бы в него же
# и завышал оценку. 400 — порядок best_iteration в предыдущих прогонах с ранней остановкой.
LGB_PARAMS = dict(objective='binary', learning_rate=0.03, n_estimators=400, num_leaves=15,
                  min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                  reg_lambda=5.0, verbose=-1,
                  # детерминизм при многопоточности: фиксированный способ построения гистограмм
                  deterministic=True, force_row_wise=True)
# Группы, не входящие в финальную модель (обоснование — README и EXPERIMENTS.md):
#   x    — популярность объявлений за сутки: после устранения утечки и сдвига train/test прироста не даёт;
#   w    — время на странице по типу события и гистограмма интервалов: дублирует квантили тайминга;
#   leak — демонстрационная протекающая версия x_;
#   u    — UA целиком не берётся; в модель идёт только подмножество UA_PARSED ниже.
EXCLUDED_GROUPS = ('u', 'x', 'w', 'leak')
# Разобранный UA: тип клиента (приложение / браузер), ОС, версия и противоречия в трафике куки
# (несколько платформ при одном UA, UA не сходится с platform). Сырая строка и её частота
# (u_ua_popularity) не используются: это идентификатор конкретного сервиса, а не описание поведения.
UA_PARSED = ['u_n_ua', 'u_n_platform', 'u_share_mismatch', 'u_share_na_platform',
             'u_share_plat_web', 'u_share_plat_desktop', 'u_share_plat_android', 'u_share_plat_ios',
             'u_family', 'u_os', 'u_version', 'u_version_lag', 'u_is_mobile']
# Финальная модель: ранговое среднее LightGBM + CatBoost + ExtraTrees на поведенческих признаках + UA_PARSED.
FINAL_MODEL = 'blend'
FINAL_EXPERIMENT = 'blend_final'


def _rank(p):
    return pd.Series(p).rank(pct=True).to_numpy()


def _numeric(X):
    """Категории -> целочисленные коды (для моделей без нативной поддержки категорий)."""
    return X.apply(lambda c: c.cat.codes if c.dtype.name == 'category' else c)


def fit_predict(Xtr, ytr, Xte, model, seeds, params=None):
    """Обучение и предсказание; для LightGBM дополнительно важность признаков (gain)."""
    if model in ('rf', 'et', 'cat'):
        Xtr, Xte = _numeric(Xtr), _numeric(Xte)
    if model in ('rf', 'et'):
        if model == 'rf':
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1)
        else:
            m = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3, max_features=0.3,
                                     random_state=SEED, n_jobs=-1)
        m.fit(Xtr.fillna(-1), ytr)
        # предсказание в один поток: при n_jobs>1 sklearn суммирует вклады деревьев в произвольном
        # порядке, и последние знаки float плавают от запуска к запуску
        m.set_params(n_jobs=1)
        return m.predict_proba(Xte.fillna(-1))[:, 1], None
    if model == 'cat':
        from catboost import CatBoostClassifier
        preds = [CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6, random_seed=SEED + s,
                                    thread_count=4, verbose=0, allow_writing_files=False).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                 for s in range(seeds)]
        return np.mean(preds, axis=0), None
    if model == 'blend':
        # ранговое среднее: модели выдают вероятности в разных шкалах, а метрике важен только порядок
        p_lgb, imp = fit_predict(Xtr, ytr, Xte, 'lgb', seeds, params)
        p_cat, _ = fit_predict(Xtr, ytr, Xte, 'cat', seeds)
        p_et, _ = fit_predict(Xtr, ytr, Xte, 'et', 1)
        # сумма трёх рангов часто совпадает у разных кук, а метрика отмечает равные score группой;
        # ничьи разбиваем средним вероятностей (осмысленный вторичный скор, не случайный шум)
        tie_break = (p_lgb + p_cat + p_et) / 3 * 1e-6
        return (_rank(p_lgb) + _rank(p_cat) + _rank(p_et)) / 3 * (1 - 1e-6) + tie_break, imp
    preds, imp = [], 0
    for s in range(seeds):
        m = lgb.LGBMClassifier(**{**LGB_PARAMS, **(params or {})}, random_state=SEED + s).fit(Xtr, ytr)
        preds.append(m.predict_proba(Xte)[:, 1])
        imp = imp + pd.Series(m.booster_.feature_importance('gain'), index=Xtr.columns)
    return np.mean(preds, axis=0), imp / seeds


def time_cv(X, y, dates, cols, model='lgb', params=None, train_mask=None):
    """OOF-предсказания по временным фолдам + метрики.
    Главная цифра — P@R70 по склеенным OOF всех фолдов (~400 ботов, стабильнее, чем по одному фолду)."""
    oof = pd.Series(np.nan, index=X.index)
    per_fold = []
    for start, end in TIME_FOLDS:
        tr = (dates < start).to_numpy()
        if train_mask is not None:   # LOCO: часть объектов исключается из обучения
            tr = tr & train_mask
        va = ((dates >= start) & (dates < end)).to_numpy()
        p, _ = fit_predict(X.loc[tr, cols], y[tr], X.loc[va, cols], model, CV_SEEDS, params)
        oof[va] = p
        per_fold.append(precision_at_recall(y[va], p))
    m = oof.notna().to_numpy()
    yo, po = y[m], oof[m].to_numpy()
    return {'p_at_r70_oof': precision_at_recall(yo, po), 'p_at_r70_fold_mean': np.mean(per_fold),
            'p_at_r70_folds': ' '.join(f'{v:.3f}' for v in per_fold),
            'pr_auc_oof': average_precision_score(yo, po), 'roc_auc_oof': roc_auc_score(yo, po)}, oof


def bootstrap_ci(y, s, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    vals = [precision_at_recall(y[i], s[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return np.percentile(vals, [5, 95])


def prepare(data_dir='data'):
    """Загрузка + признаки. Общая точка входа для train.py и validate.py."""
    train, test, events = load_data(data_dir)
    Xtr, Xte, stats = build_features(events, train, test)
    y = train['target'].to_numpy()
    dates = train['window_start_ts'].reset_index(drop=True)
    Xtr = Xtr.reset_index(drop=True)
    all_cols = [c for c in Xtr.columns if feature_group(c) != 'leak']
    final_cols = [c for c in all_cols if feature_group(c) not in EXCLUDED_GROUPS] + UA_PARSED
    print('данные:', stats, f'| признаков: всего {len(all_cols)}, в финальной модели {len(final_cols)}')
    return train, test, Xtr, Xte, y, dates, all_cols, final_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--no-ablation', action='store_true')
    ap.add_argument('--only', default='', help='запустить только эти эксперименты (через запятую)')
    args = ap.parse_args()
    Path('results').mkdir(exist_ok=True)

    t0 = time.time()
    train, test, Xtr, Xte, y, dates, all_cols, final_cols = prepare(args.data)

    # одномерная диагностика: ROC-AUC каждого признака (симметризован: 0.5 = бесполезен)
    uni = {c: roc_auc_score(y, Xtr[c].astype(float).fillna(-1e9)) for c in all_cols
           if Xtr[c].dtype.name != 'category'}
    uni = pd.Series(uni).sub(0.5).abs().add(0.5).sort_values(ascending=False).round(4)
    uni.to_csv('results/univariate_auc.csv', header=['auc'])

    def group(g):
        return [c for c in Xtr.columns if feature_group(c) == g]

    def without(*groups):
        return [c for c in final_cols if feature_group(c) not in groups]

    behavior_cols = without('u')
    pointer_basic = ['p_share_missing_web', 'p_n', 'p_uniq_share', 'p_std_x', 'p_std_y']
    experiments = {
        'baseline_rf_quickstart': ('rf', ['v_n_events', 'c_n_item']),   # повтор quickstart
        'baseline_lgb_quickstart': ('lgb', ['v_n_events', 'c_n_item']),
        FINAL_EXPERIMENT: (FINAL_MODEL, final_cols),
        # модель и набор признаков по отдельности
        'lgb_final_features': ('lgb', final_cols),
        'cat_final_features': ('cat', final_cols),
        'lgb_behavior_only': ('lgb', behavior_cols),
        'blend_behavior_only': ('blend', behavior_cols),
        # отвергнутые варианты (см. README)
        'lgb_with_ua_popularity': ('lgb', final_cols + ['u_ua_popularity']),
        'lgb_with_itempop': ('lgb', final_cols + group('x')),
        'lgb_with_itempop_leaky': ('lgb', final_cols + group('leak')),
        'lgb_with_dwell': ('lgb', final_cols + group('w')),
        'lgb_pointer_basic': ('lgb', without('p') + pointer_basic),
    }
    if not args.no_ablation:   # ablation на LightGBM (быстрее ансамбля, порядок вкладов тот же)
        for g in sorted({feature_group(c) for c in final_cols}):
            experiments[f'lgb_drop_{g}'] = ('lgb', without(g))

    if args.only:
        experiments = {k: v for k, v in experiments.items() if k in args.only.split(',')}
    log = []
    for name, (model, cols) in experiments.items():
        r, oof = time_cv(Xtr, y, dates, cols, model)
        if name == FINAL_EXPERIMENT:
            mask = oof.notna().to_numpy()
            lo, hi = bootstrap_ci(y[mask], oof[mask].to_numpy())
            r['ci90'] = f'{lo:.3f}-{hi:.3f}'
            pd.DataFrame({'cookie_id': train['cookie_id'], 'target': y, 'oof': oof}).dropna() \
              .to_csv('results/oof_final.csv', index=False)
        log.append({'experiment': name, 'n_features': len(cols), **r})
        print(f"{name:26s} P@R70 OOF={r['p_at_r70_oof']:.4f} fold_mean={r['p_at_r70_fold_mean']:.4f} "
              f"[{r['p_at_r70_folds']}] PR-AUC={r['pr_auc_oof']:.4f} ROC-AUC={r['roc_auc_oof']:.4f}"
              + (f"  CI90 {r['ci90']}" if 'ci90' in r else ''), flush=True)
    out = 'results/experiments.csv' if not args.only else 'results/experiments_only.csv'
    pd.DataFrame(log).round(4).to_csv(out, index=False)

    # финал: весь train, финальный набор признаков, усреднение по FINAL_SEEDS сидам
    score, imp = fit_predict(Xtr[final_cols], y, Xte[final_cols], FINAL_MODEL, FINAL_SEEDS)
    imp = imp.sort_values(ascending=False)
    imp.round(1).to_csv('results/feature_importance.csv', header=['gain'])
    print('топ-15 по gain (LightGBM-часть):\n' + (imp / imp.sum()).head(15).round(3).to_string())

    sub = pd.DataFrame({'cookie_id': test['cookie_id'].to_numpy(), 'score': score})
    assert len(sub) == len(test) and sub['cookie_id'].is_unique
    assert sub['score'].notna().all() and sub['score'].between(0, 1).all()
    sub.to_csv('submission.csv', index=False)
    print(f'submission.csv: {len(sub)} строк, уникальных score: {sub.score.nunique()} | {time.time() - t0:.0f} c')


if __name__ == '__main__':
    main()