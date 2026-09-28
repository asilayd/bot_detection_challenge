"""Загрузка, очистка событий и построение признаков на уровне cookie_id.

Все признаки считаются по событиям внутри окна [window_start_ts, window_end_ts),
то есть известны на момент окончания окна. Таргет здесь не используется, поэтому
признаки строятся один раз для train+test (одинаковые категории и кодировки).

Префикс имени признака = группа (используется в ablation в train.py):
    v_  объём и сессии            t_  межсобытийные интервалы (тайминг)
    w_  время на странице по типу события и форма распределения интервалов
    e_  типы событий и переходы   c_  разнообразие контента
    s_  поиск и пагинация         p_  указатель (pointer)
    u_  user-agent и платформа    h_  время суток
    a_  возраст куки и история    x_  кросс-куковые (трансдуктивные, по train+test)
    leak_  НЕКОРРЕКТНАЯ версия x_ с заглядыванием в будущее; в модели не используется
"""
from pathlib import Path

import numpy as np
import pandas as pd

SESSION_GAP_S = 30 * 60       # разрыв > 30 мин начинает новую сессию
POP_LOOKBACK = pd.Timedelta('1D')   # глубина истории для популярности объявлений (см. cross_cookie)
TEXT_COLS = ['search_query', 'item_category', 'item_location', 'seller_type']
TIME_COLS = ['cookie_created_at', 'window_start_ts', 'window_end_ts']
EVENT_NAMES = ['search_results_view', 'item_view', 'photo_swipe', 'seller_page_view',
               'favorite_add', 'contact_phone_show', 'contact_chat_open',
               'contact_message_sent', 'login', 'captcha_shown']
CONTACT_EVENTS = ['contact_phone_show', 'contact_chat_open', 'contact_message_sent']
UA_FAMILIES = ['avito_app', 'yandex', 'edge', 'opera', 'firefox', 'chrome', 'safari', 'other']
UA_OS = ['windows', 'mac', 'linux', 'android', 'ios', 'other']


# ----------------------------------------------------------------------------- загрузка
def load_data(data_dir='data'):
    d = Path(data_dir)
    train = pd.read_csv(d / 'train.csv', parse_dates=TIME_COLS)
    test = pd.read_csv(d / 'test.csv', parse_dates=TIME_COLS)
    events = pd.read_csv(d / 'events.csv.gz', parse_dates=['event_ts'])
    return train, test, events


def clean_events(events, meta):
    """Нормализация, дедупликация, окно, сортировка. Возвращает (события в окне, служебная статистика по куке)."""
    ev = events.drop(columns=['eid'])  # eid ↔ event_name один к одному
    # platform записана в разных регистрах; iphone и ios — одно и то же
    ev['platform'] = ev['platform'].str.strip().str.lower().replace({'iphone': 'ios'})
    # текстовые категории: регистр и пробелы по краям не несут смысла; считаем, сколько значений схлопнулось
    merged = {}
    for col in TEXT_COLS:
        before = ev[col].nunique()
        ev[col] = ev[col].str.strip().str.lower()
        merged[col] = before - ev[col].nunique()

    # полные дубли (ретраи логирования/запросов): удаляем, но долю сохраняем как признак
    is_dup = ev.duplicated()
    n_dup = is_dup.groupby(ev['cookie_id']).sum().rename('n_dup')
    ev = ev[~is_dup]

    ev = ev.merge(meta[['cookie_id'] + TIME_COLS], on='cookie_id', how='inner')
    # предыстория куки до окна — легальна (известна к концу окна); после окна — выбрасываем
    n_prior = (ev['event_ts'] < ev['window_start_ts']).groupby(ev['cookie_id']).sum().rename('n_prior')
    in_win = (ev['event_ts'] >= ev['window_start_ts']) & (ev['event_ts'] < ev['window_end_ts'])
    ev = ev[in_win]

    # файл не отсортирован по времени — без сортировки все Δt и переходы были бы мусором
    ev = ev.sort_values(['cookie_id', 'event_ts'], kind='mergesort').reset_index(drop=True)
    aux = pd.concat([n_dup, n_prior], axis=1).fillna(0)
    aux.attrs['text_values_merged'] = merged
    return ev, aux


# ----------------------------------------------------------------------------- утилиты
def _entropy(counts: pd.DataFrame) -> pd.Series:
    p = counts.div(counts.sum(axis=1), axis=0)
    return -(p * np.log(p.where(p > 0))).sum(axis=1)


def _share(mask: pd.Series, by: pd.Series) -> pd.Series:
    return mask.astype(float).groupby(by).mean()


def _parse_ua(ua: pd.Series) -> pd.DataFrame:
    s = ua.fillna('')
    fam = np.select(
        [s.str.startswith('Avito/'), s.str.contains('YaBrowser'), s.str.contains('Edg/'),
         s.str.contains('OPR/'), s.str.contains('Firefox/'), s.str.contains('Chrome/'),
         s.str.contains('Safari/')],
        UA_FAMILIES[:-1], default='other')
    os_ = np.select(
        [s.str.contains('Windows'), s.str.contains('Macintosh'), s.str.contains('Android'),
         s.str.contains('iPhone|iPad|iOS'), s.str.contains('Linux')],
        ['windows', 'mac', 'android', 'ios', 'linux'], default='other')
    ver = s.str.extract(r'(?:Chrome|Firefox|YaBrowser|Edg|OPR|Version|Avito)/(\d+)')[0].astype(float)
    out = pd.DataFrame({'fam': fam, 'os': os_, 'ver': ver}, index=ua.index)
    # отставание версии от самой свежей в своём семействе (старые версии — типичны для скриптов)
    out['ver_lag'] = out.groupby('fam')['ver'].transform('max') - out['ver']
    out['is_mobile'] = out['os'].isin(['android', 'ios']) | s.str.contains('Mobile')
    return out


# ----------------------------------------------------------------------------- группы признаков
def volume_timing(ev):
    g = ev['cookie_id']
    dt = ev.groupby(g)['event_ts'].diff().dt.total_seconds()
    new_sess = dt.isna() | (dt > SESSION_GAP_S)
    sess = new_sess.groupby(g).cumsum()
    dt_in = dt.where(~new_sess)  # только интервалы внутри сессии: паузы между визитами не про темп
    f = pd.DataFrame(index=g.unique())
    f['v_n_events'] = g.value_counts()
    f['v_n_sessions'] = new_sess.groupby(g).sum()
    f['v_events_per_session'] = f['v_n_events'] / f['v_n_sessions']
    f['v_span_min'] = ev.groupby(g)['event_ts'].agg(lambda s: (s.max() - s.min()).total_seconds() / 60)
    sess_len = ev.groupby([g, sess]).size()
    f['v_max_session_events'] = sess_len.groupby(level=0).max()
    sess_dur = ev.groupby([g, sess])['event_ts'].agg(lambda s: (s.max() - s.min()).total_seconds())
    f['v_active_min'] = sess_dur.groupby(level=0).sum() / 60
    f['v_events_per_active_min'] = f['v_n_events'] / f['v_active_min'].clip(lower=1)
    minute = ev['event_ts'].dt.floor('min')
    f['v_max_events_per_min'] = ev.groupby([g, minute]).size().groupby(level=0).max()

    gd = dt_in.groupby(g)
    f['t_n_intervals'] = gd.count()
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        f[f't_dt_q{int(q * 100)}'] = gd.quantile(q)
    f['t_dt_mean'] = gd.mean()
    f['t_dt_std'] = gd.std()
    f['t_dt_min'] = gd.min()
    f['t_dt_cv'] = f['t_dt_std'] / f['t_dt_mean'].replace(0, np.nan)
    # Робастная «регулярность»: IQR/медиана по логу интервалов
    ldt = np.log1p(dt_in)
    f['t_logdt_std'] = ldt.groupby(g).std()
    for thr in (2, 5, 10, 30):
        f[f't_share_dt_le{thr}'] = (dt_in <= thr).groupby(g).sum() / f['t_n_intervals'].replace(0, np.nan)
    # доля интервалов, совпадающих с предыдущим с точностью до 1 с (ритмичность)
    same = (dt_in.round() == dt_in.groupby(g).shift().round()) & dt_in.notna()
    f['t_share_repeat_dt'] = same.groupby(g).sum() / f['t_n_intervals'].replace(0, np.nan)
    # моменты высших порядков шумны на коротких куках — считаем только при n >= 8
    skew = gd.skew()
    f['t_dt_skew'] = skew.where(f['t_n_intervals'] >= 8)
    return f, dt_in, new_sess


def event_mix(ev, new_sess):
    g = ev['cookie_id']
    cnt = pd.crosstab(g, ev['event_name']).reindex(columns=EVENT_NAMES, fill_value=0)
    f = cnt.div(cnt.sum(axis=1), axis=0).add_prefix('e_share_')
    f['e_n_captcha'] = cnt['captcha_shown']
    f['e_n_contacts'] = cnt[CONTACT_EVENTS].sum(axis=1)
    f['e_n_types'] = (cnt > 0).sum(axis=1)
    f['e_type_entropy'] = _entropy(cnt)

    # переходы между соседними событиями внутри сессии
    prev = ev.groupby(g)['event_name'].shift().where(~new_sess)
    has = prev.notna()
    big = (prev + '>' + ev['event_name']).where(has)
    n_tr = has.groupby(g).sum().replace(0, np.nan)
    f['e_self_transition'] = (has & (prev == ev['event_name'])).groupby(g).sum() / n_tr
    f['e_bigram_diversity'] = big.groupby(g).nunique() / n_tr
    # контакт сразу из выдачи, без открытия объявления — невозможно для человека в интерфейсе
    f['e_contact_from_search'] = ((prev == 'search_results_view') & ev['event_name'].isin(CONTACT_EVENTS)).groupby(g).sum()
    # топ биграмм по частоте в данных (без таргета) — доли внутри куки
    top = big.value_counts(normalize=True)
    top = top[top >= 0.01].index
    bc = pd.crosstab(g[has], big[has]).reindex(columns=top, fill_value=0)
    bc = bc.div(bc.sum(axis=1), axis=0)
    bc.columns = ['e_bg_' + c.replace('>', '__') for c in bc.columns]
    return f.join(bc)


def content(ev):
    g = ev['cookie_id']
    f = pd.DataFrame(index=g.unique())
    for col, name in [('item_id', 'item'), ('item_category', 'cat'), ('item_location', 'loc'),
                      ('seller_type', 'seller')]:
        f[f'c_n_{name}'] = ev.groupby(g)[col].nunique()
    has_item = ev['item_id'].notna()
    f['c_item_repeat'] = has_item.groupby(g).sum() / f['c_n_item'].replace(0, np.nan)
    f['c_item_per_event'] = f['c_n_item'] / g.value_counts()
    f['c_cat_entropy'] = _entropy(pd.crosstab(g, ev['item_category']))
    f['c_loc_entropy'] = _entropy(pd.crosstab(g, ev['item_location']))
    f['c_share_pro_seller'] = _share(ev['seller_type'].eq('pro'), g).where(f['c_n_seller'] > 0)
    # фото на просмотр объявления: человек листает фото, сборщик — нет
    n_iv = ev['event_name'].eq('item_view').groupby(g).sum()
    f['c_photo_per_item_view'] = ev['event_name'].eq('photo_swipe').groupby(g).sum() / n_iv.replace(0, np.nan)
    return f


def search(ev):
    s = ev[ev['search_page'].notna()].copy()
    g = s['cookie_id']
    f = pd.DataFrame(index=g.unique())
    gp = s.groupby(g)['search_page']
    f['s_n_search'] = gp.size()
    f['s_page_max'] = gp.max()
    f['s_page_mean'] = gp.mean()
    f['s_share_page1'] = _share(s['search_page'].eq(1), g)
    f['s_n_query'] = s.groupby(g)['search_query'].nunique()
    f['s_search_per_query'] = f['s_n_search'] / f['s_n_query'].replace(0, np.nan)
    f['s_query_len'] = s['search_query'].str.len().groupby(g).mean()
    # линейность обхода: тот же запрос и страница ровно +1 к предыдущей выдаче
    pq = s.groupby(g)['search_query'].shift()
    pp = s.groupby(g)['search_page'].shift()
    pair = pp.notna()
    f['s_share_next_page'] = ((pq == s['search_query']) & (s['search_page'] == pp + 1)).groupby(g).sum() \
        / pair.groupby(g).sum().replace(0, np.nan)
    f['s_share_page_back'] = ((pq == s['search_query']) & (s['search_page'] < pp)).groupby(g).sum() \
        / pair.groupby(g).sum().replace(0, np.nan)
    f['s_max_pages_per_query'] = s.groupby([g, s['search_query']])['search_page'].nunique().groupby(level=0).max()
    return f


def pointer(ev):
    """Координаты курсора. EDA: у людей точки равномерны по экрану 1920x1080 и независимы
    (std x ≈ 1920/√12), у ботов — блуждание с малым шагом вокруг своей области и упоры в границы."""
    g = ev['cookie_id']
    web = ev['platform'].isin(['web', 'desktop'])   # на android/ios координат нет ни у кого
    has = ev['pointer_x'].notna()
    f = pd.DataFrame(index=g.unique())
    f['p_share_missing_web'] = (~has & web).groupby(g).sum() / web.groupby(g).sum().replace(0, np.nan)

    p = ev.loc[has, ['cookie_id', 'pointer_x', 'pointer_y']].copy()   # ev уже отсортирован по времени
    gp = p['cookie_id']
    x, y = p['pointer_x'], p['pointer_y']
    n = gp.value_counts()
    f['p_n'] = n
    f['p_uniq_share'] = (x.astype(str) + '_' + y.astype(str)).groupby(gp).nunique() / n
    for ax, v in (('x', x), ('y', y)):
        gv = v.groupby(gp)
        f[f'p_std_{ax}'] = gv.std()
        f[f'p_mean_{ax}'] = gv.mean()
        f[f'p_range_{ax}'] = gv.max() - gv.min()
        # лаг-1 автокорреляция: блуждание ≈ 1, независимые клики ≈ 0 (считаем при >= 5 точках)
        prev = gv.shift()
        ok = prev.notna()
        f[f'p_autocorr_{ax}'] = pd.concat([v[ok], prev[ok]], axis=1, keys=['a', 'b']) \
            .groupby(gp[ok]).corr().xs('a', level=1)['b'].where(n >= 5)
    # шаг между соседними точками, в пикселях и относительно разброса точек куки
    step = np.hypot(x.groupby(gp).diff(), y.groupby(gp).diff())
    f['p_step_median'] = step.groupby(gp).median()
    f['p_step_mean'] = step.groupby(gp).mean()
    f['p_step_norm'] = f['p_step_mean'] / np.hypot(f['p_std_x'], f['p_std_y']).replace(0, np.nan)
    # упор в границы экрана (результат обрезки блуждания) и «круглые» координаты
    f['p_share_border'] = _share(x.isin([0, 1920]) | y.isin([0, 1080]), gp)
    f['p_share_round100'] = _share((x % 100 == 0) | (y % 100 == 0), gp)
    f['p_n'] = f['p_n'].fillna(0)
    return f


def ua_platform(ev):
    g = ev['cookie_id']
    f = pd.DataFrame(index=g.unique())
    f['u_n_ua'] = ev.groupby(g)['user_agent'].nunique()
    f['u_n_platform'] = ev.groupby(g)['platform'].nunique()
    pc = pd.crosstab(g, ev['platform'].fillna('na'))
    f = f.join(pc.div(pc.sum(axis=1), axis=0).add_prefix('u_share_plat_'))

    ua = _parse_ua(ev['user_agent'])
    plat = ev['platform']
    # несоответствие UA и platform: сырой UA не используем, но противоречие — признак подделки
    mismatch = (
        (ua['os'].isin(['windows', 'mac', 'linux']) & plat.isin(['android', 'ios']))
        | (ua['os'].eq('android') & plat.eq('ios'))
        | (ua['os'].eq('ios') & plat.eq('android'))
        | (ua['fam'].eq('avito_app') & plat.isin(['web', 'desktop']))
        | (~ua['is_mobile'] & ua['fam'].ne('avito_app') & plat.isin(['android', 'ios']))
    )
    f['u_share_mismatch'] = _share(mismatch, g)
    f['u_share_na_platform'] = _share(plat.isna(), g)

    # разобранные поля берём по самому частому UA куки
    mode_ua = ev.groupby(g)['user_agent'].agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan)
    pu = _parse_ua(mode_ua)
    f['u_family'] = pd.Categorical(pu['fam'], categories=UA_FAMILIES)
    f['u_os'] = pd.Categorical(pu['os'], categories=UA_OS)
    f['u_version'] = pu['ver']
    f['u_version_lag'] = pu['ver_lag']
    f['u_is_mobile'] = pu['is_mobile'].astype(int)
    # популярность UA (сколько кук с ним во всём train+test) вместо самой строки
    f['u_ua_popularity'] = mode_ua.map(mode_ua.value_counts())
    return f


def dwell(ev, dt_in, new_sess):
    """Время на странице и форма распределения интервалов (группа w_).
    Человек задерживается на объявлении (читает, листает фото), сборщик уходит сразу;
    поэтому время до следующего события считаем отдельно по типу текущего события."""
    g = ev['cookie_id']
    f = pd.DataFrame(index=g.unique())
    nxt_dt = dt_in.groupby(g).shift(-1)    # интервал до следующего события той же сессии
    for name, short in [('search_results_view', 'search'), ('item_view', 'item'),
                        ('photo_swipe', 'photo'), ('seller_page_view', 'seller')]:
        d = nxt_dt.where(ev['event_name'].eq(name))
        gd = d.groupby(g)
        f[f'w_dwell_{short}_med'] = gd.median()
        f[f'w_dwell_{short}_min'] = gd.min()
        f[f'w_dwell_{short}_le5'] = (d <= 5).groupby(g).sum() / gd.count().replace(0, np.nan)
    # гистограмма интервалов по лог-корзинам: форма распределения, а не только квантили
    bins = [-1, 1, 2, 4, 8, 16, 32, 64, 128, 256, SESSION_GAP_S]
    b = pd.cut(dt_in, bins, labels=[f'w_hist_{int(r)}' for r in bins[1:]])
    h = pd.crosstab(g[b.notna()], b[b.notna()])
    f = f.join(h.div(h.sum(axis=1), axis=0))
    f['w_hist_entropy'] = _entropy(h)
    f['w_dt_q90_q10_ratio'] = dt_in.groupby(g).quantile(0.9) / dt_in.groupby(g).quantile(0.1).clip(lower=1)
    return f


def hours(ev):
    g = ev['cookie_id']
    h = ev['event_ts'].dt.hour
    hc = pd.crosstab(g, h)
    f = pd.DataFrame(index=hc.index)
    for name, (lo, hi) in {'night': (0, 6), 'morning': (6, 12), 'day': (12, 18), 'evening': (18, 24)}.items():
        f[f'h_share_{name}'] = _share((h >= lo) & (h < hi), g)
    f['h_n_hours'] = (hc > 0).sum(axis=1)
    f['h_entropy'] = _entropy(hc)
    f['h_first_hour'] = h.groupby(g).min()
    return f


def age(ev, meta, aux):
    m = meta.set_index('cookie_id')
    f = pd.DataFrame(index=m.index)
    f['a_age_days'] = (m['window_start_ts'] - m['cookie_created_at']).dt.total_seconds() / 86400
    f['a_created_in_window'] = (m['cookie_created_at'] >= m['window_start_ts']).astype(int)
    first = ev.groupby('cookie_id')['event_ts'].min()
    f['a_hours_created_to_first'] = (first - m['cookie_created_at']).dt.total_seconds() / 3600
    # событий до окна в данных нет (проверено: n_prior = 0), поэтому предысторию как признак не берём
    f['a_share_before_created'] = _share(ev['event_ts'] < ev['cookie_created_at'], ev['cookie_id'])
    f['a_dup_share'] = aux['n_dup'].reindex(f.index).fillna(0) / (ev['cookie_id'].value_counts() + aux['n_dup'].reindex(f.index).fillna(0))
    return f


def cross_cookie(ev, meta, leaky=False):
    """Трансдуктивные признаки по всем кукам train+test без таргета.
    leaky=True — НЕКОРРЕКТНАЯ версия (популярность по всем дням, включая будущее);
    оставлена только для эксперимента, показывающего размер утечки."""
    f = pd.DataFrame(index=meta['cookie_id'])
    # Популярность просмотренных объявлений: сборщики ходят по «хвосту» выдачи, куда люди не доходят.
    # Популярность объявления для куки с окном [S, W) = число кук, впервые открывших его в [W - 1 сутки, W),
    # включая саму куку. Две гарантии:
    #   * без будущего: учитываются только события раньше конца окна W;
    #   * стационарность: у всех кук одинаковая глубина истории (одни сутки). Накопительный счётчик
    #     «за всё время до W» рос бы от начала train к test и давал сдвиг распределения train/test.
    it = ev.loc[ev['item_id'].notna(), ['cookie_id', 'item_id', 'window_end_ts']]
    first = ev[ev['item_id'].notna()].groupby(['item_id', 'cookie_id'])['event_ts'].min().reset_index()
    pairs = it.drop_duplicates(['cookie_id', 'item_id'])
    pop = pd.Series(np.nan, index=pairs.index)
    for w, idx in pairs.groupby('window_end_ts').groups.items():
        in_lookback = (first['event_ts'] < w) & (first['event_ts'] >= w - POP_LOOKBACK)
        known = first.loc[leaky | in_lookback].groupby('item_id').size()
        pop.loc[idx] = pairs.loc[idx, 'item_id'].map(known).to_numpy()
    lp = np.log1p(pop)
    gc = pairs['cookie_id']
    f['x_item_pop_mean'] = lp.groupby(gc).mean()
    f['x_item_pop_median'] = lp.groupby(gc).median()
    f['x_item_pop_min'] = lp.groupby(gc).min()
    f['x_share_unique_items'] = (pop == 1).groupby(gc).mean()
    return f


# ----------------------------------------------------------------------------- сборка
def build_features(events, train, test):
    meta = pd.concat([train.drop(columns='target', errors='ignore'), test], ignore_index=True)
    ev, aux = clean_events(events, meta)
    vt, dt_in, new_sess = volume_timing(ev)
    parts = [vt, dwell(ev, dt_in, new_sess), event_mix(ev, new_sess), content(ev), search(ev), pointer(ev),
             ua_platform(ev), hours(ev), age(ev, meta, aux), cross_cookie(ev, meta),
             # протекающая версия (группа 'leak') — только для демонстрационного эксперимента в train.py
             cross_cookie(ev, meta, leaky=True).add_prefix('leak_')]
    X = pd.DataFrame(index=pd.Index(meta['cookie_id'], name='cookie_id'))
    for p in parts:
        X = X.join(p)
    # куки без событий в окне: счётчики = 0, статистики остаются NaN (LightGBM умеет NaN)
    cnt_cols = [c for c in X if c in ('v_n_events', 'v_n_sessions', 'p_n', 't_n_intervals', 's_n_search',
                                      'e_n_captcha', 'e_n_contacts')]
    X[cnt_cols] = X[cnt_cols].fillna(0)
    stats = {'events_raw': len(events), 'events_in_windows': len(ev),
             'dups_removed': int(aux['n_dup'].sum()), 'prior_events': int(aux['n_prior'].sum()),
             'text_values_merged': aux.attrs['text_values_merged'],
             'item_pop_lookback': str(POP_LOOKBACK)}   # метка версии: популярность за сутки, а не накопительная
    return X.loc[train['cookie_id']], X.loc[test['cookie_id']], stats


def feature_group(col):
    return col.split('_', 1)[0]