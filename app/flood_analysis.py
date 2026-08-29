import json
import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import ripser
import persim

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
MON_LABELS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

ELNINO_YEARS = {1982,1983,1987,1991,1992,1994,1997,1998,
                2002,2004,2006,2009,2010,2015,2016,2018,2019,2023}
PIOD_YEARS   = {1994,1997,2006,2012,2015,2019,2023}
NIOD_YEARS   = {1996,1998,2010,2016}
BOB_ACTIVE   = {2007,2008,2009,2013,2019,2020,2021,2023}

RAINFALL_CSV = os.path.join(DATA_DIR, 'district_wise_rainfall_normal.csv')
EMDAT_CSV = os.path.join(DATA_DIR, 'public_emdat_project.csv')
OUTPUT_JSON = os.path.join(DATA_DIR, 'flood_analysis_results.json')


def to_native(obj):
    """Recursively convert numpy types to plain Python so json.dumps works."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def main():
    results = {}

    # ── LOAD & CLEAN ─────────────────────────────────────────────────
    rain = pd.read_csv(RAINFALL_CSV)
    rain.columns = rain.columns.str.strip()
    rain['STATE_UT_NAME'] = rain['STATE_UT_NAME'].str.strip().str.title()
    rain['DISTRICT'] = rain['DISTRICT'].str.strip().str.title()
    for col in MONTHS + ['ANNUAL']:
        rain[col] = pd.to_numeric(rain[col], errors='coerce')

    emdat = pd.read_csv(EMDAT_CSV, encoding='latin-1', low_memory=False)
    emdat.columns = emdat.columns.str.strip()
    for col in ['Total Deaths', 'Total Affected', "Total Damage, Adjusted ('000 US$)"]:
        if col in emdat.columns:
            emdat[col] = pd.to_numeric(emdat[col], errors='coerce')
    emdat['Year'] = pd.to_numeric(emdat['Start Year'], errors='coerce')
    emdat['Month'] = pd.to_numeric(emdat['Start Month'], errors='coerce')
    emdat['Damage_M'] = emdat.get("Total Damage, Adjusted ('000 US$)", pd.Series(dtype=float)) / 1000

    india_floods = emdat[(emdat['Country'] == 'India') & (emdat['Disaster Type'] == 'Flood')].copy()
    # Steps 6-7: ALL countries with flood records, not a fixed list
    global_floods = emdat[emdat['Disaster Type'] == 'Flood'].copy()

    print(f"India flood records: {len(india_floods)}")
    print(f"Global flood records: {len(global_floods)} across {global_floods['Country'].nunique()} countries")

    # ══════════════════════════════════════════════════════════════
    # STEP 1 — MONTH-FLOOD CORRELATION (INDIA)
    # ══════════════════════════════════════════════════════════════
    print("Step 1: month-flood correlation...")
    monthly_events = india_floods.groupby('Month').size().reindex(range(1, 13), fill_value=0)
    monthly_deaths = india_floods.groupby('Month')['Total Deaths'].sum().reindex(range(1, 13), fill_value=0)
    monthly_damage = india_floods.groupby('Month')['Damage_M'].sum().reindex(range(1, 13), fill_value=0)
    national_avg = rain[MONTHS].mean().values

    r_events, p_events = pearsonr(national_avg, monthly_events.values)
    r_deaths, p_deaths = pearsonr(national_avg, monthly_deaths.values)
    r_damage, p_damage = pearsonr(national_avg, monthly_damage.values)

    norm_rain = (national_avg - national_avg.mean()) / national_avg.std()
    norm_events = (monthly_events.values - monthly_events.values.mean()) / (monthly_events.values.std() + 1e-9)
    month_importance = norm_rain * norm_events

    top_months = [MON_LABELS[i] for i in np.argsort(month_importance)[::-1][:4]]

    results['step1_month_correlation'] = {
        'months': MON_LABELS,
        'flood_events_by_month': monthly_events.values,
        'avg_rainfall_by_month': national_avg,
        'importance_score_by_month': month_importance,
        'correlations': {
            'rainfall_vs_events': {'r': r_events, 'p': p_events},
            'rainfall_vs_deaths': {'r': r_deaths, 'p': p_deaths},
            'rainfall_vs_damage': {'r': r_damage, 'p': p_damage},
        },
        'top_flood_months': top_months,
    }

    # ══════════════════════════════════════════════════════════════
    # STEP 2 — TDA ON DISTRICT RAINFALL POINT CLOUD (INDIA)
    # ══════════════════════════════════════════════════════════════
    print("Step 2: TDA on district rainfall space...")
    X_rain = rain[MONTHS].dropna().values
    X_scaled = StandardScaler().fit_transform(X_rain)

    rng = np.random.default_rng(42)
    if len(X_scaled) > 1500:
        idx = rng.choice(len(X_scaled), 1500, replace=False)
        X_tda = X_scaled[idx]
    else:
        X_tda = X_scaled

    diagrams = ripser.ripser(X_tda, maxdim=1)['dgms']
    h0 = diagrams[0]
    h0_finite = h0[h0[:, 1] < np.inf]
    lifetimes_h0 = h0_finite[:, 1] - h0_finite[:, 0]

    h1 = diagrams[1]
    h1_finite = h1[h1[:, 1] < np.inf] if len(h1) > 0 and np.any(h1[:, 1] < np.inf) else h1

    lifetime_threshold = np.percentile(lifetimes_h0, 75)
    significant_clusters = int(np.sum(lifetimes_h0 > lifetime_threshold))

    results['step2_district_topology'] = {
        'h0_points': [{'birth': b, 'death': d} for b, d in zip(h0_finite[:, 0], h0_finite[:, 1])],
        'h1_points': [{'birth': b, 'death': d} for b, d in zip(h1_finite[:, 0], h1_finite[:, 1])] if len(h1_finite) else [],
        'distinct_monsoon_regimes': significant_clusters,
        'longest_h0_lifetime': float(lifetimes_h0.max()),
        'h1_loop_count': int(len(h1_finite)),
    }

    # ══════════════════════════════════════════════════════════════
    # STEP 3 — MONSOON DRIVER FINGERPRINTING (INDIA)
    # ══════════════════════════════════════════════════════════════
    print("Step 3: driver fingerprinting...")
    annual = india_floods.groupby('Year').agg(
        Events=('DisNo.', 'count'),
        Deaths=('Total Deaths', 'sum'),
        Affected=('Total Affected', 'sum'),
        Damage_M=('Damage_M', 'sum'),
        Peak_Month=('Month', lambda x: x.mode()[0] if len(x) > 0 else np.nan)
    ).reset_index()

    annual['ElNino'] = annual['Year'].isin(ELNINO_YEARS).astype(int)
    annual['pIOD'] = annual['Year'].isin(PIOD_YEARS).astype(int)
    annual['nIOD'] = annual['Year'].isin(NIOD_YEARS).astype(int)
    annual['BoB'] = annual['Year'].isin(BOB_ACTIVE).astype(int)
    annual['JunSep_Sig'] = ((annual['Peak_Month'] >= 6) & (annual['Peak_Month'] <= 9)).astype(int)
    annual['OctDec_Sig'] = ((annual['Peak_Month'] >= 10) & (annual['Peak_Month'] <= 12)).astype(int)

    feat_cols = ['ElNino', 'pIOD', 'nIOD', 'BoB', 'Events', 'Deaths', 'Damage_M', 'JunSep_Sig', 'OctDec_Sig']
    annual_feat = annual[feat_cols].fillna(0)
    annual_scaled = StandardScaler().fit_transform(annual_feat)

    driver_impact = {}
    for driver in ['ElNino', 'pIOD', 'nIOD', 'BoB']:
        with_driver = annual[annual[driver] == 1]['Damage_M'].mean()
        without_driver = annual[annual[driver] == 0]['Damage_M'].mean()
        with_driver = 0.0 if pd.isna(with_driver) else with_driver
        without_driver = 0.0 if pd.isna(without_driver) else without_driver
        ratio = with_driver / (without_driver + 1e-6)
        driver_impact[driver] = {'with': with_driver, 'without': without_driver, 'ratio': ratio}

    pca = PCA(n_components=2)
    coords = pca.fit_transform(annual_scaled)
    annual['PC1'] = coords[:, 0]
    annual['PC2'] = coords[:, 1]

    def driver_label(row):
        tags = []
        if row['ElNino']: tags.append('ElN')
        if row['pIOD']: tags.append('pIOD')
        if row['BoB']: tags.append('BoB')
        if row['nIOD']: tags.append('nIOD')
        return '+'.join(tags) if tags else 'None'

    annual['Driver_Label'] = annual.apply(driver_label, axis=1)

    year_points = [
        {
            'year': int(row['Year']),
            'pc1': row['PC1'],
            'pc2': row['PC2'],
            'damage_m': row['Damage_M'],
            'driver_label': row['Driver_Label'],
        }
        for _, row in annual.iterrows()
    ]

    best_driver = max(driver_impact, key=lambda d: driver_impact[d]['ratio'])

    results['step3_driver_fingerprint'] = {
        'driver_impact': driver_impact,
        'best_driver': best_driver,
        'pca_explained_variance': [pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]],
        'year_points': year_points,
    }

    # ══════════════════════════════════════════════════════════════
    # STEP 4 — TDA ON YEAR-SPACE (INDIA)
    # ══════════════════════════════════════════════════════════════
    print("Step 4: TDA on year-space...")
    diagrams_yr = ripser.ripser(annual_scaled, maxdim=1)['dgms']
    h0_yr = diagrams_yr[0]
    h0_yr_finite = h0_yr[h0_yr[:, 1] < np.inf]
    lt_yr = h0_yr_finite[:, 1] - h0_yr_finite[:, 0]

    h1_yr = diagrams_yr[1]
    h1_yr_finite = h1_yr[h1_yr[:, 1] < np.inf] if len(h1_yr) > 0 and np.any(h1_yr[:, 1] < np.inf) else h1_yr

    n_clusters = int(np.sum(lt_yr > np.percentile(lt_yr, 70))) if len(lt_yr) else 0

    results['step4_year_topology'] = {
        'h0_points': [{'birth': b, 'death': d} for b, d in zip(h0_yr_finite[:, 0], h0_yr_finite[:, 1])],
        'h1_points': [{'birth': b, 'death': d} for b, d in zip(h1_yr_finite[:, 0], h1_yr_finite[:, 1])] if len(h1_yr_finite) else [],
        'meaningful_year_clusters': n_clusters,
        'h1_loop_count': int(len(h1_yr_finite)) if len(h1_yr) else 0,
    }

    # ══════════════════════════════════════════════════════════════
    # STEP 5 — ALL-MONTH DRIVER HEATMAP (INDIA)
    # ══════════════════════════════════════════════════════════════
    print("Step 5: month x driver heatmap...")
    nat_by_month = rain[MONTHS].mean()
    emdat_by_month = india_floods.groupby('Month').agg(
        Events=('DisNo.', 'count'),
        Deaths=('Total Deaths', 'sum'),
        Damage=('Damage_M', 'sum')
    ).reindex(range(1, 13), fill_value=0)

    shift_corr = {}
    ev_vec = emdat_by_month['Events'].values
    rain_vec = nat_by_month.values
    for lag, shift in [('lag0', 0), ('lag1', -1), ('lag2', -2)]:
        shift_corr[lag] = pearsonr(rain_vec, np.roll(ev_vec, shift))[0]

    shift_series = {
        'lag0': [pearsonr(rain_vec, ev_vec)[0]] * 1,
    }
    # Build the per-month lag lines the frontend needs (same value repeated
    # per month row in the original script's DataFrame construction, since
    # the source script computes one scalar per lag across all 12 months).
    lag0_val = pearsonr(rain_vec, ev_vec)[0]
    lag1_val = pearsonr(rain_vec, np.roll(ev_vec, -1))[0]
    lag2_val = pearsonr(rain_vec, np.roll(ev_vec, -2))[0]

    year_driver_df = annual[['Year', 'ElNino', 'pIOD', 'nIOD', 'BoB', 'Peak_Month', 'Events', 'Damage_M']].dropna()
    month_driver_ct = pd.DataFrame(0.0, index=range(1, 13), columns=['ElNino', 'pIOD', 'nIOD', 'BoB'])
    for _, row in year_driver_df.iterrows():
        m = int(row['Peak_Month'])
        for d in ['ElNino', 'pIOD', 'nIOD', 'BoB']:
            month_driver_ct.loc[m, d] += row[d] * row['Damage_M']
    month_driver_norm = month_driver_ct.div(month_driver_ct.sum(axis=0).replace(0, 1))

    results['step5_month_driver_heatmap'] = {
        'months': MON_LABELS,
        'driver_damage_share': {
            driver: month_driver_norm[driver].values for driver in ['ElNino', 'pIOD', 'nIOD', 'BoB']
        },
        'lag_correlation': {
            'lag0': lag0_val,
            'lag1': lag1_val,
            'lag2': lag2_val,
        },
    }

    # ══════════════════════════════════════════════════════════════
    # STEP 6 — GLOBAL COUNTRY-LEVEL TDA FINGERPRINTS (ALL COUNTRIES)
    # ══════════════════════════════════════════════════════════════
    print("Step 6: global country topology...")
    country_annual = global_floods.groupby(['Country', 'Year']).agg(
        Events=('DisNo.', 'count'),
        Deaths=('Total Deaths', 'sum'),
        Damage=('Damage_M', 'sum')
    ).reset_index()

    country_annual['ElNino'] = country_annual['Year'].isin(ELNINO_YEARS).astype(int)
    country_annual['pIOD'] = country_annual['Year'].isin(PIOD_YEARS).astype(int)
    country_annual['BoB'] = country_annual['Year'].isin(BOB_ACTIVE).astype(int)

    country_vectors = []
    country_names = []
    for country, grp in country_annual.groupby('Country'):
        if len(grp) < 3:
            continue
        mean_dmg = grp['Damage'].mean()
        vec = [
            grp['Events'].mean(),
            grp['Events'].std() if len(grp) > 1 else 0.0,
            mean_dmg,
            grp['Damage'].std() if len(grp) > 1 else 0.0,
            (grp[grp['ElNino'] == 1]['Damage'].mean() if (grp['ElNino'] == 1).any() else 0.0) / (mean_dmg + 1e-6),
            (grp[grp['pIOD'] == 1]['Damage'].mean() if (grp['pIOD'] == 1).any() else 0.0) / (mean_dmg + 1e-6),
            (grp[grp['BoB'] == 1]['Damage'].mean() if (grp['BoB'] == 1).any() else 0.0) / (mean_dmg + 1e-6),
            grp['Deaths'].mean(),
        ]
        vec = [0.0 if pd.isna(v) else v for v in vec]
        country_vectors.append(vec)
        country_names.append(country)

    C_mat = np.array(country_vectors)
    C_scaled = StandardScaler().fit_transform(C_mat)

    diagrams_c = ripser.ripser(C_scaled, maxdim=1)['dgms']
    h0_c = diagrams_c[0]
    h0_c_finite = h0_c[h0_c[:, 1] < np.inf]
    lt_c = h0_c_finite[:, 1] - h0_c_finite[:, 0]
    n_country_clusters = int(np.sum(lt_c > np.percentile(lt_c, 60))) if len(lt_c) else 0

    pca2 = PCA(n_components=2)
    country_coords = pca2.fit_transform(C_scaled)

    df_c = pd.DataFrame(C_mat, columns=['mean_ev', 'std_ev', 'mean_dmg', 'std_dmg',
                                         'elnino_ratio', 'piod_ratio', 'bob_ratio', 'mean_deaths'])
    df_c['Country'] = country_names
    df_c['PC1'] = country_coords[:, 0]
    df_c['PC2'] = country_coords[:, 1]

    def dominant_driver(row):
        d = {'El Nino': row['elnino_ratio'], 'pIOD': row['piod_ratio'], 'BoB': row['bob_ratio']}
        return max(d, key=d.get)

    df_c['Dom_Driver'] = df_c.apply(dominant_driver, axis=1)

    country_points = [
        {
            'country': row['Country'],
            'pc1': row['PC1'],
            'pc2': row['PC2'],
            'mean_damage_m': row['mean_dmg'],
            'mean_deaths': row['mean_deaths'],
            'dominant_driver': row['Dom_Driver'],
            'elnino_ratio': row['elnino_ratio'],
            'piod_ratio': row['piod_ratio'],
            'bob_ratio': row['bob_ratio'],
        }
        for _, row in df_c.iterrows()
    ]

    results['step6_country_topology'] = {
        'countries_analyzed': len(country_names),
        'distinct_clusters': n_country_clusters,
        'pca_explained_variance': [pca2.explained_variance_ratio_[0], pca2.explained_variance_ratio_[1]],
        'country_points': country_points,
    }

    # ══════════════════════════════════════════════════════════════
    # STEP 7 — WASSERSTEIN DISTANCE BETWEEN ALL COUNTRIES
    # ══════════════════════════════════════════════════════════════
    print("Step 7: pairwise Wasserstein distances (this is the slow part)...")
    country_diagrams = {}
    for country, grp in country_annual.groupby('Country'):
        if len(grp) < 4:
            continue
        feat = grp[['Events', 'Deaths', 'Damage', 'ElNino', 'pIOD', 'BoB']].fillna(0).values
        if feat.shape[0] < 3:
            continue
        feat_s = StandardScaler().fit_transform(feat)
        dgm = ripser.ripser(feat_s, maxdim=0)['dgms'][0]
        country_diagrams[country] = dgm

    c_names = list(country_diagrams.keys())
    n = len(c_names)
    W_mat = np.zeros((n, n))

    print(f"  Computing {n*(n-1)//2} pairwise distances for {n} countries...")
    for i in range(n):
        for j in range(i + 1, n):
            d = persim.wasserstein(country_diagrams[c_names[i]], country_diagrams[c_names[j]])
            W_mat[i, j] = d
            W_mat[j, i] = d

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((c_names[i], c_names[j], float(W_mat[i, j])))
    pairs_sorted = sorted(pairs, key=lambda x: x[2])

    results['step7_wasserstein'] = {
        'countries': c_names,
        'distance_matrix': W_mat,
        'most_similar_pairs': [
            {'country_a': a, 'country_b': b, 'distance': d} for a, b, d in pairs_sorted[:15]
        ],
    }

    # ══════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════
    results['summary'] = {
        'top_flood_months_india': top_months,
        'driver_ratios': {k: v['ratio'] for k, v in driver_impact.items()},
        'dominant_driver_india': best_driver,
        'distinct_monsoon_regimes': significant_clusters,
        'meaningful_year_clusters_india': n_clusters,
        'countries_analyzed_globally': len(country_names),
        'global_country_clusters': n_country_clusters,
        'most_similar_country_pair': {
            'country_a': pairs_sorted[0][0],
            'country_b': pairs_sorted[0][1],
            'distance': pairs_sorted[0][2],
        } if pairs_sorted else None,
    }

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(to_native(results), f, indent=2)

    print(f"\nDone. Wrote {OUTPUT_JSON}")


if __name__ == '__main__':
    main()
