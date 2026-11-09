"""Supplementary analyses computed only from committed aggregate outputs; no raw data or Kaggle token needed."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
SUP = OUT / 'supplementary'
FIG, TABLE = SUP / 'figures', SUP / 'tables'
for folder in [FIG, TABLE]:
    folder.mkdir(parents=True, exist_ok=True)
SEED = 42
TEAL, ORANGE, GREY, SLATE = '#167d9a', '#e18a3b', '#9eabb6', '#4b5d6b'
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams.update({'figure.dpi': 115, 'savefig.dpi': 180, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold', 'font.size': 11})

berlin = json.loads((OUT / 'results.json').read_text())
expanded = json.loads((OUT / 'expanded' / 'results.json').read_text())


def save_figure(fig, name):
    fig.savefig(FIG / f'{name}.png', bbox_inches='tight', facecolor='white')
    plt.close(fig)


def save_table(frame, name):
    frame.to_csv(TABLE / f'{name}.csv', index=False)
    return frame


def coefficient(fit, term, label, scale=1.0):
    lo, hi = fit.conf_int().loc[term]
    return {'question': label, 'estimate': scale * fit.params[term], 'ci_low': scale * lo,
            'ci_high': scale * hi, 'p_value': float(fit.pvalues[term]), 'n': int(fit.nobs)}


def hac(formula, data):
    # Lag-1 HAC errors match the project's Berlin sensitivity specification for annual series.
    return smf.ols(formula, data).fit(cov_type='HAC', cov_kwds={'maxlags': 1}, use_t=True)


# 1 · Are the demographic gaps changing over time? ---------------------------------------------
coverage = pd.read_csv(OUT / 'tables' / 'source_coverage.csv')[['YEAR', 'female_share_pct']]
age_means = pd.read_csv(OUT / 'tables' / 'annual_age_means.csv')
trend = (pd.read_csv(OUT / 'tables' / 'annual_sex_contrast.csv')
         .merge(coverage, on='YEAR').merge(age_means[['YEAR', '30–34', '60–64']], on='YEAR'))
trend['sex_gap'] = trend.female_minus_male
trend['sex_gap_pct'] = 100 * trend.sex_gap / trend.Male
trend['age_gap'] = trend['60–64'] - trend['30–34']
trend['age_gap_pct'] = 100 * trend.age_gap / trend['30–34']
trend['year_c'] = trend.YEAR - 2009

trend_rows, trend_fits = [], {}
for outcome, label in [('sex_gap', 'Female − male gap (min/year)'),
                       ('sex_gap_pct', 'Female − male gap (% of male mean per year)'),
                       ('age_gap', 'Age 60–64 − 30–34 gap (min/year)'),
                       ('age_gap_pct', 'Age 60–64 − 30–34 gap (% of 30–34 mean per year)')]:
    fit = hac(f'{outcome} ~ year_c', trend)
    trend_fits[outcome] = fit
    row = {'outcome': outcome, **coefficient(fit, 'year_c', label)}
    ordinary = smf.ols(f'{outcome} ~ year_c', trend).fit()
    row.update({'ols_ci_low': ordinary.conf_int().loc['year_c', 0], 'ols_ci_high': ordinary.conf_int().loc['year_c', 1],
                'value_1999': trend[outcome].iloc[0], 'value_2018': trend[outcome].iloc[-1],
                'fitted_change_1999_2018': 19 * fit.params['year_c']})
    trend_rows.append(row)
trend_tests = save_table(pd.DataFrame(trend_rows), 'gap_trend_tests')

share_year_r = float(trend.female_share_pct.corr(trend.YEAR))
share_gap_r = float(trend.female_share_pct.corr(trend.sex_gap))
joint = hac('sex_gap ~ year_c + female_share_pct', trend)
collinearity = {'share_year_r': share_year_r, 'share_gap_r': share_gap_r,
                'vif': 1 / (1 - share_year_r ** 2),
                'joint_year': coefficient(joint, 'year_c', 'Year, adjusting for female share'),
                'joint_share': coefficient(joint, 'female_share_pct', 'Female share (per pp), adjusting for year')}

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
grid = pd.DataFrame({'year_c': np.linspace(trend.year_c.min(), trend.year_c.max(), 50)})
for ax, outcome, ylabel, title in [
        (axes[0], 'sex_gap', 'Female − male, age-standardized (min)', 'The female–male gap widened'),
        (axes[1], 'age_gap', 'Age 60–64 − 30–34, sex-standardized (min)', 'The older–younger gap widened')]:
    band = trend_fits[outcome].get_prediction(grid).summary_frame(alpha=.05)
    ax.fill_between(grid.year_c + 2009, band.mean_ci_lower, band.mean_ci_upper, color=ORANGE, alpha=.18, linewidth=0)
    ax.plot(grid.year_c + 2009, band['mean'], color=ORANGE, linewidth=2)
    ax.plot(trend.YEAR, trend[outcome], 'o', color=TEAL)
    row = trend_tests.set_index('outcome').loc[outcome]
    ax.text(.03, .95, f"{row.estimate:+.2f} min/year\n95% CI {row.ci_low:+.2f} to {row.ci_high:+.2f} (HAC)",
            transform=ax.transAxes, va='top', fontsize=10)
    ax.set(xlabel='Race year', ylabel=ylabel, title=title, xticks=range(1999, 2019, 4))
points = axes[2].scatter(trend.female_share_pct, trend.sex_gap, c=trend.YEAR, cmap='viridis', s=45)
fig.colorbar(points, ax=axes[2], label='Race year', ticks=range(1999, 2019, 4), format='%d')
axes[2].set(xlabel='Female share of finish records (%)', ylabel='Female − male gap (min)',
            title=f'Gap tracks female participation (r = {share_gap_r:.2f})')
axes[2].text(.03, .95, f'Share vs year: r = {share_year_r:.2f}\nThe two explanations cannot be separated',
             transform=axes[2].transAxes, va='top', fontsize=10)
fig.tight_layout()
save_figure(fig, 'S1_gap_trends')

# 2 · Which factors have the largest associations? -------------------------------------------
primary = {row['hypothesis']: row for row in berlin['primary_tests']}
sex, age, temp = (primary['Female minus male (minutes)'], primary['Age 60–64 minus 30–34 (minutes)'],
                  primary['Temperature (minutes per 1°C)'])
chicago = {row['question']: row for row in expanded['chicago_results']}
race_means = np.array([row['adjusted_mean'] for row in expanded['race_comparison']])
spread = lambda values: np.quantile(values, .9) - np.quantile(values, .1)
rng = np.random.default_rng(SEED)
boot = [spread(rng.choice(race_means, len(race_means))) for _ in range(10_000)]
berlin_range = berlin['temperature_max'] - berlin['temperature_min']

factor_rows = [
    ('Race field: P90 − P10 of 21 adjusted race means', 'Race field', spread(race_means), *np.quantile(boot, [.025, .975]),
     'Fall marathons; bootstrap over races; selection and course are not separated'),
    ('Age: 60–64 vs 30–34 (Berlin)', 'Demographics', age['estimate'], age['ci_low'], age['ci_high'],
     'Sex-standardized annual contrast'),
    ('Sex: female vs male (Berlin)', 'Demographics', sex['estimate'], sex['ci_low'], sex['ci_high'],
     'Age-standardized annual contrast'),
    (f'Temperature: coolest → warmest Berlin year (+{berlin_range:.1f}°C)', 'Race-day weather',
     berlin_range * temp['estimate'], berlin_range * temp['ci_low'], berlin_range * temp['ci_high'],
     'Observed range of daily mean temperature'),
    ('Temperature: +5°C (Berlin, daily mean)', 'Race-day weather', 5 * temp['estimate'], 5 * temp['ci_low'],
     5 * temp['ci_high'], 'Standardized annual mean, linear year trend'),
    ('Temperature: +5°C (Chicago, 9 a.m.)', 'Race-day weather', *[chicago['Chicago: temperature, minutes per 5°C'][k]
     for k in ['estimate', 'ci_low', 'ci_high']], 'Standardized annual mean, 26 race years'),
    ('Humidity: +10 points RH (Chicago)', 'Race-day weather', *[chicago['Chicago: RH, minutes per 10 percentage points'][k]
     for k in ['estimate', 'ci_low', 'ci_high']], 'Adjusted for temperature and wind; inconclusive'),
]
factors = save_table(pd.DataFrame(factor_rows, columns=['factor', 'category', 'minutes', 'ci_low', 'ci_high', 'note']),
                     'factor_comparison')

metrics = pd.DataFrame(berlin['prediction_metrics']).set_index('model')
full = metrics.loc['Age + sex + year + temperature']
r2_drop = save_table(pd.DataFrame([
    {'removed': 'Sex', 'r2_full': full.r2, 'r2_without': metrics.loc['Full without sex', 'r2'],
     'mae_without': metrics.loc['Full without sex', 'mae']},
    {'removed': 'Age class', 'r2_full': full.r2, 'r2_without': metrics.loc['Full without age', 'r2'],
     'mae_without': metrics.loc['Full without age', 'mae']},
    {'removed': 'Temperature', 'r2_full': full.r2, 'r2_without': metrics.loc['Age + sex + year', 'r2'],
     'mae_without': metrics.loc['Age + sex + year', 'mae']},
]).assign(r2_drop=lambda d: d.r2_full - d.r2_without, mae_full=full.mae), 'r2_drop')

fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={'width_ratios': [1.75, 1]})
palette = {'Race field': SLATE, 'Demographics': TEAL, 'Race-day weather': ORANGE}
for i, row in factors.iterrows():
    axes[0].errorbar(row.minutes, i, xerr=[[row.minutes - row.ci_low], [row.ci_high - row.minutes]],
                     fmt='o', capsize=4, color=palette[row.category], markersize=7)
    axes[0].text(row.ci_high + .8, i, f'{row.minutes:.1f}', va='center', fontsize=10)
axes[0].set_yticks(factors.index, factors.factor)
axes[0].invert_yaxis()
axes[0].axvline(0, color='#888888', linestyle='--')
axes[0].set(xlabel='Difference in mean finish time (minutes, 95% CI)',
            title='Typical differences, on one scale', xlim=(-8, 48))
for label, color in palette.items():
    axes[0].plot([], [], 'o', color=color, label=label)
axes[0].legend(frameon=False, loc='lower right', fontsize=9)
axes[1].barh(r2_drop.removed, r2_drop.r2_drop, color=[TEAL, TEAL, ORANGE])
for i, row in r2_drop.iterrows():
    axes[1].text(row.r2_drop + .002, i, f'−{row.r2_drop:.3f}', va='center', fontsize=10)
axes[1].invert_yaxis()
axes[1].set(xlabel=f'Held-out R² lost when removed (full model R² = {full.r2:.3f})',
            title='Individual prediction, Berlin 2014–2018', xlim=(0, .09))
fig.tight_layout()
save_figure(fig, 'S2_factor_comparison')

# 3 · Comparison with published research ------------------------------------------------------
berlin_annual = pd.read_csv(OUT / 'tables' / 'annual_weather_analysis.csv')
chicago_annual = pd.read_csv(OUT / 'expanded' / 'tables' / 'chicago_annual.csv')
chicago_main = chicago_annual.loc[chicago_annual.Year.ne(2007)].copy()
heat_rows = []
for label, est, lo, hi, base in [
        ('Berlin (this project)', 5 * temp['estimate'], 5 * temp['ci_low'], 5 * temp['ci_high'],
         berlin_annual.standardized_mean.mean()),
        ('Chicago (this project)', *[chicago['Chicago: temperature, minutes per 5°C'][k] for k in ['estimate', 'ci_low', 'ci_high']],
         chicago_main.standardized_mean.mean())]:
    heat_rows.append({'source': label, 'measure': 'Slowing per 5°C air temperature', 'group': 'All finishers, annual mean',
                      'value_pct': 100 * est / base, 'ci_low_pct': 100 * lo / base, 'ci_high_pct': 100 * hi / base})
# Ely et al. (2007), Med Sci Sports Exerc 39:487–493, regression slopes per 5°C WBGT.
for place, value in [('25th place', 1.1), ('50th place', 1.5), ('100th place', 1.8), ('300th place', 3.2)]:
    heat_rows.append({'source': 'Ely et al. 2007', 'measure': 'Slowing per 5°C WBGT', 'group': place,
                      'value_pct': value, 'ci_low_pct': np.nan, 'ci_high_pct': np.nan})
pacing = pd.DataFrame(expanded['pacing_summary'])
# Deaner et al. (2015), Med Sci Sports Exerc 47:607–616: 14 U.S. marathons in 2011.
deaner = {'M': 15.6, 'F': 11.7}
pace_rows = [{'source': 'Deaner et al. 2015', 'measure': 'Mean second-half slowdown', 'group': s,
              'value_pct': v, 'ci_low_pct': np.nan, 'ci_high_pct': np.nan} for s, v in deaner.items()]
pace_rows += [{'source': f'Boston {r.year} (this project)', 'measure': 'Mean second-half slowdown', 'group': r.sex,
               'value_pct': r.mean_slow_pct, 'ci_low_pct': np.nan, 'ci_high_pct': np.nan} for r in pacing.itertuples()]
literature = save_table(pd.DataFrame(heat_rows + pace_rows), 'literature_comparison')

fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1.15, 1]})
heat = pd.DataFrame(heat_rows)
labels = []
for i, row in heat.iterrows():
    project = 'project' in row.source
    if project:
        axes[0].errorbar(row.value_pct, i, xerr=[[row.value_pct - row.ci_low_pct], [row.ci_high_pct - row.value_pct]],
                         fmt='o', capsize=4, color=TEAL, markersize=8)
    else:
        axes[0].plot(row.value_pct, i, 's', color=GREY, markersize=8)
    labels.append(row.source if project else f'Ely 2007, {row.group} (WBGT)')
axes[0].set_yticks(heat.index, labels)
axes[0].invert_yaxis()
axes[0].set(xlabel='Slower finish per +5°C (% of mean time)', title='Heat penalty: same order as published estimates',
            xlim=(0, 4.5))
width = .38
for j, sex_code in enumerate(['M', 'F']):
    data = pacing.loc[pacing.sex.eq(sex_code)]
    color = TEAL if sex_code == 'M' else ORANGE
    axes[1].bar(np.arange(3) + (j - .5) * width, data.mean_slow_pct, width, color=color,
                label=f"{'Men' if sex_code == 'M' else 'Women'}, Boston")
    axes[1].axhline(deaner[sex_code], color=color, linestyle='--', linewidth=1.5)
    axes[1].text(2.5, deaner[sex_code] + .25, f"Deaner 2015, {'men' if sex_code == 'M' else 'women'} {deaner[sex_code]}%",
                 va='bottom', fontsize=9, color=color)
axes[1].set(xticks=range(3), xticklabels=pacing.year.unique(), xlabel='Race year', xlim=(-.6, 3.6),
            ylabel='Mean second-half slowdown (%)', title='Men slow more, as in 14 U.S. marathons')
axes[1].legend(frameon=False, fontsize=9, loc='upper left')
fig.tight_layout()
save_figure(fig, 'S3_literature_comparison')

# 4 · Weather discussion: Boston conditions, an optimal temperature, and Riegel's formula --------
# Start-line conditions from findmymarathon.com and contemporary news reports; Boston runs west to east.
boston_weather = pd.DataFrame([
    {'year': 2015, 'start_temp_f': 41, 'start_rh_pct': 93, 'wind': 'E 12 mph (headwind)', 'conditions': 'rain'},
    {'year': 2016, 'start_temp_f': 67, 'start_rh_pct': 28, 'wind': 'ENE 8 mph (headwind)', 'conditions': 'sunny, warm'},
    {'year': 2017, 'start_temp_f': 72, 'start_rh_pct': 37, 'wind': 'WNW 17 mph (tailwind)', 'conditions': 'sunny, warm'},
])
boston_weather['start_temp_c'] = (boston_weather.start_temp_f - 32) * 5 / 9
riegel_b = 1.06
boston = pacing.merge(boston_weather, on='year')
boston['implied_exponent'] = np.log2(2 + boston.median_slow_pct / 100)
boston['riegel_underestimate_pct'] = 100 * ((2 + boston.median_slow_pct / 100) / 2 ** riegel_b - 1)
boston['riegel_underestimate_min_4h'] = 240 * (1 - 2 ** riegel_b / (2 + boston.median_slow_pct / 100))
save_table(boston, 'boston_weather_pacing')

chicago_main['temp_dev'] = chicago_main.temp_c - chicago_main.temp_c.mean()
linear = smf.ols('standardized_mean ~ temp_dev + year_c', chicago_main).fit(cov_type='HC3', use_t=True)
quadratic = smf.ols('standardized_mean ~ temp_dev + I(temp_dev**2) + year_c', chicago_main).fit(cov_type='HC3', use_t=True)
b1, b2 = quadratic.params['temp_dev'], quadratic.params['I(temp_dev ** 2)']
optimum = chicago_main.temp_c.mean() - b1 / (2 * b2) if b2 > 0 else np.nan
with_2007 = chicago_annual.assign(temp_dev=lambda d: d.temp_c - chicago_main.temp_c.mean())
quadratic_2007 = smf.ols('standardized_mean ~ temp_dev + I(temp_dev**2) + year_c', with_2007).fit(cov_type='HC3', use_t=True)
chicago_quadratic = save_table(pd.DataFrame([
    {**coefficient(quadratic, 'I(temp_dev ** 2)', 'Temperature² (main, exclude 2007)'),
     'optimum_c': optimum, 'aic_linear': linear.aic, 'aic_quadratic': quadratic.aic},
    {**coefficient(quadratic_2007, 'I(temp_dev ** 2)', 'Temperature² (include interrupted 2007)'),
     'optimum_c': chicago_main.temp_c.mean() - quadratic_2007.params['temp_dev'] / (2 * quadratic_2007.params['I(temp_dev ** 2)']),
     'aic_linear': np.nan, 'aic_quadratic': quadratic_2007.aic},
]), 'chicago_quadratic')

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
for sex_code, color, name in [('M', TEAL, 'Men'), ('F', ORANGE, 'Women')]:
    data = boston.loc[boston.sex.eq(sex_code)]
    axes[0].plot(data.start_temp_c, data.median_slow_pct, 'o-', color=color, label=name, markersize=8)
for row in boston.loc[boston.sex.eq('M')].itertuples():
    note = f"{row.year}: RH {row.start_rh_pct}%\n{row.wind.split('(')[1].rstrip(')')}" + (', rain' if row.conditions == 'rain' else '')
    offset = {2015: (6, 18), 2016: (-115, 4), 2017: (-30, 16)}[row.year]
    axes[0].annotate(note, (row.start_temp_c, row.median_slow_pct), textcoords='offset points',
                     xytext=offset, fontsize=9)
axes[0].set(xlabel='Start-line temperature (°C)', ylabel='Median second-half slowdown (%)',
            title='Boston: the humid, cold year slowed least', ylim=(0, 18), xlim=(2, 25))
axes[0].legend(frameon=False, loc='lower right')

detrended = chicago_main.standardized_mean - quadratic.params['year_c'] * chicago_main.year_c
grid = np.linspace(chicago_main.temp_c.min(), chicago_main.temp_c.max(), 100)
dev = grid - chicago_main.temp_c.mean()
axes[1].axvspan(3.8, 9.9, color=GREY, alpha=.25, label='Helou 2012 optimum range')
axes[1].scatter(chicago_main.temp_c, detrended, color=TEAL, s=40, zorder=3)
axes[1].plot(grid, quadratic.params['Intercept'] + b1 * dev + b2 * dev ** 2, color=ORANGE, linewidth=2, label='Quadratic fit')
axes[1].plot(grid, linear.params['Intercept'] + linear.params['temp_dev'] * dev,
             color=SLATE, linestyle='--', label='Linear fit')
q_row = chicago_quadratic.iloc[0]
axes[1].text(.03, .95, f'Minimum ≈ {q_row.optimum_c:.1f}°C\nCurvature p = {q_row.p_value:.2f} (HC3, n = {q_row.n})',
             transform=axes[1].transAxes, va='top', fontsize=10)
axes[1].set(xlabel='Chicago 9 a.m. temperature (°C)', ylabel='Standardized mean, trend removed (min)',
            title='A U-shape is suggested, not established')
axes[1].legend(frameon=False, fontsize=9, loc='lower right')

for j, (sex_code, color, name) in enumerate([('M', TEAL, 'Men'), ('F', ORANGE, 'Women')]):
    data = boston.loc[boston.sex.eq(sex_code)]
    axes[2].bar(np.arange(3) + (j - .5) * width, data.implied_exponent, width, color=color, label=name)
axes[2].axhline(riegel_b, color='#333333', linestyle='--')
axes[2].text(-.55, riegel_b + .002, 'Riegel default 1.06\n(≈ 8.5% slowdown)', fontsize=9)
axes[2].set(xticks=range(3), xticklabels=boston.year.unique(), ylim=(1.0, 1.12), xlabel='Race year',
            ylabel='Implied half → full exponent', title="Riegel's formula is optimistic in warm years")
axes[2].legend(frameon=False, fontsize=9, loc='upper left')
fig.tight_layout()
save_figure(fig, 'S4_weather_discussion')

summary = {
    'inputs': ['outputs/results.json', 'outputs/expanded/results.json', 'outputs/tables/*.csv',
               'outputs/expanded/tables/chicago_annual.csv'],
    'gap_trends': trend_tests.to_dict(orient='records'),
    'collinearity': collinearity,
    'factor_comparison': factors.to_dict(orient='records'),
    'r2_drop': r2_drop.to_dict(orient='records'),
    'literature_comparison': literature.to_dict(orient='records'),
    'boston_weather_pacing': boston.to_dict(orient='records'),
    'chicago_quadratic': chicago_quadratic.to_dict(orient='records'),
}
(SUP / 'results.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')
print(trend_tests[['question', 'estimate', 'ci_low', 'ci_high', 'p_value']].to_string(index=False))
print(f"Share–year r = {share_year_r:.2f}; VIF = {collinearity['vif']:.1f}")
print(factors[['factor', 'minutes', 'ci_low', 'ci_high']].to_string(index=False))
print(chicago_quadratic.to_string(index=False))
print(f'Saved {len(list(FIG.glob("*.png")))} figures and {len(list(TABLE.glob("*.csv")))} tables in {SUP.relative_to(ROOT)}')
