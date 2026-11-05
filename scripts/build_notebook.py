"""Build the readable notebook source; execute it separately with nbconvert."""
from pathlib import Path
import nbformat as nbf
from textwrap import dedent
ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(dedent(s).strip()))
def code(s): cells.append(nbf.v4.new_code_cell(dedent(s).strip()))

md('''
# Marathon performance: weather, pacing, and the information we are missing
## Six Kaggle datasets · Berlin, Chicago, Boston, and U.S. fall marathons
**Group 9 · Going for the Gold**
Introduction to Statistics · Final project

This notebook investigates marathon performance across six real-world Kaggle datasets.
Chapters 1–8 preserve the initial Berlin study (1999–2018, ages 20–69). Chapters 9–14 extend it
with Chicago weather, multi-race comparisons, Boston pacing, repeat-runner history, and home weather.
Chapter 15 adds actual daily training logs; a seventh, small training dataset is audited but excluded from substantive inference.
The companion `marathon_story.html` integrates the findings; `berlin_story.html` preserves the initial study.

**Read in order or select “Restart Kernel and Run All Cells.”** Raw data are cached locally;
execution does not need a Kaggle token or internet connection. See `README.md` for setup.

1. Questions and study design
2. Source audit and reproducible cleaning
3. Descriptive statistics and distributions
4. Demographic comparisons and uncertainty
5. Weather associations at the race-year level
6. Diagnostics and sensitivity checks
7. Prediction on five unseen race years
8. Berlin conclusions and exported evidence
9. Chicago replication, humidity, and wind
10. Differences across races after demographic adjustment
11. Boston pacing and prediction after 10 km
12. Prior-race history among conservatively matched repeat runners
13. Home weather and second-half slowdown
14. Evidence audit, multiple testing, and integrated conclusions
15. Actual training logs and a marathon-day duration proxy
''')
md('''
## 1 · Questions before calculations

- Do finishing times differ across recorded sex categories and age groups?
- Are warmer race days associated with slower finishes after controlling for demographic composition and calendar time?
- How well do demographics and weather predict an individual finish in a later race?
- Can Berlin answer the humidity hypothesis? **No: its files lack humidity. Chapter 9 estimates humidity from Chicago dew point.**

The source column is named `GENDER`, but contains only `male` and `female`. We call this **recorded sex**
throughout and do not infer gender identity or biology from a category label.

**Primary estimands:** (1) the female-minus-male mean-time difference, standardized to the same age mix;
(2) the 60–64-minus-30–34 age-group difference, standardized to the same sex mix;
(3) the association between mean daily temperature and a demographic-standardized annual mean finish,
adjusting for a linear calendar-year trend. All effects are measured in **minutes**.
We report 95% confidence intervals and Holm-adjust the three primary p-values (family threshold 0.05).
These choices were finalized after an initial data audit and exploratory look, so this is **not a preregistered study**.

**Two units of analysis:** a row is a recorded finish, not necessarily a unique person.
For inference, we use 20 annual comparisons/weather observations, giving each year equal weight.
Hundreds of thousands of finishers do not create hundreds of thousands of independent weather days.
Year-to-year independence is an approximation: recurring runners and time trends can still matter.
All findings are observational, conditional on starting, finishing, and appearing in the source.
''')
code('''
from pathlib import Path
import json
import hashlib
import sys
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from IPython.display import display, Markdown

ROOT = Path.cwd()
if not (ROOT / 'scripts' / 'build_notebook.py').is_file():
    raise RuntimeError('Open/run this notebook from the finalproject folder.')
RAW = ROOT / 'data' / 'raw' / 'berlin'
OUT = ROOT / 'outputs'
FIG = OUT / 'figures'
TABLE = OUT / 'tables'
for folder in [FIG, TABLE]:
    folder.mkdir(parents=True, exist_ok=True)
SEED = 42
sns.set_theme(style='whitegrid', context='notebook', palette=['#167d9a', '#e18a3b'])
plt.rcParams.update({'figure.dpi': 115, 'savefig.dpi': 180, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titleweight': 'bold', 'font.size': 11})
pd.set_option('display.max_columns', 16)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

def save_figure(name):
    plt.savefig(FIG / f'{name}.png', bbox_inches='tight', facecolor='white')
    plt.show()

def save_table(frame, name, index=False):
    frame.to_csv(TABLE / f'{name}.csv', index=index)
    return frame

def p_text(p):
    return '< 0.001' if p < 0.001 else f'= {p:.3f}'
''')
md('''
## 2 · Source audit: the filename is not the analysis population

Source: [Berlin marathon data, Kaggle](https://www.kaggle.com/datasets/aiaiaidavid/berlin-marathons-data),
version 2, listed as CC0. It includes anonymized finish records and a separate weather file.
The uploader credits Beat Knechtle and Volker Scheer. The related
[primary research article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8293098/) describes results from the organizer
and daily weather from the German Weather Service's Berlin Dahlem station.
The Kaggle files themselves do not provide station identifiers, race dates, or runner identifiers;
we preserve that provenance uncertainty rather than treating daily weather as personal race-hour exposure.

The advertised span is 1974–2019, but the result file lacks 1978 and 1980.
Early ages use legacy codes, some historical times have only minute precision, and 2019 contains no women.
We choose **1999–2018**: a contiguous 20-year modern period with both recorded categories each year.
This does not prove every finish was captured. Coverage and timing definitions may still vary.
''')
code('''
manifest = json.loads((ROOT / 'data' / 'raw' / 'manifest.json').read_text())
for filename, expected_hash in manifest['files'].items():
    assert hashlib.sha256((RAW / filename).read_bytes()).hexdigest() == expected_hash
raw = pd.read_csv(RAW / 'Berlin_Marathon_data_1974_2019.csv',
                  dtype={'AGE': 'string', 'COUNTRY': 'string'})
weather = pd.read_csv(RAW / 'Berlin_Marathon_weather_data_since_1974.csv')
assert weather['YEAR'].is_unique, 'Weather must have one row per year.'
print(f'{len(raw):,} result records; {raw.YEAR.nunique()} result years; {len(weather)} weather years.')
display(raw.head())
display(weather.head())
''')
code('''
coverage = pd.crosstab(raw.YEAR, raw.GENDER).reindex(columns=['male', 'female'], fill_value=0)
coverage['total'] = coverage.sum(axis=1)
coverage['female_share_pct'] = 100 * coverage['female'] / coverage['total']
coverage['primary_period'] = coverage.index.to_series().between(1999, 2018)
save_table(coverage.reset_index(), 'source_coverage')
print('Missing result years:', sorted(set(range(1974, 2020)) - set(raw.YEAR)))
print('2019 records by category:', raw.loc[raw.YEAR.eq(2019), 'GENDER'].value_counts().to_dict())
print('Most common age codes:')
display(raw.AGE.value_counts(dropna=False).head(18).to_frame('records'))
display(coverage.tail(22))
''')
md('''
### Cleaning rules and age interpretation

The age field consists of **age classes**, not measured ages. Code 20 represents 20–29;
30 represents 30–34; subsequent codes are five-year groups. This matches the
[Berlin results API data dictionary](https://github.com/hmelberg/berlin-marathon#age-classes).
We analyze the common 20–69 classes, avoiding legacy labels and sparse oldest groups.
A sensitivity analysis later adds the numeric 70–84 classes, pooling ages 65–84 to avoid empty cells.

Parse `HH:MM:SS` into minutes. Missing/unparseable/nonpositive times are excluded.
Keep valid unusually fast or slow finishes: being an outlier is not evidence of an error.
We separately check sensitivity to trimming the outer 1% of times.

**Do not drop identical rows:** these anonymized fields do not identify a runner.
Two people can legitimately share age class, sex, year, country and finish time.
No imputation is made for missing outcomes, unknown age classes, or humidity.
Country is overwhelmingly missing and is not a predictor.
''')
code('''
work = raw.copy()
work['minutes'] = pd.to_timedelta(work.TIME, errors='coerce').dt.total_seconds() / 60
work['age_code'] = pd.to_numeric(work.AGE, errors='coerce')
AGE_LABELS = {20:'20–29', 30:'30–34', 35:'35–39', 40:'40–44', 45:'45–49',
              50:'50–54', 55:'55–59', 60:'60–64', 65:'65–69'}
flow = [{'step': 'All source records', 'remaining': len(work), 'excluded_at_step': 0}]
filtered = work
for label, mask in [
    ('Restrict to 1999–2018', work.YEAR.between(1999, 2018)),
    ('Known age classes 20–69', work.age_code.isin(AGE_LABELS)),
    ('Recorded male/female category', work.GENDER.isin(['male', 'female'])),
    ('Parseable positive finish time', work.minutes.notna() & work.minutes.gt(0))
]:
    before = len(filtered)
    filtered = filtered.loc[mask.loc[filtered.index]].copy()
    flow.append({'step': label, 'remaining': len(filtered), 'excluded_at_step': before-len(filtered)})
df = filtered.merge(weather, on='YEAR', how='left', validate='many_to_one', indicator=True)
assert df['_merge'].eq('both').all()
df = df.drop(columns='_merge')
df['age_group'] = pd.Categorical(df.age_code.map(AGE_LABELS), categories=list(AGE_LABELS.values()), ordered=True)
df['sex'] = df.GENDER.map({'male': 'Male', 'female': 'Female'})
df['group'] = df['age_group'].astype(str) + ' / ' + df.sex
df['year_c'] = df.YEAR - 2009
assert df.YEAR.nunique() == 20 and df.minutes.gt(0).all()
assert df.groupby(['YEAR', 'age_group', 'sex'], observed=True).size().shape[0] == 20 * 9 * 2
flow = save_table(pd.DataFrame(flow), 'cleaning_flow')
display(flow)
print(f'Primary cohort: {len(df):,} recorded finishes in {df.YEAR.nunique()} race years.')
print(f'Identical source rows retained: {raw.duplicated().sum():,}.')
print(f'Country missing in source: {raw.COUNTRY.isna().mean():.1%}.')
''')
md('''
## 3 · What does an ordinary finish look like?

Means summarize the average burden in minutes; medians describe the middle finisher.
The 10th and 90th percentiles show the breadth of performance without assuming normality.
The plots use all eligible rows except the Q–Q diagnostic, which uses a fixed random sample for readability.
''')
code('''
overall = df.minutes.describe(percentiles=[.1, .25, .5, .75, .9]).to_frame('minutes')
sex_summary = df.groupby('sex').minutes.agg(n='size', mean='mean', median='median', sd='std',
                                          p10=lambda s:s.quantile(.1), p90=lambda s:s.quantile(.9))
save_table(overall.reset_index(names='statistic'), 'overall_summary')
save_table(sex_summary.reset_index(), 'sex_summary')
display(overall)
display(sex_summary)
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
sns.histplot(data=df, x='minutes', hue='sex', hue_order=['Male','Female'], bins=np.arange(120, 541, 5),
             stat='density', common_norm=False, element='step', fill=True, alpha=.22, ax=axes[0])
axes[0].set(xlabel='Finish time (minutes)', ylabel='Density within category',
            title='Group averages hide substantial overlap')
sns.boxplot(data=df, x='sex', y='minutes', order=['Male','Female'], hue='sex', legend=False,
            showfliers=False, width=.45, ax=axes[1])
axes[1].set(xlabel='Recorded sex', ylabel='Finish time (minutes)', title='A wide range within each category')
fig.text(.5, -.02, 'Boxplot hides individual outlier markers only; all valid times remain in the analysis.', ha='center', fontsize=10)
fig.tight_layout()
save_figure('01_distributions')
''')
code('''
fig, ax = plt.subplots(figsize=(6.5, 4.5))
stats.probplot(df.minutes.sample(5000, random_state=SEED), dist='norm', plot=ax)
ax.set(title='Finish times are not perfectly normal', xlabel='Theoretical normal quantiles', ylabel='Observed finish time (minutes)')
fig.tight_layout()
save_figure('02_qq_times')
''')
md('''
The Q–Q plot compares the shape of finish times with a normal distribution. Tail departures are expected:
finish times cannot be negative, and slower finishes create a long upper tail.
We do not claim that individual times are normal. Inference below is based on annual differences
or annual means, with its own much smaller effective sample size and diagnostics.
''')
code('''
age_summary = df.groupby(['age_group', 'sex'], observed=True).minutes.agg(n='size', mean='mean', median='median').reset_index()
save_table(age_summary, 'age_summary')
fig, ax = plt.subplots(figsize=(10, 4.8))
sns.lineplot(data=age_summary, x='age_group', y='mean', hue='sex', hue_order=['Male','Female'],
             marker='o', linewidth=2.5, ax=ax)
ax.set(xlabel='Age class (not exact age)', ylabel='Mean finish time (minutes)',
       title='The age pattern is not a straight line')
fig.tight_layout()
save_figure('03_age_pattern')
display(age_summary)
''')
md('''
## 4 · Demographic differences: compare like with like

A raw sex comparison mixes different ages and race years. We first calculate the mean for every
**year × age class × sex** cell. Then:

- For the sex comparison, weight each age class by the same pooled age distribution for both categories.
- For the age comparison, weight each sex by the same pooled sex distribution in every age class.
- Compute each contrast once per year, then estimate the mean of the 20 annual contrasts.

A one-sample t interval/test on annual differences uses 19 degrees of freedom, rather than pretending
all finishes are independent. It assumes annual contrasts are approximately independent and sufficiently
well behaved. These intervals describe variability across comparable race years, not uncertainty about
the arithmetic mean of the already observed records. A Friedman test provides a secondary omnibus check
of age-group ordering with race year as the block.
''')
code('''
cell = df.groupby(['YEAR', 'age_group', 'sex'], observed=True).minutes.agg(mean='mean', n='size').reset_index()
age_weights = df.groupby('age_group', observed=True).size() / len(df)
sex_weights = df.sex.value_counts(normalize=True)
cell['age_weight'] = cell.age_group.map(age_weights).astype(float)
cell['sex_weight'] = cell.sex.map(sex_weights)
sex_annual = (cell.assign(value=cell['mean'] * cell.age_weight)
              .groupby(['YEAR','sex']).value.sum().unstack())
sex_annual['female_minus_male'] = sex_annual.Female - sex_annual.Male
age_annual = (cell.assign(value=cell['mean'] * cell.sex_weight)
              .groupby(['YEAR','age_group'], observed=True).value.sum().unstack())
age_difference = age_annual['60–64'] - age_annual['30–34']

def annual_contrast(values, name):
    values = np.asarray(values, dtype=float)
    estimate = values.mean()
    se = stats.sem(values)
    lo, hi = stats.t.interval(.95, df=len(values)-1, loc=estimate, scale=se)
    test = stats.ttest_1samp(values, 0)
    return {'hypothesis':name, 'estimate':estimate, 'ci_low':lo, 'ci_high':hi,
            'p_value':test.pvalue, 'n_years':len(values)}
sex_result = annual_contrast(sex_annual.female_minus_male, 'Female minus male (minutes)')
age_result = annual_contrast(age_difference, 'Age 60–64 minus 30–34 (minutes)')
friedman = stats.friedmanchisquare(*(age_annual[c] for c in age_annual.columns))
save_table(sex_annual.reset_index(), 'annual_sex_contrast')
save_table(age_annual.reset_index(), 'annual_age_means')
display(pd.DataFrame([sex_result, age_result]))
print(f'Secondary age omnibus Friedman statistic = {friedman.statistic:.2f}; p {p_text(friedman.pvalue)}.')
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(sex_annual.index, sex_annual.female_minus_male, 'o-', color='#167d9a')
axes[0].axhline(sex_result['estimate'], linestyle='--', color='#333333')
axes[0].set(title='The sex-category gap persists across years', ylabel='Female − male mean (minutes)', xlabel='Race year')
axes[1].plot(age_difference.index, age_difference, 'o-', color='#e18a3b')
axes[1].axhline(age_result['estimate'], linestyle='--', color='#333333')
axes[1].set(title='Older and younger age classes differ', ylabel='Age 60–64 − age 30–34 mean (minutes)', xlabel='Race year')
for ax in axes:
    ax.set_xticks([1999,2004,2009,2014,2018])
fig.tight_layout()
save_figure('04_annual_contrasts')
display(Markdown(f"**Magnitude:** the age-standardized female-minus-male difference averages **{sex_result['estimate']:.1f} minutes** "
                 f"(95% CI {sex_result['ci_low']:.1f}–{sex_result['ci_high']:.1f}). "
                 f"The sex-standardized 60–64-minus-30–34 contrast averages **{age_result['estimate']:.1f} minutes** "
                 f"(95% CI {age_result['ci_low']:.1f}–{age_result['ci_high']:.1f}). "
                 "These are differences between groups of finishers, not changes that an individual would necessarily experience."))
''')
md(r'''
## 5 · Weather: 20 race days, not hundreds of thousands of experiments

To compare annual finishing times with a stable demographic mix, use direct standardization:

\[
\bar{t}^{\mathrm{std}}_y=\sum_g w_g\bar{t}_{yg},
\]

where \(g\) is an age-class/sex combination and \(w_g\) is its proportion in the pooled primary cohort.
Every year must contain all 18 groups, and weights must sum to 1. This avoids mistaking a change
in the composition of the field for a weather effect. It does not adjust for training, experience,
qualification, equipment, or changes within an age class.

Primary model (one row per year, equal year weights):

\[
\bar{t}^{\mathrm{std}}_y=\beta_0+\beta_T T_y+\beta_Y(y-2009)+\epsilon_y.
\]

\(T_y\) is **mean daily temperature**, not the temperature experienced by an individual runner.
A linear year trend is a limited adjustment for secular change. With only 20 years we keep the model small.
We use HC3 heteroskedasticity-robust standard errors with a t reference distribution (17 residual degrees
of freedom). No full year fixed effects: these would absorb all annual weather variation.
''')
code('''
group_weights = df.groupby(['age_group','sex'], observed=True).size() / len(df)
wide = cell.pivot(index='YEAR', columns=['age_group','sex'], values='mean').reindex(columns=group_weights.index)
assert wide.notna().all().all(), 'Do not silently standardize a year with missing demographic cells.'
assert np.isclose(group_weights.sum(), 1)
annual = df.groupby('YEAR').minutes.agg(n='size', raw_mean='mean', median='median').reset_index()
annual['standardized_mean'] = annual.YEAR.map(wide.mul(group_weights, axis=1).sum(axis=1))
annual = annual.merge(weather, on='YEAR', validate='one_to_one')
annual['year_c'] = annual.YEAR - 2009
annual = save_table(annual, 'annual_weather_analysis')
weather_model = smf.ols('standardized_mean ~ AVG_TEMP_C + year_c', data=annual).fit(cov_type='HC3', use_t=True)
lo, hi = weather_model.conf_int().loc['AVG_TEMP_C']
weather_result = {'hypothesis':'Temperature (minutes per 1°C)', 'estimate':weather_model.params['AVG_TEMP_C'],
                  'ci_low':lo, 'ci_high':hi, 'p_value':weather_model.pvalues['AVG_TEMP_C'], 'n_years':len(annual)}
primary_tests = pd.DataFrame([sex_result, age_result, weather_result])
primary_tests['p_holm'] = multipletests(primary_tests.p_value, method='holm')[1]
primary_tests['reject_at_0_05'] = primary_tests.p_holm < .05
save_table(primary_tests, 'primary_tests')
display(annual[['YEAR','n','raw_mean','standardized_mean','AVG_TEMP_C']])
display(primary_tests)
print(weather_model.summary())
''')
code('''
# Remove the estimated calendar trend for a two-dimensional weather display.
annual['trend_adjusted_mean'] = annual.standardized_mean - weather_model.params['year_c'] * annual.year_c
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
axes[0].plot(annual.YEAR, annual.raw_mean, 'o-', label='Observed field', color='#9eabb6')
axes[0].plot(annual.YEAR, annual.standardized_mean, 'o-', label='Same age/sex mix', color='#167d9a')
axes[0].set(title='Composition explains part of the time trend', xlabel='Race year', ylabel='Mean finish time (minutes)')
axes[0].legend(frameon=False)
axes[0].set_xticks([1999, 2004, 2009, 2014, 2018])
axes[1].scatter(annual.AVG_TEMP_C, annual.trend_adjusted_mean, color='#167d9a', s=55)
# Label only selected years so the figure remains readable; the full table is above.
for _, row in annual.loc[annual.YEAR.isin([1999,2000,2006,2013,2018])].iterrows():
    axes[1].annotate(str(int(row.YEAR)), (row.AVG_TEMP_C,row.trend_adjusted_mean),
                     xytext=(4,5), textcoords='offset points', fontsize=9)
grid = pd.DataFrame({'AVG_TEMP_C':np.linspace(annual.AVG_TEMP_C.min(),annual.AVG_TEMP_C.max(),100), 'year_c':0})
pred_grid = weather_model.get_prediction(grid).summary_frame()
axes[1].plot(grid.AVG_TEMP_C, pred_grid['mean'], color='#e18a3b', linewidth=2)
axes[1].fill_between(grid.AVG_TEMP_C, pred_grid.mean_ci_lower, pred_grid.mean_ci_upper, color='#e18a3b',alpha=.15)
axes[1].set(title='Warmer years tend to be slower', xlabel='Mean race-day temperature (°C)',
            ylabel='Standardized mean, trend removed (minutes)')
fig.tight_layout()
save_figure('05_weather')
display(Markdown(f"The fitted association is **{weather_result['estimate']:.2f} minutes per 1°C** "
                 f"(95% CI {lo:.2f}–{hi:.2f}). A 5°C difference corresponds to "
                 f"**{5*weather_result['estimate']:.1f} minutes** in the standardized annual mean, "
                 "within the observed temperature range. This is an association, not a personal pacing formula or a causal estimate."))
''')
md('''
### Other weather variables: exploratory, not a search for the smallest p-value

The three temperature measures are related. We do not put all of them in a 20-row model and interpret
unstable coefficients. Below are annual Pearson correlations between the standardized outcome and each
available weather measure. They are descriptive and do not control for calendar time.
Precipitation is **not** relative humidity. The `CLOUD_hrs` label is retained from the source and is not
used to make physical claims because its measurement definition is unclear.
''')
code('''
weather_columns = ['AVG_TEMP_C','MAX_TEMP_C','MIN_TEMP_C','PRECIP_mm','SUNSHINE_hrs','ATMOS_PRESS_mbar']
correlations = []
for col in weather_columns:
    result = stats.pearsonr(annual[col], annual.standardized_mean)
    correlations.append({'variable':col, 'pearson_r':result.statistic, 'unadjusted_p':result.pvalue, 'n_years':len(annual)})
correlations = save_table(pd.DataFrame(correlations), 'exploratory_correlations')
display(correlations)
fig, ax = plt.subplots(figsize=(8, 5.5))
sns.heatmap(annual[['standardized_mean']+weather_columns].corr(), annot=True, fmt='.2f', cmap='vlag',
            vmin=-1,vmax=1, center=0, ax=ax, cbar_kws={'label':'Annual Pearson correlation'})
ax.set_title('Weather measures overlap; correlations are exploratory')
fig.tight_layout()
save_figure('06_correlations')
''')
md('''
## 6 · Does the temperature finding survive reasonable changes?

We check: no calendar trend; a quadratic trend; adding precipitation; extending ages to 84 (pooling ages 65–84 to avoid empty cells);
trimming the outer 1% of finish times; restricting the era to 2005–2018; and allowing lag-1 serial
correlation in errors (HAC). These are sensitivity analyses, not competing models from which we select
the most significant. Leave-one-year-out refits show whether a single weather day drives the slope.
Diagnostics concern the 20 annual residuals, not the hundreds of thousands of runner rows.
''')
code('''
def standardized_annual(subset):
    weights = subset.groupby(['age_code','GENDER']).size() / len(subset)
    means = subset.groupby(['YEAR','age_code','GENDER']).minutes.mean().unstack(['age_code','GENDER'])
    means = means.reindex(columns=weights.index)
    assert means.notna().all().all()
    out = means.mul(weights, axis=1).sum(axis=1).rename('standardized_mean').reset_index()
    out = out.merge(weather, on='YEAR', validate='one_to_one')
    out['year_c'] = out.YEAR - 2009
    return out

primary_formula = 'standardized_mean ~ AVG_TEMP_C + year_c'
trim_lo, trim_hi = df.minutes.quantile([.01,.99])
extended = work.loc[work.YEAR.between(1999,2018) & work.age_code.isin(list(AGE_LABELS)+[70,75,80])
                    & work.minutes.gt(0) & work.GENDER.isin(['male','female'])].copy()
# Rare older categories have empty year/sex cells; combine them before standardizing.
extended['age_code'] = extended.age_code.clip(upper=65)
specifications = [
    ('Primary: age 20–69, linear trend', annual, primary_formula),
    ('No calendar trend', annual, 'standardized_mean ~ AVG_TEMP_C'),
    ('Quadratic calendar trend', annual, primary_formula+' + I(year_c**2)'),
    ('Add daily precipitation', annual, primary_formula+' + PRECIP_mm'),
    ('Include 70–84, pooled with 65–69', standardized_annual(extended), primary_formula),
    ('Trim outer 1% of finish times', standardized_annual(df.loc[df.minutes.between(trim_lo,trim_hi)]), primary_formula),
    ('Later period: 2005–2018', standardized_annual(df.loc[df.YEAR.ge(2005)]), primary_formula)
]
sensitivity_rows = []
for name, data, formula in specifications:
    fit = smf.ols(formula, data=data).fit(cov_type='HC3', use_t=True)
    limits = fit.conf_int().loc['AVG_TEMP_C']
    sensitivity_rows.append({'specification':name, 'slope':fit.params['AVG_TEMP_C'],
                             'ci_low':limits.iloc[0], 'ci_high':limits.iloc[1], 'n_years':len(data)})
hac = smf.ols(primary_formula, data=annual).fit(cov_type='HAC', cov_kwds={'maxlags':1,'use_correction':True}, use_t=True)
hac_ci = hac.conf_int().loc['AVG_TEMP_C']
sensitivity_rows.append({'specification':'Primary, lag-1 HAC errors', 'slope':hac.params['AVG_TEMP_C'],
                         'ci_low':hac_ci.iloc[0], 'ci_high':hac_ci.iloc[1], 'n_years':len(annual)})
sensitivity = save_table(pd.DataFrame(sensitivity_rows), 'weather_sensitivity')
loo_rows = []
for year in annual.YEAR:
    fit = smf.ols(primary_formula, data=annual.loc[annual.YEAR.ne(year)]).fit()
    loo_rows.append({'omitted_year':int(year), 'slope':fit.params['AVG_TEMP_C']})
loo = save_table(pd.DataFrame(loo_rows), 'leave_one_year_out')
display(sensitivity)
print(f'Leave-one-year-out temperature slopes: {loo.slope.min():.2f} to {loo.slope.max():.2f} min/°C.')
''')
code('''
fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(sensitivity))
ax.errorbar(sensitivity.slope, y, xerr=[sensitivity.slope-sensitivity.ci_low,
                                      sensitivity.ci_high-sensitivity.slope], fmt='o', color='#167d9a', capsize=4)
ax.axvline(0,color='#999999',linestyle='--')
ax.set_yticks(y, sensitivity.specification)
ax.invert_yaxis()
ax.set(xlabel='Temperature association (minutes per 1°C), 95% CI', title='The result across alternative specifications')
fig.tight_layout()
save_figure('07_sensitivity')
''')
code('''
ordinary_fit = smf.ols(primary_formula, data=annual).fit()
influence = ordinary_fit.get_influence()
diagnostics = annual[['YEAR']].copy()
diagnostics['fitted'] = ordinary_fit.fittedvalues
diagnostics['residual'] = ordinary_fit.resid
diagnostics['cooks_distance'] = influence.cooks_distance[0]
diagnostics['leverage'] = influence.hat_matrix_diag
save_table(diagnostics, 'annual_diagnostics')
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].scatter(diagnostics.fitted, diagnostics.residual, color='#167d9a')
axes[0].axhline(0,color='#777777',linestyle='--')
for _, row in diagnostics.nlargest(3,'cooks_distance').iterrows():
    axes[0].annotate(str(int(row.YEAR)), (row.fitted,row.residual), xytext=(4,3),textcoords='offset points',fontsize=9)
axes[0].set(title='Residuals vs fitted',xlabel='Fitted mean (minutes)',ylabel='Residual (minutes)')
stats.probplot(ordinary_fit.resid,dist='norm',plot=axes[1])
axes[1].set(title='Annual residual Q–Q',xlabel='Normal quantiles',ylabel='Residual (minutes)')
axes[2].plot(annual.YEAR,ordinary_fit.resid,'o-',color='#167d9a')
axes[2].axhline(0,color='#777777',linestyle='--')
axes[2].set(title='Residuals over time',xlabel='Race year',ylabel='Residual (minutes)')
fig.tight_layout()
save_figure('08_diagnostics')
print('Largest Cook distances (not automatic exclusions):')
display(diagnostics.nlargest(3,'cooks_distance'))
''')
md('''
### Exploratory age–temperature interaction

Does the older-minus-younger mean-time gap change in warmer years? Regress the annual standardized
60–64-minus-30–34 contrast on temperature and year. This is a secondary analysis with only 20 years;
a wide interval cannot establish that age groups respond equally. It does not measure individual
heat tolerance or account for acclimatization.
''')
code('''
interaction_data = annual[['YEAR','AVG_TEMP_C','year_c']].copy()
interaction_data['age_gap'] = interaction_data.YEAR.map(age_difference)
interaction_model = smf.ols('age_gap ~ AVG_TEMP_C + year_c',interaction_data).fit(cov_type='HC3',use_t=True)
interaction_ci = interaction_model.conf_int().loc['AVG_TEMP_C']
interaction_result = {'slope':interaction_model.params['AVG_TEMP_C'], 'ci_low':interaction_ci.iloc[0],
                      'ci_high':interaction_ci.iloc[1], 'p_value':interaction_model.pvalues['AVG_TEMP_C']}
display(pd.DataFrame([interaction_result]))
''')
md('''
## 7 · Can we predict a runner's finish in a future race?

A significant group association does not guarantee a useful individual prediction.
We train on **1999–2013** and evaluate once on **2014–2018**. All five test years are excluded from fitting.
The feature sets are fixed before evaluating the test set, with no test-driven tuning.

Models are deliberately interpretable: a training-mean baseline; age-class × sex group means;
groups plus a linear calendar trend; and groups plus trend and temperature.
We also remove sex or age from the full model to compare predictive contributions on the same test rows.
The full interaction group accounts for different age patterns by recorded sex.

All encoding and coefficients are learned on training data only. Calendar year and daily weather are
known here retrospectively. This is **prediction conditional on observed weather**, not an operational
pre-race weather forecast. There are no split times, final ranks, or other variables derived from finish time.
Unidentified returning runners can occur across years, so the split tests a future race, not necessarily a new person.

**MAE** is the average absolute error in minutes. **RMSE** penalizes large misses more.
**R²** compares squared error with predicting the test-set mean; negative values are possible.
Report every model and every held-out year. Do not choose a winner from this test set and call it validated.
''')
code('''
train = df.loc[df.YEAR.le(2013)].copy()
test = df.loc[df.YEAR.ge(2014)].copy()
assert set(train.YEAR).isdisjoint(test.YEAR)
print(f'Training: {len(train):,} finishes, {train.YEAR.min()}–{train.YEAR.max()}')
print(f'Test: {len(test):,} finishes, {test.YEAR.min()}–{test.YEAR.max()}')

model_specs = {
    'Training mean': ([], []),
    'Age + sex': (['group'], []),
    'Age + sex + year': (['group'], ['year_c']),
    'Age + sex + year + temperature': (['group'], ['year_c','AVG_TEMP_C']),
    'Full without sex': (['age_group'], ['year_c','AVG_TEMP_C']),
    'Full without age': (['sex'], ['year_c','AVG_TEMP_C'])
}
predictions = {}
metric_rows, year_metric_rows = [], []
for name, (categorical, numeric) in model_specs.items():
    if not categorical and not numeric:
        model = DummyRegressor(strategy='mean')
        model.fit(np.zeros((len(train),1)),train.minutes)
        prediction = model.predict(np.zeros((len(test),1)))
    else:
        encoder = ColumnTransformer([
            ('categories',OneHotEncoder(drop='first',handle_unknown='error'),categorical),
            ('numeric','passthrough',numeric)
        ])
        model = make_pipeline(encoder,LinearRegression())
        model.fit(train[categorical+numeric],train.minutes)
        prediction = model.predict(test[categorical+numeric])
    predictions[name] = prediction
    metric_rows.append({'model':name,'mae':mean_absolute_error(test.minutes,prediction),
                        'rmse':np.sqrt(mean_squared_error(test.minutes,prediction)),
                        'r2':r2_score(test.minutes,prediction)})
    for year in sorted(test.YEAR.unique()):
        mask = test.YEAR.eq(year).to_numpy()
        year_metric_rows.append({'model':name,'year':int(year),'n':int(mask.sum()),
                                'mae':mean_absolute_error(test.minutes.to_numpy()[mask],prediction[mask]),
                                'bias_pred_minus_actual':float((prediction[mask]-test.minutes.to_numpy()[mask]).mean())})
metrics = save_table(pd.DataFrame(metric_rows),'prediction_metrics')
year_metrics = save_table(pd.DataFrame(year_metric_rows),'prediction_by_year')
display(metrics)
display(year_metrics.pivot(index='year',columns='model',values='mae'))
''')
code('''
full_name = 'Age + sex + year + temperature'
base_mae = metrics.set_index('model').loc['Training mean','mae']
full_mae = metrics.set_index('model').loc[full_name,'mae']
no_weather_mae = metrics.set_index('model').loc['Age + sex + year','mae']
fig, axes = plt.subplots(1,2,figsize=(12,4.5),gridspec_kw={'width_ratios':[1.1,1]})
show_metrics = metrics.iloc[:4]
axes[0].barh(show_metrics.model,show_metrics.mae,color=['#9eabb6','#167d9a','#42a3b5','#e18a3b'])
axes[0].invert_yaxis()
axes[0].set(xlabel='Mean absolute error (minutes); lower is better',title='Group information has limited predictive power')
for i, value in enumerate(show_metrics.mae):
    axes[0].text(value+.25,i,f'{value:.1f}',va='center')
axes[0].set_xlim(0,show_metrics.mae.max()+5)
for name,color in [('Age + sex + year','#167d9a'),(full_name,'#e18a3b')]:
    data = year_metrics.loc[year_metrics.model.eq(name)]
    axes[1].plot(data.year,data.mae,'o-',label=name,color=color)
axes[1].set(xlabel='Held-out race year',ylabel='MAE (minutes)',title='Weather raises MAE in four of five years',xticks=range(2014,2019))
axes[1].legend(fontsize=8,frameon=False)
fig.tight_layout()
save_figure('09_prediction')
print(f'Adding temperature changes pooled MAE by {full_mae-no_weather_mae:+.3f} minutes (negative is better).')
print(f'Full-model improvement over baseline: {100*(base_mae-full_mae)/base_mae:.1f}%.')
''')
code('''
# Actual versus predicted, with every held-out row contributing to a density plot.
fig, ax = plt.subplots(figsize=(6.5,5))
hb = ax.hexbin(test.minutes,predictions[full_name],gridsize=45,mincnt=1,bins='log',cmap='Blues')
ax.plot([120,540],[120,540],'--',color='#e18a3b',label='Perfect prediction')
ax.set(xlabel='Actual finish (minutes)',ylabel='Predicted finish (minutes)',
       title='Many runners differ sharply from their group mean',xlim=(120,540),ylim=(120,540))
fig.colorbar(hb,ax=ax,label='Finish records (log color scale)')
ax.legend(frameon=False)
fig.tight_layout()
save_figure('10_actual_predicted')
''')
md('''
## 8 · What the evidence can—and cannot—say

The demographic contrasts describe the people who finished these races. They cannot isolate the effects of
aging, sex, training, participation barriers, or experience. The temperature model compares years in one
city and adjusts for only a fixed demographic mix and a simple trend. Race organization, timing conventions,
course changes, wind, humidity, and unmeasured changes in the entrants remain potential confounders.
No runner identifier is available; finish records should never be described as unique people.

The modern analysis period contains only **20 weather observations**. Robust standard errors and sensitivity
checks help assess stability but cannot remove unmeasured confounding. Finisher-only data cannot quantify
weather's association with withdrawal or failure to finish. The estimates do not identify causal effects.

The Berlin-only humidity hypothesis remains unanswered. Chapter 9 adds a Chicago humidity estimate;
Chapters 12–13 add previous-race records and a home-weather proxy. Actual training exposure and
harmonized route-level weather remain unmeasured. K-means is omitted because creating
performance labels from finish time would not answer the central questions.
''')
code('''
sex_holm, age_holm, weather_holm = primary_tests.p_holm.to_list()
full_r2 = metrics.set_index('model').loc[full_name,'r2']
display(Markdown(f"""
### Results in plain language

- **Demographics:** the age-standardized female-minus-male difference is {sex_result['estimate']:.1f} minutes;
  the sex-standardized 60–64-minus-30–34 difference is {age_result['estimate']:.1f} minutes.
  Holm-adjusted p-values are {sex_holm:.3g} and {age_holm:.3g}. The overlapping distributions matter as much as the averages.
- **Temperature:** a 5°C warmer day is associated with {5*weather_result['estimate']:.1f} additional minutes
  in the standardized annual mean (95% CI {5*weather_result['ci_low']:.1f}–{5*weather_result['ci_high']:.1f});
  Holm-adjusted p {p_text(weather_holm)}. Leave-one-year-out slopes range from {loo.slope.min():.2f} to {loo.slope.max():.2f} min/°C.
- **Prediction:** the full model misses individual finish times by {full_mae:.1f} minutes on average
  in 2014–2018 (R² = {full_r2:.3f}), compared with {base_mae:.1f} minutes for the training-mean baseline.
  Adding weather changes MAE by {full_mae-no_weather_mae:+.2f} minutes compared with the same model without temperature (positive means worse).
- **Humidity in Berlin:** unmeasured; the Chicago extension below adds a derived humidity analysis.

These data support meaningful group patterns while leaving much of individual performance unexplained.
"""))
''')
md('''
### Export a reproducible evidence bundle

All figures and tables above are saved in `outputs/`. This cell saves a compact JSON result summary
for the separate narrative report. The report is rendered from these calculated results, not manually
invented numbers. Dependencies, source hashes, and a local download script support reproduction.
''')
code('''
summary = {
    'n_raw':len(raw), 'n_analysis':len(df), 'n_years':int(df.YEAR.nunique()),
    'n_2019_male_only':int(raw.YEAR.eq(2019).sum()),
    'n_train':len(train), 'n_test':len(test), 'mean_minutes':float(df.minutes.mean()),
    'median_minutes':float(df.minutes.median()),
    'p10':float(df.minutes.quantile(.1)), 'p90':float(df.minutes.quantile(.9)),
    'temperature_min':float(annual.AVG_TEMP_C.min()), 'temperature_max':float(annual.AVG_TEMP_C.max()),
    'primary_tests':primary_tests.to_dict(orient='records'),
    'prediction_metrics':metrics.to_dict(orient='records'),
    'sensitivity':sensitivity.to_dict(orient='records'),
    'loo_slope_min':float(loo.slope.min()), 'loo_slope_max':float(loo.slope.max()),
    'interaction':interaction_result, 'friedman_p':float(friedman.pvalue),
    'raw_trend_change':float(annual.raw_mean.iloc[-1]-annual.raw_mean.iloc[0]),
    'standardized_trend_change':float(annual.standardized_mean.iloc[-1]-annual.standardized_mean.iloc[0]),
    'source':manifest,
    'environment':{'python':platform.python_version(), **{name:importlib.metadata.version(name)
                   for name in ['pandas','numpy','scipy','matplotlib','seaborn','statsmodels','scikit-learn','nbformat']}}
}
(OUT / 'results.json').write_text(json.dumps(summary,indent=2,default=float)+'\\n')
# Regenerate the self-contained narrative from this run's figures and numbers.
import runpy
runpy.run_path(str(ROOT / 'scripts' / 'render_berlin_report.py'), run_name='__main__')
print('Saved Berlin outputs/results.json and berlin_story.html.')
''')
md('''
### Sources and methods references

- [Kaggle: Berlin marathon data, version 2](https://www.kaggle.com/datasets/aiaiaidavid/berlin-marathons-data) — raw results, weather, license and attribution.
- [Primary research: Elite Marathoners Run Faster With Increasing Temperatures in Berlin Marathon](https://pmc.ncbi.nlm.nih.gov/articles/PMC8293098/) — collection context; our analysis concerns the broader field, not just elite runners.
- [Berlin results API age-class documentation](https://github.com/hmelberg/berlin-marathon#age-classes) — age-code interpretation.
- [Statsmodels robust covariance documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html) — HC3 and HAC implementation.
- [Scikit-learn regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) — MAE, RMSE and R² definitions.

All numerical findings in this notebook are calculated from the downloaded source. External research is
context and provenance, not a substitute for this project's results.
''')
from extension_cells import add_extension
add_extension(md, code)
from training_cells import add_training
add_training(md, code)
nb.cells = cells
nb.metadata = {'kernelspec':{'display_name':'Python (marathon project)','language':'python','name':'python3'},
               'language_info':{'name':'python','version':'3.12'}}
nbf.write(nb, ROOT / 'marathon_analysis.ipynb')
print(f'Created notebook: {len(cells)} cells ({sum(c.cell_type=="code" for c in cells)} code).')
