"""Render the statistical story using only executed notebook outputs."""
from pathlib import Path
import json
import base64
from html import escape
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
s = json.loads((OUT / 'results.json').read_text())
sex, age, heat = s['primary_tests']
metrics = {x['model']:x for x in s['prediction_metrics']}
full = metrics['Age + sex + year + temperature']
baseline = metrics['Training mean']
no_heat = metrics['Age + sex + year']
years = pd.read_csv(OUT / 'tables' / 'prediction_by_year.csv')
annual = pd.read_csv(OUT / 'tables' / 'annual_weather_analysis.csv')
flow = pd.read_csv(OUT / 'tables' / 'cleaning_flow.csv')

def figure(name, caption):
    data = base64.b64encode((OUT / 'figures' / f'{name}.png').read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{escape(caption)}"><figcaption>{caption}</figcaption></figure>'

def pvalue(p):
    return '&lt; 0.001' if p < .001 else f'= {p:.3f}'

def row_table(headers, rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{escape(str(x))}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{escape(str(x))}</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

prediction_table = row_table(['Model','MAE (min)','RMSE (min)','R²'],
    [[x['model'],f"{x['mae']:.2f}",f"{x['rmse']:.2f}",f"{x['r2']:.3f}"] for x in s['prediction_metrics']])
primary_table = row_table(['Comparison','Estimate','95% interval','Holm-adjusted p'],
    [[x['hypothesis'],f"{x['estimate']:.2f}",f"{x['ci_low']:.2f} to {x['ci_high']:.2f}",f"{x['p_holm']:.3g}"] for x in s['primary_tests']])
sensitivity_table = row_table(['Specification','Minutes per °C','95% interval','Years'],
    [[x['specification'],f"{x['slope']:.2f}",f"{x['ci_low']:.2f} to {x['ci_high']:.2f}",x['n_years']] for x in s['sensitivity']])
flow_table = row_table(['Cleaning step','Records retained','Excluded at this step'],
    [[r.step,f'{r.remaining:,}',f'{r.excluded_at_step:,}'] for r in flow.itertuples()])
interaction=s['interaction']
weather_improvement=no_heat['mae']-full['mae']
weather_sentence = (f'Adding temperature reduces average absolute error by only <strong>{weather_improvement:.2f} minutes</strong>'
                    if weather_improvement>=0 else f'Adding temperature increases average absolute error by <strong>{-weather_improvement:.2f} minutes</strong>')
sex_ablation=metrics['Full without sex']['mae']-full['mae']
age_ablation=metrics['Full without age']['mae']-full['mae']
report=f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>What makes a marathon slower? | Going for the Gold</title>
<style>
:root{{--ink:#163043;--muted:#546575;--teal:#167d9a;--orange:#d98131;--paper:#f4f6f7;}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--paper);font:17px/1.75 Georgia,serif}}
header{{background:#123044;color:white;padding:70px max(6vw,24px) 55px}}header .inner,main,footer{{max-width:1050px;margin:auto}}
.kicker{{font:700 12px/1.5 system-ui,sans-serif;letter-spacing:.15em;text-transform:uppercase;color:#7ed0da}}
h1{{font:700 clamp(38px,6vw,66px)/1.08 system-ui,sans-serif;letter-spacing:-.045em;max-width:850px;margin:20px 0}}
.deck{{font:20px/1.6 system-ui,sans-serif;max-width:760px;color:#d5e3eb}}
.meta{{font:13px system-ui,sans-serif;color:#b3c9d6;margin-top:25px}}main{{background:white;padding:45px 65px}}
h2{{font:700 29px/1.25 system-ui,sans-serif;letter-spacing:-.025em;margin:50px 0 18px}}h3{{font:700 19px/1.4 system-ui,sans-serif}}
p{{margin:14px 0}}a{{color:#096b89}}.intro{{font-size:21px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:35px 0}}
.card{{background:#edf5f7;border-top:4px solid var(--teal);padding:21px 20px;font:14px/1.5 system-ui,sans-serif}}
.card strong{{display:block;font-size:30px;letter-spacing:-.035em;margin-bottom:8px}}.callout{{border-left:4px solid var(--orange);background:#fff6ec;padding:18px 24px;margin:28px 0}}
figure{{margin:30px -20px}}img{{display:block;width:100%;height:auto}}figcaption{{font:13px/1.6 system-ui,sans-serif;color:var(--muted);padding:12px 20px}}
table{{border-collapse:collapse;width:100%;font:13px/1.5 system-ui,sans-serif;margin:20px 0}}th{{background:#e9f1f4;text-align:left}}th,td{{padding:11px 12px;border-bottom:1px solid #d8e0e5}}
.table-wrap{{overflow-x:auto}}.small{{font:14px/1.7 system-ui,sans-serif;color:var(--muted)}}footer{{padding:30px;font:13px/1.7 system-ui,sans-serif;color:var(--muted)}}
li{{margin:9px 0}}nav{{font:14px/1.7 system-ui,sans-serif;margin-bottom:35px}}nav a{{margin-right:15px}}details{{margin-top:30px}}summary{{cursor:pointer;font:700 17px system-ui,sans-serif}}
@media(max-width:700px){{main{{padding:25px 22px}}.cards{{grid-template-columns:1fr}}figure{{margin:25px -10px}}h2{{font-size:25px}}}}
@media print{{body,main{{background:white}}header{{padding:25px;color:#163043;background:white}}.deck,.meta,.kicker{{color:#546575}}main{{padding:0 20px}}nav{{display:none}}figure,.card,table{{break-inside:avoid}}h2{{break-after:avoid}}details{{display:block}}}}
</style></head><body>
<header><div class="inner"><div class="kicker">Going for the Gold · Group 9 · Introduction to Statistics</div>
<h1>What makes a marathon slower?</h1>
<p class="deck">Twenty years in Berlin reveal a weather signal, persistent demographic differences, and a large gap between explaining averages and predicting people.</p>
<p class="meta">Berlin Marathon · 1999–2018 · {s['n_analysis']:,} recorded finishes · Primary age classes 20–69</p></div></header>
<main>
<nav aria-label="Report sections"><a href="#data">The data</a><a href="#demographics">The runners</a><a href="#weather">The weather</a><a href="#prediction">Prediction</a><a href="#limits">Limits</a><a href="#evidence">Evidence</a></nav>
<p class="intro">Imagine two editions of the same marathon, one cooler and one warmer. Even if the age and sex composition of the field stayed the same, would the average finishing time change? In these Berlin records, warmer race days tend to come with slower finishes. But knowing the weather and a runner’s demographic group still leaves much of that runner’s finish time unexplained.</p>
<div class="cards"><div class="card"><strong>{5*heat['estimate']:.1f} minutes</strong>Longer standardized mean finish associated with a 5°C warmer day.<br>95% CI: {5*heat['ci_low']:.1f}–{5*heat['ci_high']:.1f} min.</div>
<div class="card"><strong>{sex['estimate']:.1f} minutes</strong>Average female-minus-male difference after matching the age mix across categories.</div>
<div class="card"><strong>{full['mae']:.1f} minutes</strong>Average absolute prediction error for individual finishes in five later race years.</div></div>
<p class="small">All numbers are calculated from the companion notebook. “Associated with” does not mean “caused by.” A row is a recorded finish, not necessarily a unique runner.</p>

<h2 id="data">The first result was a missing group of runners</h2>
<p>The <a href="https://www.kaggle.com/datasets/aiaiaidavid/berlin-marathons-data">Kaggle dataset</a> advertises results from 1974 through 2019, with a separate file of daily weather. That sounds like a ready-made long-term study. Inspection shows otherwise: early years contain unfamiliar age codes, 1978 and 1980 have no result records, and all <strong>{s['n_2019_male_only']:,} records in 2019 are male</strong>.</p>
<p>Including that last year in a simple average would change the population being compared. Its faster average could be mistaken for improved performance or favorable weather. We exclude 2019 and use the contiguous period 1999–2018, which has both recorded sex categories in every year. This is a data-quality decision, not a decision to remove an inconvenient result.</p>
<p>The age field requires another correction: a value of 40 is an age class, not an exact age. The primary analysis uses the common classes from 20–29 through 65–69. After applying the period, age, category, and valid-time rules, <strong>{s['n_analysis']:,} finish records</strong> remain. Identical anonymized rows are retained because different runners can share the same recorded attributes and finish time. Valid long finishes are not automatically deleted as outliers.</p>
<p>There is also a firm boundary to the original proposal: <strong>the source does not measure humidity.</strong> Rainfall is not a substitute. The humidity hypothesis remains unanswered.</p>

<h2 id="demographics">A typical finish is about four hours—but “typical” covers a lot</h2>
<p>The median finish in the study cohort is <strong>{s['median_minutes']:.1f} minutes</strong>; the mean is {s['mean_minutes']:.1f}. The middle 80% of records span {s['p10']:.1f} to {s['p90']:.1f} minutes. That wide spread is the first reason to be cautious about predicting a person from a group average.</p>
{figure('01_distributions','Figure 1. Finish-time distributions overlap substantially. Each histogram is normalized within its recorded category. Outlier markers are hidden in the boxplot only; valid long times remain in calculations.')}
<p>The age profile is curved rather than a simple straight-line decline. Means are relatively close across the younger adult classes and rise more clearly in older groups. These are different groups of finishers observed at different ages—not the same people followed as they age. Differences in experience, participation, and who is able to finish may contribute.</p>
{figure('03_age_pattern','Figure 2. Pooled mean finishing times by age class and recorded sex, 1999–2018. Lines connect category means; they do not represent individual aging trajectories.')}
<p>To make the comparisons fairer, we compare the sex categories using the same age distribution, and compare age classes using the same sex distribution. Each race year contributes one comparison. The average female-minus-male difference is <strong>{sex['estimate']:.1f} minutes</strong> (95% CI {sex['ci_low']:.1f}–{sex['ci_high']:.1f}); the 60–64-minus-30–34 difference is <strong>{age['estimate']:.1f} minutes</strong> (95% CI {age['ci_low']:.1f}–{age['ci_high']:.1f}). Both primary demographic tests remain below 0.05 after Holm adjustment.</p>
<p>These are substantial differences in group averages. They do not mean that every runner in one group is faster than every runner in another. The large overlap in Figure 1 is part of the result, not noise to be ignored.</p>

<h2 id="weather">Warmer days leave a signal in the average</h2>
<p>Weather is shared within a race year. The effective weather sample is therefore <strong>{s['n_years']} race days</strong>, not {s['n_analysis']:,} independent experiments. We calculate an annual mean for each age-class/sex combination and combine those means with fixed weights. This asks how each year would look if every edition had the same demographic mix.</p>
<p>We then relate those standardized annual means to daily average temperature, with a linear calendar-year trend to account partly for longer-term changes. Recorded daily average temperatures range from {s['temperature_min']:.1f}°C to {s['temperature_max']:.1f}°C. They are a proxy for conditions on race day, not the exact heat exposure along the course.</p>
{figure('05_weather','Figure 3. Left: annual observed and demographic-standardized means. Right: standardized annual means with the fitted linear calendar trend removed, plotted against daily average temperature. The shaded band is a 95% confidence interval for the fitted mean, not a prediction interval for runners.')}
<p>The estimated association is <strong>{heat['estimate']:.2f} additional minutes per 1°C</strong> (95% CI {heat['ci_low']:.2f}–{heat['ci_high']:.2f}; Holm-adjusted p {pvalue(heat['p_holm'])}). A 5°C difference translates to about <strong>{5*heat['estimate']:.1f} minutes</strong> in the standardized field mean, within the observed range.</p>
<div class="callout">This estimate is not a rule to add {5*heat['estimate']:.1f} minutes to an individual’s goal time. It is a comparison of annual averages after limited adjustment. Humidity, wind, training, and other unmeasured changes can still confound it.</div>
<p>The finding is not dependent on retaining one particular year. Omitting each year in turn gives temperature slopes from <strong>{s['loo_slope_min']:.2f} to {s['loo_slope_max']:.2f} minutes per °C</strong>. The notebook also tests alternative trends, additional precipitation adjustment, older age classes, trimming, a later era, and serial-correlation-robust uncertainty. All variants are reported rather than selecting the smallest p-value.</p>
{figure('07_sensitivity','Figure 4. Temperature slopes and 95% confidence intervals under alternative specifications. Each analysis still treats race year as the weather observation; sensitivity checks do not establish causality.')}
<p>An exploratory age–temperature model estimates a change of {interaction['slope']:.2f} minutes per °C in the older-minus-younger gap (95% CI {interaction['ci_low']:.2f} to {interaction['ci_high']:.2f}; unadjusted p {pvalue(interaction['p_value'])}). This secondary comparison has only 20 years and should not be taken as a definitive statement about age-specific heat tolerance.</p>

<h2 id="prediction">Explaining averages is easier than predicting a person</h2>
<p>For a more demanding check, we train models using 1999–2013 and predict finishes in <strong>2014–2018</strong>. No records from those five race years are used to fit coefficients or encodings. The test set contains {s['n_test']:,} finishes. This preserves the challenge of encountering a new race year.</p>
{figure('09_prediction','Figure 5. Prediction errors on five years excluded from model fitting. MAE measures the typical absolute miss in minutes; lower is better. Adding temperature raises MAE in four of the five held-out years.')}
<p>The full model—age class, recorded sex, calendar trend, and temperature—has an MAE of <strong>{full['mae']:.1f} minutes</strong>, compared with {baseline['mae']:.1f} for simply predicting the training-set mean. Its test R² is {full['r2']:.3f}. That is some predictive value, but the remaining error is large.</p>
<p>{weather_sentence} compared with the same model without temperature. RMSE changes only from {no_heat['rmse']:.2f} to {full['rmse']:.2f} minutes, so the small difference depends on the error metric. The simpler age-and-sex model has the lowest observed MAE ({metrics['Age + sex']['mae']:.2f} minutes) among these six models; this is a test-set comparison, not independent validation of a model selected afterward. Removing sex from the full model changes MAE by {sex_ablation:+.2f} minutes; removing age changes it by {age_ablation:+.2f}. These are comparisons of predictive information on the same test years, not rankings of causal importance. The two demographic ablations also remove the corresponding age-by-sex interaction.</p>
{prediction_table}
<p>The weather association and the lack of improvement in individual MAE are compatible: weather can shift a year’s average while differences in ability, preparation, experience, and race execution dominate the spread among individuals. Those individual factors are not measured here.</p>
<p class="small">The forecasts condition on historically observed daily weather. They do not evaluate a pre-race weather forecast. No final rank or in-race split time is used as a predictor. Repeated runners cannot be identified, so a held-out year is not guaranteed to contain entirely new people.</p>

<h2 id="limits">What this project establishes</h2>
<p>The study answers three parts of the proposal: demographic groups differ in average finishing time; warmer Berlin race days are associated with slower standardized averages; and these variables provide only limited individual predictive accuracy. It cannot answer the humidity question. It also cannot isolate what would happen if the same runner trained differently, aged, or ran under experimentally changed weather.</p>
<p>The most defensible story is therefore about two scales. At the scale of a race field, demographic composition and weather help explain patterns measured in minutes. At the scale of a single runner, those patterns leave substantial uncertainty. Both findings matter.</p>
<ul>
<li><strong>Selection:</strong> only recorded finishers are analyzed. Entrants who did not finish are absent; missing records and changing participation can affect results.</li>
<li><strong>Time and place:</strong> the study concerns one event over 20 years. A linear trend cannot absorb every course, timing, equipment, or entry-policy change.</li>
<li><strong>Exposure:</strong> daily station weather is not each runner’s race-hour exposure. Humidity and wind are not modeled.</li>
<li><strong>Dependence:</strong> unidentified runners may recur. Annual observations reduce pseudoreplication but may still be serially related.</li>
<li><strong>Exploration:</strong> the analysis was designed after an initial audit and exploratory look. Three primary tests receive Holm adjustment; secondary checks remain exploratory.</li>
</ul>

<h2 id="evidence">Methods and evidence you can inspect</h2>
<p>Sex and age intervals use 20 annual standardized contrasts and t inference with 19 degrees of freedom. The primary temperature regression uses equal-weight annual observations, HC3 robust standard errors, and a t reference with 17 residual degrees of freedom. Direct standardization uses pooled age-class/sex proportions. Those pooled descriptive weights are separate from prediction: predictive preprocessing and coefficients use training years only. Intervals are pointwise 95% intervals; Holm adjustment applies to p-values, not to interval coverage.</p>
{primary_table}
<details><summary>Show the cleaning audit</summary>{flow_table}</details>
<details><summary>Show all weather sensitivity estimates</summary>{sensitivity_table}</details>
<p>The executed notebook, <code>marathon_analysis.ipynb</code>, contains the analysis in separate explained cells. Its tables and figures are also available under <code>outputs/</code>. This report embeds its charts, so it can be shared as a single HTML file and printed to PDF from a browser.</p>
<h3>Sources</h3>
<ol>
<li><a href="https://www.kaggle.com/datasets/aiaiaidavid/berlin-marathons-data">Berlin marathon data, Kaggle</a>, version 2. Results and weather; listed as CC0. Retrieved {escape(s['source']['downloaded_utc'][:10])}. Exact input hashes are in <code>data/raw/manifest.json</code>.</li>
<li><a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8293098/">Elite Marathoners Run Faster With Increasing Temperatures in Berlin Marathon</a>. Primary research provides collection context, including organizer results and Berlin Dahlem daily weather. The Kaggle CSVs do not independently specify station IDs or race dates. Our analysis concerns the broader field rather than only elite runners.</li>
<li><a href="https://github.com/hmelberg/berlin-marathon#age-classes">Berlin results API age-class documentation</a>. Supports interpreting numeric codes as age brackets.</li>
<li><a href="https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html">Statsmodels robust covariance documentation</a> and <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics">scikit-learn regression metrics</a>.</li>
</ol>
</main><footer>Group 9 · Going for the Gold · All reported statistics are generated from the executed notebook. No Kaggle credentials are included in this report.</footer>
</body></html>'''
(ROOT / 'berlin_story.html').write_text(report,encoding='utf-8')
print('Rendered berlin_story.html with embedded figures.')
