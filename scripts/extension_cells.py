"""Additional notebook chapters. All analysis code remains visible in notebook cells."""
def add_extension(md, code):
    md('''
    # 9 · Beyond Berlin: independent events and complementary measurements

    The original analysis used one source. This extension analyzes **four additional results/weather datasets**, followed by real training logs in Chapter 15,
    and audits a fifth candidate. It expands the questions rather than merely adding more rows.
    All extensions are **exploratory**, planned after reviewing the source schemas and some preliminary summaries.
    We report uncertainty, competing specifications, and null findings. No synthetic data are used.

    | Kaggle source | Role | Important restriction |
    |---|---|---|
    | [Chicago results v3](https://www.kaggle.com/datasets/runningwithrock/chicago-marathon-results) | Replicate heat association; add estimated humidity and wind | Age codes are group upper boundaries; 2007 race halted |
    | [U.S. fall marathons v1](https://www.kaggle.com/datasets/runningwithrock/2010-2019-fall-marathons) | Compare fields across races | Fall races with ≥500 finishers, not a random sample of all races |
    | [Boston 2015–2017 v4](https://www.kaggle.com/datasets/rojour/boston-results) | Pacing, checkpoint prediction, previous-race history | Missing splits; no universal runner identifier |
    | [Boston 2024 weather/splits v2](https://www.kaggle.com/datasets/runningwithrock/2024-boston-marathon-weather-and-splits) | Home-weather proxy and slowdown | U.S. runners with ZIP codes only |
    | [Training pilot v2](https://www.kaggle.com/datasets/girardi69/marathon-time-predictions) | Data-quality/leakage audit only | 87 rows; mileage includes the race; ambiguous speed/time fields |

    **Do not sum these source counts into a count of unique people or unique finishes.** Chicago 2010–2019
    overlaps the fall dataset; Boston runners can recur; Berlin may include the same people too.
    The datasets support separate analyses. They are never stacked into one unexamined regression.
    Running with Rock supplied three of the additional datasets; different events are not necessarily
    independent data-collection pipelines. Attribution and source licenses are preserved in the manifests.
    ''')
    code('''
    EXP = OUT / 'expanded'
    ET = EXP / 'tables'
    EF = EXP / 'figures'
    for folder in [ET, EF]:
        folder.mkdir(parents=True, exist_ok=True)

    def etable(frame, name, index=False):
        frame.to_csv(ET / f'{name}.csv', index=index)
        return frame

    def efigure(name):
        plt.savefig(EF / f'{name}.png', bbox_inches='tight', facecolor='white')
        plt.show()

    def coefficient(fit, term, question, scale=1):
        ci = fit.conf_int().loc[term]
        return {'question':question, 'estimate':float(fit.params[term]*scale),
                'ci_low':float(ci.iloc[0]*scale), 'ci_high':float(ci.iloc[1]*scale),
                'p_value':float(fit.pvalues[term])}

    source_rows = []
    for name in ['chicago','fall_marathons','boston_2015_2017','boston_2024','training_pilot']:
        folder = ROOT / 'data' / 'raw' / name
        info = json.loads((folder / 'manifest.json').read_text())
        for filename, expected in info['files'].items():
            assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == expected
        source_rows.append({'source':name,'dataset':info['dataset'],'version':info['version'],
                            'license':info['license_as_listed'],'url':info['url']})
    source_catalog = etable(pd.DataFrame(source_rows), 'source_catalog')
    display(source_catalog)
    selection_decisions = {
        'runningwithrock/chicago-marathon-results':('Analyze','Second long event series; adds dew point and wind'),
        'runningwithrock/2010-2019-fall-marathons':('Analyze','Cross-race field comparisons; audit overlap with Chicago'),
        'rojour/boston-results':('Analyze','Splits and conservative historical linkage'),
        'runningwithrock/2024-boston-marathon-weather-and-splits':('Analyze','Home-weather proxy; pre-race date and ZIP audit'),
        'girardi69/marathon-time-predictions':('Audit only','Tiny sample; mileage includes race; speed anomaly; missing training fields'),
        'likithagedipudi/run-club-marathon-performance-dataset':('Exclude','Publisher explicitly describes synthetic generation'),
        'runningwithrock/nyc-marathon-results-all-years':('Not selected','Adds historical coverage but no weather; overlaps fall sample'),
        'runningwithrock/2024-marathon-results':('Not selected','Adds recency; not the missing training or weather measurements'),
        'mexwell/long-distance-running-dataset':('Analyze','Real training logs; supplementary race-day duration proxy, not official finish times')
    }
    selection = pd.DataFrame([{'dataset':ref,'decision':decision,'reason':reason}
                              for ref,(decision,reason) in selection_decisions.items()])
    etable(selection,'dataset_selection')
    expansion_audit = []
    ''')
    md(r'''
    ## 9.1 · Chicago: humidity becomes measurable, with qualifications

    Chicago results cover 1996–2023, excluding the canceled 2020 edition. We use the same age span as Berlin,
    **20–69**, but standardize using ten-year classes × recorded sex. The source's age values (24, 29, 34, …)
    denote class upper boundaries, not exact ages. Each retained year must contain all ten strata.

    The weather file includes 6 a.m., 9 a.m., and noon temperature, dew point, and wind. The primary exposure
    is **9 a.m. temperature**, interpreted as Fahrenheit and converted to Celsius. The header lacks units;
    that interpretation is consistent with U.S. Weather Spark conventions and the recorded weather range,
    but remains a source-documentation assumption. Wind is retained in **source units** because the file
    does not explicitly state its units. The analysis does not assert a per-mph effect.

    Estimated relative humidity uses the [EPA-documented Magnus approximation](https://qed.epa.gov/hms/meteorology/humidity/algorithms/):

    $$RH=100\exp\left[\frac{17.625T_d}{243.04+T_d}-\frac{17.625T}{243.04+T}\right],$$

    with air and dew-point temperatures in °C. This is a derived humidity proxy, not a directly reported
    humidity measurement. We do not enter temperature, dew point, and RH together as separate independent exposures.

    **2007 is excluded from the main regression because organizers halted the race**, changing who could finish
    ([World Athletics report](https://worldathletics.org/news/news/ivuti-and-adere-clinch-chicago-dramatic-chica)).
    We report a sensitivity analysis including it. The main weather sample is 26 race years.
    ''')
    code('''
    chicago_raw = pd.read_csv(ROOT / 'data/raw/chicago/Chicago Marathon Results.csv',
                              usecols=['Year','Gender','Age','Finish','Finish Time'])
    chicago_weather = pd.read_csv(ROOT / 'data/raw/chicago/Chicago Marathon Weather.csv')
    assert chicago_weather.Year.is_unique
    assert np.allclose(pd.to_timedelta(chicago_raw['Finish Time']).dt.total_seconds(),chicago_raw.Finish)
    assert set(chicago_raw.Age.unique()).issubset({19,24,29,34,39,44,49,54,59,64,69,74,79,80})
    chicago = chicago_raw.loc[chicago_raw.Age.between(20,69) & chicago_raw.Gender.isin(['M','F'])
                              & chicago_raw.Finish.gt(0)].copy()
    chicago['minutes'] = chicago.Finish / 60
    chicago['age_decade'] = (chicago.Age // 10 * 10).astype(int)
    chicago['sex'] = chicago.Gender
    chicago_weather['temp_c'] = (chicago_weather['9AM Temp'] - 32) / 1.8
    chicago_weather['dew_c'] = (chicago_weather['9AM Dew'] - 32) / 1.8
    chicago_weather['rh_pct'] = 100 * np.exp(
        17.625*chicago_weather.dew_c/(243.04+chicago_weather.dew_c)
        -17.625*chicago_weather.temp_c/(243.04+chicago_weather.temp_c))
    chicago_weather['wind_native'] = chicago_weather['9AM Wind']
    chicago_weather['year_c'] = chicago_weather.Year - 2009
    assert chicago_weather.rh_pct.between(0,100).all()
    assert chicago_weather.dew_c.le(chicago_weather.temp_c).all()
    # Standardization weights exclude the interrupted edition too.
    chi_reference = chicago.loc[chicago.Year.ne(2007)]
    chi_weights = chi_reference.groupby(['age_decade','sex']).size()/len(chi_reference)
    chi_means = chicago.groupby(['Year','age_decade','sex']).minutes.mean().unstack(['age_decade','sex'])
    chi_means = chi_means.reindex(columns=chi_weights.index)
    assert chi_means.notna().all().all() and np.isclose(chi_weights.sum(),1)
    chi_annual = chi_means.mul(chi_weights,axis=1).sum(axis=1).rename('standardized_mean').reset_index()
    chi_annual = chi_annual.merge(chicago.groupby('Year').minutes.agg(n='size',raw_mean='mean').reset_index(),on='Year')
    chi_annual = chi_annual.merge(chicago_weather,on='Year',validate='one_to_one',how='left')
    assert chi_annual.temp_c.notna().all()
    chi_main = chi_annual.loc[chi_annual.Year.ne(2007)].copy()
    etable(chi_annual,'chicago_annual')
    expansion_audit.append({'source':'Chicago','raw_rows':len(chicago_raw),'eligible_rows':len(chicago),
                            'analysis_rows':int(chi_main.n.sum()),'inference_unit_count':len(chi_main),
                            'note':'Ages 20–69; main weather fit excludes interrupted 2007.'})
    print(f'Chicago: {len(chicago_raw):,} raw finishes; {len(chicago):,} age-eligible; '
          f'{int(chi_main.n.sum()):,} in {len(chi_main)} main-analysis years.')
    display(chi_annual[['Year','n','standardized_mean','temp_c','rh_pct','wind_native']])
    ''')
    code('''
    chi_heat = smf.ols('standardized_mean ~ temp_c + year_c',chi_main).fit(cov_type='HC3',use_t=True)
    chi_environment = smf.ols('standardized_mean ~ temp_c + rh_pct + wind_native + year_c',chi_main).fit(cov_type='HC3',use_t=True)
    chi_results = pd.DataFrame([
        coefficient(chi_heat,'temp_c','Chicago: temperature, minutes per 5°C',5),
        coefficient(chi_environment,'temp_c','Chicago: temperature adjusted for RH/wind, per 5°C',5),
        coefficient(chi_environment,'rh_pct','Chicago: RH, minutes per 10 percentage points',10),
        coefficient(chi_environment,'wind_native','Chicago: wind, minutes per 5 source units',5)
    ])
    etable(chi_results,'chicago_coefficients')
    display(chi_results)
    print('Correlation of temperature with dew point:',chi_main[['temp_c','dew_c']].corr().iloc[0,1])
    print('Adjusted environmental model:')
    print(chi_environment.summary())
    ''')
    code('''
    chi_sens = []
    for name, data, formula in [
        ('Main: 9 a.m., exclude 2007',chi_main,'standardized_mean ~ temp_c + year_c'),
        ('Include interrupted 2007',chi_annual,'standardized_mean ~ temp_c + year_c'),
        ('Quadratic year trend',chi_main,'standardized_mean ~ temp_c + year_c + I(year_c**2)'),
        ('Add RH and wind',chi_main,'standardized_mean ~ temp_c + rh_pct + wind_native + year_c'),
        ('Exclude pandemic-return 2021',chi_main.loc[chi_main.Year.ne(2021)],'standardized_mean ~ temp_c + year_c'),
        ('9 a.m./noon mean temperature',chi_main.assign(temp_c=((chi_main['9AM Temp']+chi_main['12PM Temp'])/2-32)/1.8),
         'standardized_mean ~ temp_c + year_c')
    ]:
        fit = smf.ols(formula,data).fit(cov_type='HC3',use_t=True)
        chi_sens.append(coefficient(fit,'temp_c',name,5))
    chi_sens = etable(pd.DataFrame(chi_sens),'chicago_sensitivity')
    chi_loo = []
    for year in chi_main.Year:
        fit = smf.ols('standardized_mean ~ temp_c + year_c',chi_main.loc[chi_main.Year.ne(year)]).fit()
        chi_loo.append({'omitted_year':int(year),'slope_per_5c':5*fit.params.temp_c})
    chi_loo = etable(pd.DataFrame(chi_loo),'chicago_leave_one_year_out')
    display(chi_sens)
    print(f'Leave-one-year-out slopes: {chi_loo.slope_per_5c.min():.2f}–{chi_loo.slope_per_5c.max():.2f} minutes per 5°C.')
    ''')
    code('''
    fig, axes = plt.subplots(1,2,figsize=(12,4.6))
    chi_adjusted = chi_main.standardized_mean - chi_heat.params.year_c*chi_main.year_c
    axes[0].scatter(chi_main.temp_c,chi_adjusted,s=50,color='#167d9a')
    xx = np.linspace(chi_main.temp_c.min(),chi_main.temp_c.max(),100)
    axes[0].plot(xx,chi_heat.params.Intercept+chi_heat.params.temp_c*xx,color='#e18a3b')
    for year in [1999,2008,2021]:
        r=chi_main.loc[chi_main.Year.eq(year)].iloc[0]
        axes[0].annotate(str(year),(r.temp_c,r.standardized_mean-chi_heat.params.year_c*r.year_c),
                         xytext=(4,4),textcoords='offset points',fontsize=9)
    axes[0].set(xlabel='Chicago 9 a.m. air temperature (°C)',ylabel='Standardized mean, trend removed (min)',
                title='The heat association extends to Chicago')
    compare=pd.DataFrame([{'race':'Berlin: daily mean','estimate':5*weather_result['estimate'],
                           'ci_low':5*weather_result['ci_low'],'ci_high':5*weather_result['ci_high']},
                          {'race':'Chicago: 9 a.m.','estimate':chi_results.iloc[0].estimate,
                           'ci_low':chi_results.iloc[0].ci_low,'ci_high':chi_results.iloc[0].ci_high}])
    axes[1].errorbar(compare.estimate,[0,1],xerr=[compare.estimate-compare.ci_low,compare.ci_high-compare.estimate],
                     fmt='o',capsize=5,color='#167d9a')
    axes[1].set_yticks([0,1],compare.race)
    axes[1].axvline(0,color='#888888',linestyle='--')
    axes[1].set(xlabel='Minutes per 5°C, 95% CI',title='Same direction; different exposure measures')
    fig.tight_layout()
    efigure('11_chicago_replication')
    ''')
    md('''
    The Berlin and Chicago estimates have different weather definitions and demographic reference populations.
    Their agreement in direction is useful external evidence, but a formal pooled effect would require
    harmonized weather exposures. Estimated RH and wind enter a small 26-year model; wide intervals are
    inconclusive rather than evidence of zero effect. RH depends on temperature, so an unadjusted RH correlation
    can be especially misleading. We report adjusted estimates and do not equate humidity with precipitation.
    ''')
    md('''
    # 10 · How much do race fields differ after matching demographics?

    The fall-marathon source contains over two million results from 96 named races, across 546 race-year
    combinations. It excludes small races and seasons outside September–November. Some age values are exact;
    others are class boundaries. We harmonize only to broad decade groups.

    Retain ages 20–69, M/F categories, positive finish times, and race-years with **at least 10 finishes in
    each of ten age-decade/sex cells**. Retain races with at least eight eligible years. These rules support
    stable comparisons but select larger, recurrent races. Do not extrapolate to all marathons.

    Standardize every race-year to the same pooled age/sex distribution. Then fit annual means against race
    and year indicators, and predict each race with equal weight for all ten calendar years. Each race-year
    has equal weight in this second-stage comparison. Rankings describe the **field plus event context**:
    they cannot isolate hills, course speed, entry policies, weather, or training.
    Chicago overlaps the standalone source and is not an additional independent weather replication here.
    ''')
    code('''
    fall_raw = pd.read_csv(ROOT / 'data/raw/fall_marathons/Results.csv',
                           usecols=['Race','Year','Gender','Age','Finish'])
    fall = fall_raw.loc[fall_raw.Age.between(20,69) & fall_raw.Gender.isin(['M','F']) & fall_raw.Finish.gt(0)].copy()
    fall['age_decade'] = (fall.Age//10*10).astype(int)
    fall['minutes'] = fall.Finish/60
    fall_cells = fall.groupby(['Race','Year','age_decade','Gender']).minutes.agg(n='size',mean='mean').reset_index()
    fall_coverage = fall_cells.groupby(['Race','Year']).agg(cells=('n','size'),minimum_cell_n=('n','min'))
    eligible_events = fall_coverage.loc[(fall_coverage.cells==10)&(fall_coverage.minimum_cell_n>=10)].reset_index()
    race_year_counts = eligible_events.groupby('Race').Year.nunique()
    eligible_events = eligible_events.loc[eligible_events.Race.isin(race_year_counts[race_year_counts>=8].index)]
    fall_analysis = fall.merge(eligible_events[['Race','Year']],on=['Race','Year'],validate='many_to_one')
    fall_weights = fall_analysis.groupby(['age_decade','Gender']).size()/len(fall_analysis)
    fall_means = fall_analysis.groupby(['Race','Year','age_decade','Gender']).minutes.mean().unstack(['age_decade','Gender'])
    fall_means = fall_means.reindex(columns=fall_weights.index)
    assert fall_means.notna().all().all() and np.isclose(fall_weights.sum(),1)
    fall_annual = fall_means.mul(fall_weights,axis=1).sum(axis=1).rename('standardized_mean').reset_index()
    fall_annual = fall_annual.merge(fall_analysis.groupby(['Race','Year']).minutes.agg(raw_mean='mean',n='size').reset_index(),
                                  on=['Race','Year'],validate='one_to_one')
    fall_model = smf.ols('standardized_mean ~ C(Race) + C(Year)',fall_annual).fit()
    etable(fall_annual,'fall_race_years')
    expansion_audit.append({'source':'Fall marathons','raw_rows':len(fall_raw),'eligible_rows':len(fall),
                            'analysis_rows':len(fall_analysis),'inference_unit_count':len(fall_annual),
                            'note':f'{fall_analysis.Race.nunique()} races, >=8 eligible years; overlaps Chicago.'})
    print(f'Source: {len(fall_raw):,} rows, {fall_raw.Race.nunique()} races, {fall_raw.groupby(["Race","Year"]).ngroups} events.')
    print(f'Analysis: {len(fall_analysis):,} rows, {fall_analysis.Race.nunique()} races, {len(fall_annual)} events.')
    ''')
    code('''
    race_comparison = []
    for race in sorted(fall_annual.Race.unique()):
        subset = fall_annual.loc[fall_annual.Race.eq(race)]
        adjusted = fall_model.predict(pd.DataFrame({'Race':race,'Year':range(2010,2020)})).mean()
        race_comparison.append({'race':race,'n_years':len(subset),'n_finishes':int(subset.n.sum()),
                                'raw_mean':subset.raw_mean.mean(),'adjusted_mean':adjusted,
                                'annual_min':subset.standardized_mean.min(),'annual_max':subset.standardized_mean.max()})
    race_comparison = pd.DataFrame(race_comparison).sort_values('adjusted_mean').reset_index(drop=True)
    race_spread = float(race_comparison.adjusted_mean.quantile(.9)-race_comparison.adjusted_mean.quantile(.1))
    raw_race_spread = float(race_comparison.raw_mean.quantile(.9)-race_comparison.raw_mean.quantile(.1))
    etable(race_comparison,'race_comparison')
    display(race_comparison)
    print(f'90th–10th percentile spread of race means: raw {raw_race_spread:.1f}, adjusted {race_spread:.1f} min.')
    fig,ax=plt.subplots(figsize=(10,8))
    for i,row in race_comparison.iterrows():
        ax.plot([row.raw_mean,row.adjusted_mean],[i,i],color='#b8c4cc',linewidth=2)
    ax.scatter(race_comparison.raw_mean,race_comparison.index,color='#aab8c2',label='Raw annual means',s=35)
    ax.scatter(race_comparison.adjusted_mean,race_comparison.index,color='#167d9a',label='Same demographic mix + year adjustment',s=40)
    ax.set_yticks(race_comparison.index,race_comparison.race)
    ax.invert_yaxis();ax.legend(frameon=False,fontsize=9)
    ax.set(xlabel='Mean finish time (minutes)',title='Matching demographics does not erase race differences')
    fig.tight_layout();efigure('12_race_comparison')
    ''')
    md('''
    Race ordering is descriptive, not a league table of course difficulty. Runners self-select races, and
    charity entrants, qualification policies, time limits, and geography can change the field.
    A difference that remains after age/sex adjustment is an invitation to collect those variables,
    not permission to call the remaining difference an elevation effect.
    ''')
    md('''
    # 11 · Boston: where does performance break down?

    Boston 2015–2017 includes intermediate checkpoints. Define **second-half slowdown** as
    `100 × (second_half / first_half − 1)`: 10 means the second half took 10% longer.
    Do not call this a diagnosis of “hitting the wall”; terrain, pacing, heat, injury, and stops can all contribute.

    Checkpoint validation is specific to each analysis. A valid halfway comparison needs a half time and finish
    with `0 < half < finish`. Segment profiles require all checkpoints to increase. A first-10-km prediction
    needs `0 < 10K < finish`. We do not throw away an otherwise usable record because an unrelated split is missing.
    ''')
    code('''
    boston_parts = []
    for year in [2015,2016,2017]:
        part = pd.read_csv(ROOT / f'data/raw/boston_2015_2017/marathon_results_{year}.csv',low_memory=False)
        part['year'] = year
        for col in ['5K','10K','15K','20K','Half','25K','30K','35K','40K','Official Time']:
            part[col+'_min'] = pd.to_timedelta(part[col],errors='coerce').dt.total_seconds()/60
        part['sex'] = part['M/F']
        part['finish'] = part['Official Time_min']
        part['half'] = part['Half_min']
        part['second'] = part.finish-part.half
        boston_parts.append(part)
    boston_all = pd.concat(boston_parts,ignore_index=True)
    boston_valid = boston_all.loc[boston_all.half.gt(0) & boston_all.second.gt(0)
                                 & boston_all.sex.isin(['M','F']) & boston_all.Age.between(18,90)].copy()
    boston_valid['slow_pct'] = 100*(boston_valid.second/boston_valid.half-1)
    pacing_summary = boston_valid.groupby(['year','sex']).agg(n=('slow_pct','size'),
        mean_slow_pct=('slow_pct','mean'),median_slow_pct=('slow_pct','median'),
        negative_split_pct=('slow_pct',lambda x:100*x.lt(0).mean()),
        large_slowdown_pct=('slow_pct',lambda x:100*x.ge(20).mean())).reset_index()
    etable(pacing_summary,'boston_pacing_summary')
    expansion_audit.append({'source':'Boston 2015–2017','raw_rows':len(boston_all),
                            'eligible_rows':len(boston_valid),'analysis_rows':len(boston_valid),
                            'inference_unit_count':3,'note':'Three event years; repeated participants possible.'})
    display(pacing_summary)
    ''')
    code('''
    pace_adjustments = []
    for year, data in boston_valid.groupby('year'):
        fit = smf.ols('slow_pct ~ C(sex) + bs(Age,df=4) + bs(half,df=5)',data).fit(cov_type='HC3',use_t=True)
        result = coefficient(fit,'C(sex)[T.M]',f'{year}: male minus female slowdown, percentage points')
        result['year'] = int(year)
        pace_adjustments.append(result)
    pace_adjustments = etable(pd.DataFrame(pace_adjustments),'boston_adjusted_pacing')
    display(pace_adjustments)
    # Inference is conditional on each event. Do not pretend three weather days provide a heat test.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    sns.barplot(data=pacing_summary,x='year',y='median_slow_pct',hue='sex',hue_order=['M','F'],ax=axes[0])
    axes[0].set(xlabel='Race year',ylabel='Median second-half slowdown (%)',title='Slowdown differs by year and recorded sex')
    axes[1].errorbar(pace_adjustments.year,pace_adjustments.estimate,
                     yerr=[pace_adjustments.estimate-pace_adjustments.ci_low,pace_adjustments.ci_high-pace_adjustments.estimate],
                     fmt='o',capsize=5,color='#167d9a')
    axes[1].axhline(0,color='#888888',linestyle='--')
    axes[1].set(xticks=[2015,2016,2017],xlabel='Race year',ylabel='Male − female slowdown (percentage points)',
                title='Difference persists after age/half-time adjustment')
    fig.tight_layout();efigure('13_boston_pacing')
    ''')
    md('''
    Adjusting for first-half time compares runners who reached halfway at similar times; it is not a
    pre-race causal comparison. First-half time reflects ability, pacing decisions, and conditions and may
    itself be affected by the variables under study. The age and half-time splines allow curved relationships.
    Confidence intervals describe conditional within-year associations; they do not establish that the same
    effect holds at all marathons or explain why it occurs.
    ''')
    code('''
    checkpoint_cols = [f'{k}K_min' for k in [5,10,15,20,25,30,35,40]]+['finish']
    checkpoint_distances = np.array([5,10,15,20,25,30,35,40,42.195])
    increments = boston_all[checkpoint_cols].diff(axis=1)
    increments.iloc[:,0] = boston_all[checkpoint_cols[0]]
    valid_profile = increments.notna().all(axis=1)&increments.gt(0).all(axis=1)
    profile = increments.loc[valid_profile].div(np.diff(np.r_[0,checkpoint_distances]),axis=1)
    # Divide by the 5–10 km pace, avoiding crowded opening-kilometer pace as the reference.
    profile = profile.div(profile.iloc[:,1],axis=0)*100
    profile['year'] = boston_all.loc[valid_profile,'year']
    segment_profile = profile.groupby('year').median().T
    segment_profile.index = ['0–5','5–10','10–15','15–20','20–25','25–30','30–35','35–40','40–42.2']
    etable(segment_profile.reset_index(names='segment_km'),'boston_segment_profile')
    fig,ax=plt.subplots(figsize=(10,4.5))
    segment_profile.plot(marker='o',ax=ax)
    ax.axhline(100,color='#888888',linestyle='--')
    ax.set(xlabel='Course segment (km)',ylabel='Median pace relative to own 5–10 km pace (%)',
           title='Segment profiles locate slowdown, without identifying its cause')
    fig.tight_layout();efigure('14_boston_segments')
    print(f'{valid_profile.sum():,} complete, increasing split profiles of {len(boston_all):,} source records.')
    ''')
    md('''
    ## 11.1 · What can we predict after 10 km?

    Prediction time matters. A race-day checkpoint is not a pre-race predictor. To avoid a tautological
    advantage from predicting a total that already contains the observed split, the outcome is **time remaining
    after 10 km**, not the whole finish. Compare a training-mean baseline, demographics, an extrapolation of
    10-km pace, and a learned demographic-plus-10-km model.

    Fit on 2015–2016, test on 2017. As a sensitivity check, evaluate only 2017 records whose normalized
    name/sex/country/birth-year proxy does not appear in training. This screen reduces known repeat runners;
    it cannot certify distinct people. No `Proj Time`, final rank, or later checkpoint enters the models.
    ''')
    code('''
    from sklearn.preprocessing import SplineTransformer
    # Conservative linkage proxy, used only for split checks/history—not exported as personal data.
    def runner_key(frame):
        name = frame.Name.fillna('').str.normalize('NFKC').str.upper().str.replace(r'\\s+',' ',regex=True).str.strip()
        return name+'|'+frame.sex+'|'+frame.Country.fillna('')+'|'+(frame.year-frame.Age).astype(str)

    boston_all['runner_key'] = runner_key(boston_all)
    checkpoint = boston_all.loc[boston_all['10K_min'].gt(0)&boston_all.finish.gt(boston_all['10K_min'])
                               & boston_all.sex.isin(['M','F']) & boston_all.Age.between(18,90)].copy()
    checkpoint['remaining'] = checkpoint.finish-checkpoint['10K_min']
    checkpoint_train = checkpoint.loc[checkpoint.year.lt(2017)]
    checkpoint_test = checkpoint.loc[checkpoint.year.eq(2017)]
    new_proxy_mask = ~checkpoint_test.runner_key.isin(set(checkpoint_train.runner_key))

    def smooth_regression(categorical, numeric):
        preprocessor = ColumnTransformer([
            ('category',OneHotEncoder(drop='first',handle_unknown='ignore'),categorical),
            ('numeric',SplineTransformer(n_knots=5,degree=3,include_bias=False,extrapolation='linear'),numeric)
        ])
        return make_pipeline(preprocessor,LinearRegression())

    checkpoint_predictions = {
        'Training remaining-time mean':np.repeat(checkpoint_train.remaining.mean(),len(checkpoint_test)),
        'Constant 10-km pace':checkpoint_test['10K_min'].to_numpy()*(42.195-10)/10
    }
    for name,numeric in [('Demographics',['Age']),('Demographics + first 10 km',['Age','10K_min'])]:
        model = smooth_regression(['sex'],numeric)
        model.fit(checkpoint_train[['sex']+numeric],checkpoint_train.remaining)
        checkpoint_predictions[name] = model.predict(checkpoint_test[['sex']+numeric])
    checkpoint_metrics = []
    for sample,mask in [('All eligible 2017',np.ones(len(checkpoint_test),dtype=bool)),
                        ('No training identity-proxy match',new_proxy_mask.to_numpy())]:
        target = checkpoint_test.remaining.to_numpy()[mask]
        for name,prediction in checkpoint_predictions.items():
            checkpoint_metrics.append({'sample':sample,'model':name,'n':int(mask.sum()),
                                       'mae':mean_absolute_error(target,prediction[mask]),
                                       'rmse':np.sqrt(mean_squared_error(target,prediction[mask])),
                                       'r2':r2_score(target,prediction[mask])})
    checkpoint_metrics=etable(pd.DataFrame(checkpoint_metrics),'checkpoint_prediction')
    display(checkpoint_metrics)
    ''')
    md('''
    # 12 · Is a previous marathon more informative than demographic averages?

    Names are not unique IDs. We cautiously link only exact normalized name, recorded sex, country,
    and `year − age` matches across adjacent years; ambiguous keys within either year are excluded.
    This strict rule misses legitimate matches when birthdays or recorded fields change and can still
    falsely link different people. The analysis is restricted to the matched repeat-finisher subset.
    A sensitivity subset additionally requires the same city and state.

    Train on **2015 → 2016** pairs and test on **2016 → 2017** pairs. Prior race time is available before
    the target race. Some people can occur in both transitions: this is legitimate future prediction for
    known runners, not a claim of evaluating entirely unseen people. Compare a demographic spline model,
    simply repeating the previous time, and a calibrated demographic-plus-prior-time model.
    Do not interpret previous finish time as a direct measurement of training volume or experience.
    ''')
    code('''
    def adjacent_pairs(first_year):
        candidates=boston_all.loc[boston_all.finish.gt(0)&boston_all.Age.between(18,90)
                                  & boston_all.sex.isin(['M','F']) & boston_all.Name.notna()
                                  & boston_all.Country.notna()].copy()
        left=candidates.loc[candidates.year.eq(first_year)]
        right=candidates.loc[candidates.year.eq(first_year+1)]
        left=left.loc[~left.duplicated('runner_key',keep=False)]
        right=right.loc[~right.duplicated('runner_key',keep=False)]
        return left.merge(right,on='runner_key',suffixes=('_prior','_next'),validate='one_to_one')

    history_train,history_test = adjacent_pairs(2015),adjacent_pairs(2016)
    history_predictions = {'Repeat previous finish':history_test.finish_prior.to_numpy()}
    for name,numeric in [('Demographics',['Age_next']),('Demographics + previous finish',['Age_next','finish_prior'])]:
        model=smooth_regression(['sex_next'],numeric)
        model.fit(history_train[['sex_next']+numeric],history_train.finish_next)
        history_predictions[name]=model.predict(history_test[['sex_next']+numeric])
    same_location = np.ones(len(history_test),dtype=bool)
    for col in ['City','State']:
        same_location &= (history_test[col+'_prior'].fillna('').str.upper().str.strip()
                          ==history_test[col+'_next'].fillna('').str.upper().str.strip()).to_numpy()
    history_metrics=[]
    for sample,mask in [('All strict matches',np.ones(len(history_test),dtype=bool)),
                        ('Also same city/state',same_location)]:
        target=history_test.finish_next.to_numpy()[mask]
        for name,prediction in history_predictions.items():
            history_metrics.append({'sample':sample,'model':name,'n':int(mask.sum()),
                'mae':mean_absolute_error(target,prediction[mask]),'r2':r2_score(target,prediction[mask])})
    history_metrics=etable(pd.DataFrame(history_metrics),'history_prediction')
    display(history_metrics)
    print(f'{len(history_train):,} training transitions and {len(history_test):,} held-out transitions.')
    # Paired bootstrap over held-out matched runners; conditional on this one test event.
    gain = (np.abs(history_test.finish_next.to_numpy()-history_predictions['Demographics'])
            -np.abs(history_test.finish_next.to_numpy()-history_predictions['Repeat previous finish']))
    rng=np.random.default_rng(SEED)
    bootstrap_gain=np.array([rng.choice(gain,len(gain),replace=True).mean() for _ in range(2000)])
    history_gain_ci=np.quantile(bootstrap_gain,[.025,.975])
    print(f'Previous-time baseline reduces MAE by {gain.mean():.2f} min; paired bootstrap 95% CI '
          f'{history_gain_ci[0]:.2f}–{history_gain_ci[1]:.2f}. This CI does not generalize across race years.')
    ''')
    code('''
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    hs=history_metrics.loc[history_metrics['sample'].eq('All strict matches')]
    cs=checkpoint_metrics.loc[checkpoint_metrics['sample'].eq('All eligible 2017')]
    axes[0].barh(hs.model,hs.mae,color=['#e18a3b','#aab8c2','#167d9a'])
    axes[0].invert_yaxis();axes[0].set(xlabel='Finish-time MAE (minutes)',title='Before the start: previous race time helps')
    axes[1].barh(cs.model,cs.mae,color=['#aab8c2','#e18a3b','#aab8c2','#167d9a'])
    axes[1].invert_yaxis();axes[1].set(xlabel='Remaining-time MAE (minutes)',title='After 10 km: current pace helps')
    for ax,values in [(axes[0],hs.mae),(axes[1],cs.mae)]:
        for i,value in enumerate(values):ax.text(value+.3,i,f'{value:.1f}',va='center',fontsize=10)
        ax.set_xlim(0,max(values)+5)
    fig.tight_layout();efigure('15_information_prediction')
    ''')
    md('''
    These panels have different populations and different prediction times; their errors must not be
    compared as if they were one model competition. Within each panel the models share exactly the same
    test rows. History predicts a complete future finish; the checkpoint model predicts only what remains
    after 10 km. The simpler previous-finish baseline is retained even if it outperforms the calibrated model.
    ''')
    md('''
    # 13 · Does recent home weather predict less slowdown in Boston 2024?

    This directly addresses the classmate's acclimatization question, but only through a proxy.
    The source contains U.S. runners with ZIP codes and daily home-area temperatures. Home weather does not
    establish where, when, or whether someone trained outdoors. Travel, indoor training, and acclimatization
    remain unobserved.

    **Join audit:** preserve ZIP codes as five-character strings, padding lost leading zeros in both files.
    The weather file contains April 1–15. Exclude **April 15 (race day)**, leaving 14 strictly preceding days.
    Require complete daily coverage for a ZIP; use a many-to-one join and report unmatched runners.
    This avoids mixing race-day weather into the stated pre-race window. We also exclude all 22 records
    belonging to 11 duplicated bib numbers with conflicting split/finish values; the correct record cannot
    be identified from this source. A sensitivity check includes them without choosing a preferred value.

    Outcome: percentage second-half slowdown. Primary exploratory model: home mean temperature, age spline,
    recorded sex, and first-half-time spline. A 5°C increase is the reporting unit. Use standard errors
    clustered by the **first three ZIP digits** to acknowledge shared regional weather. This is a partial
    spatial adjustment, not a perfect geographic independence assumption. Models without half-time adjustment,
    ZIP-level clustering, and minimum-temperature exposure are sensitivity checks.
    ''')
    code('''
    home_athletes=pd.read_csv(ROOT/'data/raw/boston_2024/Athletes.csv',dtype={'Zip':'string'})
    home_weather=pd.read_csv(ROOT/'data/raw/boston_2024/Weather.csv',dtype={'Zip':'string'})
    raw_zip_matches=home_athletes.Zip.isin(set(home_weather.Zip)).sum()
    for frame in [home_athletes,home_weather]:
        frame['Zip']=frame.Zip.str.strip().str.zfill(5)
        assert frame.Zip.str.fullmatch(r'\\d{5}').all()
    home_weather['date']=pd.to_datetime(home_weather.Date,format='%m/%d/%y')
    assert not home_weather.duplicated(['Zip','date']).any()
    pre_weather=home_weather.loc[home_weather.date.between('2024-04-01','2024-04-14')]
    home_exposure=pre_weather.groupby('Zip').agg(days=('date','nunique'),home_f=('Mean Temp','mean'),
                                               home_min_f=('Min Temp','mean'),home_max_f=('Max Temp','mean')).reset_index()
    home_exposure=home_exposure.loc[home_exposure.days.eq(14)]
    home_joined=home_athletes.merge(home_exposure,on='Zip',how='left',validate='many_to_one',indicator=True)
    home_unmatched=int(home_joined['_merge'].ne('both').sum())
    home_ambiguous_bibs = home_athletes.loc[home_athletes.Bib.duplicated(keep=False),'Bib'].unique()
    home_ambiguous_rows = int(home_athletes.Bib.isin(home_ambiguous_bibs).sum())
    assert (home_athletes['First Half']+home_athletes['Second Half']).eq(home_athletes.Finish).all()
    home_with_conflicts=home_joined.loc[home_joined['_merge'].eq('both') & home_joined['First Half'].gt(0)
                         & home_joined['Second Half'].gt(0) & home_joined.Gender.isin(['M','F'])
                         & home_joined.Age.between(18,90)].copy()
    home = home_with_conflicts.loc[~home_with_conflicts.Bib.isin(home_ambiguous_bibs)].copy()
    assert home.Bib.is_unique
    home['home_c']=(home.home_f-32)/1.8
    home['home5']=home.home_c/5
    home['home_min5']=(home.home_min_f-32)/9
    home['half_min']=home['First Half']/60
    home['slow_pct']=100*(home['Second Half']/home['First Half']-1)
    home['zip3']=home.Zip.str[:3]
    assert np.allclose(home.slow_pct/100,home['Percent Change'])
    expansion_audit.append({'source':'Boston 2024','raw_rows':len(home_athletes),'eligible_rows':len(home),
                            'analysis_rows':len(home),'inference_unit_count':home.zip3.nunique(),
                            'note':f'{home_unmatched} unmatched ZIPs; {home_ambiguous_rows} conflicting-bib rows excluded; one event.'})
    print(f'Before ZIP normalization: {raw_zip_matches:,} matching records. After normalization: '
          f'{len(home_with_conflicts):,}; unmatched: {home_unmatched}; then exclude {home_ambiguous_rows} conflicting-bib rows.')
    print(f'{home.Zip.nunique():,} home ZIPs; {home.zip3.nunique()} ZIP-prefix clusters; 14 days per matched ZIP.')
    ''')
    code('''
    home_formula='slow_pct ~ home5 + bs(Age,df=4) + C(Gender) + bs(half_min,df=5)'
    home_fit=smf.ols(home_formula,home).fit(cov_type='cluster',cov_kwds={'groups':home.zip3},use_t=True)
    home_result=coefficient(home_fit,'home5','Home temperature: slowdown points per 5°C')
    home_sensitivity=[]
    for name,formula,group,term in [
        ('Unadjusted','slow_pct ~ home5','zip3','home5'),
        ('Age and sex only','slow_pct ~ home5 + bs(Age,df=4) + C(Gender)','zip3','home5'),
        ('Age, sex, first-half time',home_formula,'zip3','home5'),
        ('Cluster by individual ZIP',home_formula,'Zip','home5'),
        ('Mean daily minimum temperature',home_formula.replace('home5','home_min5'),'zip3','home_min5')
    ]:
        fit=smf.ols(formula,home).fit(cov_type='cluster',cov_kwds={'groups':home[group]},use_t=True)
        home_sensitivity.append(coefficient(fit,term,name))
    # Include ambiguous bib rows as a sensitivity check, never select one arbitrarily.
    hc = home_with_conflicts.copy()
    hc['home5']=(hc.home_f-32)/9
    hc['half_min']=hc['First Half']/60
    hc['slow_pct']=100*(hc['Second Half']/hc['First Half']-1)
    hc['zip3']=hc.Zip.str[:3]
    hc_fit=smf.ols(home_formula,hc).fit(cov_type='cluster',cov_kwds={'groups':hc.zip3},use_t=True)
    home_sensitivity.append(coefficient(hc_fit,'home5','Include conflicting bib records'))
    home_sensitivity=etable(pd.DataFrame(home_sensitivity),'home_weather_sensitivity')
    display(home_sensitivity)
    # Grouped descriptions display the actual support, not a fitted causal response.
    home['temp_bin']=pd.cut(home.home_c,[-10,0,5,10,15,20,25,35],right=False)
    home_bins=home.groupby('temp_bin',observed=True).agg(n=('slow_pct','size'),home_mean_c=('home_c','mean'),
                                                      mean_slow_pct=('slow_pct','mean'),median_slow_pct=('slow_pct','median')).reset_index()
    etable(home_bins,'home_weather_bins')
    display(home_bins)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    axes[0].bar(home_bins.temp_bin.astype(str),home_bins.n,color='#167d9a')
    axes[0].tick_params(axis='x',rotation=35)
    axes[0].set(xlabel='Mean home-area temperature, April 1–14 (°C)',ylabel='Finish records',title='Most runners have cooler home weather')
    axes[1].errorbar(home_sensitivity.estimate,np.arange(len(home_sensitivity)),
                     xerr=[home_sensitivity.estimate-home_sensitivity.ci_low,home_sensitivity.ci_high-home_sensitivity.estimate],
                     fmt='o',capsize=4,color='#167d9a')
    axes[1].set_yticks(range(len(home_sensitivity)),home_sensitivity.question)
    axes[1].axvline(0,color='#888888',linestyle='--');axes[1].invert_yaxis()
    axes[1].set(xlabel='Slowdown percentage points per 5°C, 95% CI',title='No clear evidence of a protective association')
    fig.tight_layout();efigure('16_home_weather')
    ''')
    md('''
    A null or specification-sensitive home-weather association does not disprove heat acclimatization.
    This proxy does not measure training exposure, the study includes only one race, and conditioning on
    first-half time changes the comparison. We can answer whether this measured proxy predicts the recorded
    slowdown, not whether acclimatization is physiologically effective.
    ''')
    md('''
    # 14 · What additional data did not earn a scientific claim?

    We inspected nine expansion candidates. NYC overlaps the fall sample and adds no weather in its current
    description; the 2024 all-marathon source adds recency but not the missing training exposure.
    The Run Club dataset explicitly states that it is **synthetic**, so it is excluded from real-world claims.

    The 87-row training pilot is real-world but not suitable for the proposed pre-race prediction:
    the uploader says mileage includes the marathon, one speed is implausibly large, cross-training is
    mostly missing, and finish-derived categories would leak the outcome. We do not silently repair units
    or treat those data as reliable training history. This records a limitation of the sources searched,
    not a claim that no suitable training dataset exists anywhere.
    ''')
    code('''
    pilot=pd.read_csv(ROOT/'data/raw/training_pilot/MarathonData.csv')
    pilot_audit=pd.DataFrame([
        {'check':'Rows','value':len(pilot)},
        {'check':'Missing cross-training entries','value':int(pilot.CrossTraining.isna().sum())},
        {'check':'Missing demographic category','value':int(pilot.Category.isna().sum())},
        {'check':'Speed entries above 40 (declared speed context)','value':int(pilot.sp4week.gt(40).sum())},
        {'check':'Maximum recorded speed field','value':float(pilot.sp4week.max())}
    ])
    etable(pilot_audit,'training_pilot_audit');display(pilot_audit)
    # This audit excludes names and other direct runner identifiers from exported results.
    audit_table=etable(pd.DataFrame(expansion_audit),'expansion_cleaning_audit')
    display(audit_table)
    # Four focal exploratory tests; sensitivities are not new independent confirmations.
    focal=pd.DataFrame([chi_results.iloc[0].to_dict(),chi_results.iloc[2].to_dict(),
                        chi_results.iloc[3].to_dict(),home_result])
    focal['q_bh']=multipletests(focal.p_value,method='fdr_bh')[1]
    etable(focal,'expansion_focal_tests');display(focal)
    ''')
    md('''
    ## Integrated interpretation and remaining work

    **Weather:** Chicago checks whether the Berlin heat association survives a different event. Humidity
    is now estimated from temperature and dew point; wind is observed, but the small annual sample and
    weather correlation limit the precision of separate effects. Do not call wide intervals evidence of no effect.

    **Race context:** demographic adjustment leaves substantial differences across fields. Course difficulty
    remains a possible explanation, alongside runner selection and unmeasured conditions; it is not identified.

    **Pacing and prediction:** Boston supplies observable within-race slowdown and prior-race history.
    Predictive comparisons must preserve when information becomes available and use the same test population.
    Repeated-race history is informative without being a measure of training frequency.

    **Acclimatization:** recent home weather can be examined as a proxy; actual training exposure cannot.
    A clean join, a truly pre-race window, spatial clustering, and sensitivity checks are essential.

    This remains observational analysis. No new dataset erases the need to handle selection, dependence,
    measurement definitions, source overlap, and missing variables. All focal extension tests are exploratory;
    BH-adjusted q-values are reported for the four focal weather/home-weather tests. Race rankings and
    checkpoint/model comparisons are descriptive, not additional significance tests.
    ''')
    code('''
    expanded_summary={
        'sources':source_catalog.to_dict(orient='records'),
        'audit':audit_table.to_dict(orient='records'),
        'chicago_results':chi_results.to_dict(orient='records'),
        'chicago_sensitivity':chi_sens.to_dict(orient='records'),
        'chicago_loo_min':float(chi_loo.slope_per_5c.min()),'chicago_loo_max':float(chi_loo.slope_per_5c.max()),
        'race_count':int(fall_analysis.Race.nunique()),'race_event_count':len(fall_annual),
        'race_raw_spread':raw_race_spread,'race_adjusted_spread':race_spread,
        'race_comparison':race_comparison.to_dict(orient='records'),
        'pacing_summary':pacing_summary.to_dict(orient='records'),
        'pacing_adjusted':pace_adjustments.to_dict(orient='records'),
        'checkpoint_metrics':checkpoint_metrics.to_dict(orient='records'),
        'history_metrics':history_metrics.to_dict(orient='records'),
        'history_train_n':len(history_train),'history_test_n':len(history_test),
        'history_mae_gain':float(gain.mean()),'history_gain_ci':history_gain_ci.tolist(),
        'home_result':home_result,'home_sensitivity':home_sensitivity.to_dict(orient='records'),
        'home_n':len(home),'home_zip_count':int(home.Zip.nunique()),'home_cluster_count':int(home.zip3.nunique()),
        'home_unmatched':home_unmatched,'home_ambiguous_rows':home_ambiguous_rows,
        'home_normalized_matches':len(home_with_conflicts),'home_raw_zip_matches':int(raw_zip_matches),
        'focal_tests':focal.to_dict(orient='records'),'pilot_audit':pilot_audit.to_dict(orient='records')
    }
    history_display = history_metrics.loc[history_metrics['sample'].eq('All strict matches')].set_index('model')
    checkpoint_display = checkpoint_metrics.loc[checkpoint_metrics['sample'].eq('All eligible 2017')].set_index('model')
    display(Markdown(f"""
    ### Expanded results in plain language

    - **Chicago replication:** {chi_results.iloc[0].estimate:.1f} extra minutes per 5°C
      (95% CI {chi_results.iloc[0].ci_low:.1f}–{chi_results.iloc[0].ci_high:.1f}) in the standardized annual mean.
    - **Humidity and wind:** estimated separately in Chicago, but intervals include zero and remain wide.
    - **Race context:** a {race_spread:.1f}-minute 90th–10th percentile spread remains between adjusted means
      across {fall_analysis.Race.nunique()} supported race fields; this is not an elevation effect estimate.
    - **Prior history:** held-out MAE is {history_display.loc['Demographics','mae']:.1f} minutes for demographics
      versus {history_display.loc['Repeat previous finish','mae']:.1f} for the previous-time baseline among matched repeaters.
    - **After 10 km:** remaining-time MAE falls from {checkpoint_display.loc['Demographics','mae']:.1f}
      to {checkpoint_display.loc['Demographics + first 10 km','mae']:.1f} minutes when observed pace is available.
    - **Home weather:** {home_result['estimate']:.2f} slowdown points per 5°C
      (95% CI {home_result['ci_low']:.2f}–{home_result['ci_high']:.2f}); no clear protective association from this proxy.
    """))
    (EXP/'results.json').write_text(json.dumps(expanded_summary,indent=2,default=float)+'\\n')
    print('Saved the expanded evidence bundle; Chapter 15 adds training and renders the integrated report.')
    ''')
