"""Supplementary real training-log analysis with an explicitly imperfect race-day outcome."""
def add_training(md,code):
    md('''
    # 15 · Actual training logs: volume, frequency, and a carefully labeled outcome

    A further search found a sixth substantive Kaggle dataset:
    [Long-distance running dataset, v1](https://www.kaggle.com/datasets/mexwell/long-distance-running-dataset),
    a mirror of [Afonseca, Watanabe & Duarte's open research data](https://pmc.ncbi.nlm.nih.gov/articles/PMC8958962/).
    Unlike the 87-row pilot, this source contains anonymous athlete IDs and daily running totals.
    We download only the 2019 daily file. It contains **13,290,380 athlete-day rows**, including zero-activity
    days, for 36,412 athletes. These rows are not 13 million independent runs or race finishes.
    Age groups are coarse and were estimated by the original researchers; they are not exact ages.

    We select athletes labeled as Boston 2019 or Chicago 2019 participants and extract the 56 days before
    each event. Boston's date is [April 15, 2019, verified by the BAA](https://www.baa.org/news/baa-announces-2019-boston-marathon-registration-dates/);
    Chicago's October 13 date is in the downloaded Chicago weather file. **Race day is never part of a training feature.**

    The outcome needs care: the logs do not contain certified finish times. We use **logged race-day running
    duration**, restricting the day's distance to 40–46 km and duration to 120–480 minutes. Daily totals can
    include warm-up, cool-down, pauses or recording artifacts. This is a marathon-day proxy, not official
    finishing time. We do not normalize duration by GPS distance, because measurement error may inflate distance.

    Require all 56 day rows and positive total pre-race mileage. Flag a training window if any day has negative
    or missing distance/duration, inconsistent zero distance/time, or positive-distance pace outside 2–20 min/km.
    Exclude flagged windows rather than silently repair them. These are consistency rules, not proof every
    excluded activity is wrong. Zero logs are absence of recorded running, not proof of no exercise.
    ''')
    code('''
    training_dir=ROOT/'data/raw/training_logs'
    training_manifest=json.loads((training_dir/'manifest.json').read_text())
    for filename,expected in training_manifest['files'].items():
        with (training_dir/filename).open('rb') as handle:
            assert hashlib.file_digest(handle,'sha256').hexdigest()==expected
    race_dates={'BOSTON 2019':'2019-04-15','CHICAGO 2019':'2019-10-13'}
    selected_windows=[]
    training_raw_rows=0
    training_athletes=set()
    # Stream a 956 MB source rather than load all 13 million rows into memory.
    cols=['datetime','athlete','distance','duration','gender','age_group','major']
    for chunk in pd.read_csv(training_dir/'run_ww_2019_d.csv',usecols=cols,chunksize=250000):
        training_raw_rows+=len(chunk)
        training_athletes.update(chunk.athlete.unique())
        for race,date in race_dates.items():
            start=str((pd.Timestamp(date)-pd.Timedelta(days=56)).date())
            keep=chunk.major.str.contains(race,regex=False,na=False)&chunk.datetime.between(start,date)
            selected=chunk.loc[keep].copy()
            selected['race']=race
            selected['race_date']=date
            selected_windows.append(selected)
    logs=pd.concat(selected_windows,ignore_index=True)
    del selected_windows,chunk
    logs['date']=pd.to_datetime(logs.datetime)
    logs['race_date']=pd.to_datetime(logs.race_date)
    assert not logs.duplicated(['athlete','race','date']).any()
    pre_logs=logs.loc[logs.date.lt(logs.race_date)].copy()
    race_logs=logs.loc[logs.date.eq(logs.race_date)].copy()
    pre_logs['daily_pace']=pre_logs.duration/pre_logs.distance
    pre_logs['flagged']=(pre_logs.distance.isna()|pre_logs.duration.isna()|pre_logs.distance.lt(0)|pre_logs.duration.lt(0)
        |(pre_logs.distance.eq(0)^pre_logs.duration.eq(0))
        |(pre_logs.distance.gt(0)&~pre_logs.daily_pace.between(2,20)))
    pre_logs['week_before']=((pre_logs.race_date-pre_logs.date).dt.days-1)//7+1
    pre_logs['active']=pre_logs.distance.gt(0)
    print(f'{training_raw_rows:,} daily rows; {len(training_athletes):,} athletes; {len(logs):,} relevant window/day rows.')
    print(f'{pre_logs.flagged.sum()} flagged training days in '
          f'{pre_logs.loc[pre_logs.flagged].groupby(["athlete","race"]).ngroups} athlete/race windows.')
    ''')
    code('''
    training_windows=pre_logs.groupby(['athlete','race']).agg(
        observed_days=('date','nunique'),km=('distance','sum'),training_minutes=('duration','sum'),
        active_days=('active','sum'),flagged_days=('flagged','sum'),
        latest_training_date=('date','max')).reset_index()
    active_weeks=pre_logs.loc[pre_logs.active].groupby(['athlete','race']).week_before.nunique().rename('active_weeks')
    training_windows=training_windows.merge(active_weeks,on=['athlete','race'],how='left',validate='one_to_one')
    training_windows=training_windows.merge(race_logs[['athlete','race','distance','duration','gender','age_group','race_date']],
                                           on=['athlete','race'],validate='one_to_one')
    valid_training=(training_windows.observed_days.eq(56)&training_windows.flagged_days.eq(0)
                    &training_windows.km.gt(0)&training_windows.gender.isin(['M','F'])
                    &training_windows.age_group.notna())
    valid_outcome=training_windows.distance.between(40,46)&training_windows.duration.between(120,480)
    training_cohort=training_windows.loc[valid_training&valid_outcome].copy()
    training_cohort['weekly_km']=training_cohort.km/8
    training_cohort['km10']=training_cohort.weekly_km/10
    training_cohort['days_per_week']=training_cohort.active_days/8
    training_cohort['training_pace']=training_cohort.training_minutes/training_cohort.km
    training_cohort['race_day_minutes']=training_cohort.duration
    assert (training_cohort.latest_training_date<training_cohort.race_date).all()
    assert training_cohort.days_per_week.between(0,7).all()
    training_audit=training_windows.groupby('race').size().rename('labeled_participants').to_frame()
    training_audit['valid_pre_race_windows']=training_windows.loc[valid_training].groupby('race').size()
    training_audit['retained_outcome_proxies']=training_cohort.groupby('race').size()
    training_audit=etable(training_audit.reset_index(),'training_log_audit')
    training_summary=training_cohort.groupby('race').agg(n=('athlete','size'),
        median_weekly_km=('weekly_km','median'),median_days_per_week=('days_per_week','median'),
        median_training_pace=('training_pace','median'),median_race_day_minutes=('race_day_minutes','median')).reset_index()
    etable(training_summary,'training_log_summary')
    display(training_audit);display(training_summary)
    ''')
    md('''
    ## 15.1 · Associations within the selected log sample

    Estimate the association per 10 km/week using sex, coarse age group, and event adjustment, with
    uncertainty clustered by anonymous athlete ID (some ran both events). Add average training pace in a
    second specification to distinguish volume from demonstrated training speed; this does not isolate a
    causal mileage effect. Frequency and mileage are highly correlated, so a model that claims their separate
    “importance” from these records would be unstable. We display frequency descriptively and focus inference
    on the volume association. These supplemental comparisons are exploratory and separate from the four
    focal weather tests; no preregistered training hypothesis is claimed.
    ''')
    code('''
    training_base='race_day_minutes ~ km10 + C(gender) + C(age_group) + C(race)'
    training_effects=[]
    for label,data,formula in [
        ('Demographics and event adjustment',training_cohort,training_base),
        ('Also adjust training pace',training_cohort,training_base+' + training_pace'),
        ('At least six active weeks',training_cohort.loc[training_cohort.active_weeks.ge(6)],training_base+' + training_pace'),
        ('Narrower 40–44 km race-day gate',training_cohort.loc[training_cohort.distance.le(44)],training_base+' + training_pace')
    ]:
        fit=smf.ols(formula,data).fit(cov_type='cluster',cov_kwds={'groups':data.athlete},use_t=True)
        item=coefficient(fit,'km10',label)
        item['n']=len(data)
        training_effects.append(item)
    training_effects=etable(pd.DataFrame(training_effects),'training_volume_effects')
    display(training_effects)
    training_correlations=training_cohort[['weekly_km','days_per_week','training_pace','race_day_minutes']].corr()
    etable(training_correlations.reset_index(names='variable'),'training_correlations')
    print('Correlation of mileage and running-day frequency:',training_correlations.loc['weekly_km','days_per_week'])
    ''')
    md('''
    ## 15.2 · Transfer a training-based prediction to a different event

    Fit on Boston 2019, evaluate on Chicago 2019, and remove from the Chicago test set **every athlete ID
    used in Boston training**. This tests both a new event and new identified athletes. Demographics alone,
    volume/frequency, and volume/frequency/pace models are compared on the same test rows. No race-day distance
    or duration is an input. Recorded race-day distance is used only to define the outcome-quality cohort.

    This is a supplementary forecast of a **logged duration proxy**, not a new claim of accurately predicting
    certified finish times. GPS/recording differences, course differences, and participant selection can affect
    transfer. We evaluate all specified models, without selecting parameters on Chicago outcomes.
    ''')
    code('''
    log_train=training_cohort.loc[training_cohort.race.eq('BOSTON 2019')].copy()
    log_test_all=training_cohort.loc[training_cohort.race.eq('CHICAGO 2019')].copy()
    log_test=log_test_all.loc[~log_test_all.athlete.isin(set(log_train.athlete))].copy()
    assert set(log_train.athlete).isdisjoint(log_test.athlete)
    training_predictions={'Boston training-sample mean':np.repeat(log_train.race_day_minutes.mean(),len(log_test))}
    for label,numerics in [('Demographics',[]),('Demographics + volume/frequency',['weekly_km','days_per_week']),
                           ('Demographics + volume/frequency/pace',['weekly_km','days_per_week','training_pace'])]:
        # Linear model: simple enough to inspect; no test-set tuning.
        pre=ColumnTransformer([('cat',OneHotEncoder(drop='first',handle_unknown='error'),['gender','age_group']),
                               ('num','passthrough',numerics)])
        model=make_pipeline(pre,LinearRegression())
        columns=['gender','age_group']+numerics
        model.fit(log_train[columns],log_train.race_day_minutes)
        training_predictions[label]=model.predict(log_test[columns])
    training_metrics=pd.DataFrame([{'model':label,'n_train':len(log_train),'n_test':len(log_test),
        'mae':mean_absolute_error(log_test.race_day_minutes,pred),
        'rmse':np.sqrt(mean_squared_error(log_test.race_day_minutes,pred)),
        'r2':r2_score(log_test.race_day_minutes,pred)} for label,pred in training_predictions.items()])
    etable(training_metrics,'training_prediction');display(training_metrics)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for race,color in [('BOSTON 2019','#167d9a'),('CHICAGO 2019','#e18a3b')]:
        subset=training_cohort.loc[training_cohort.race.eq(race)].copy()
        subset['volume_bin']=pd.cut(subset.weekly_km,[0,20,40,60,80,100,200],right=False)
        bins=subset.groupby('volume_bin',observed=True).agg(km=('weekly_km','median'),minutes=('race_day_minutes','median'),n=('athlete','size'))
        axes[0].plot(bins.km,bins.minutes,'o-',label=race,color=color)
    axes[0].set(xlabel='Logged running volume (km/week), bin medians',ylabel='Median logged race-day duration (minutes)',
                title='More logged training accompanies faster race-day logs')
    axes[0].legend(frameon=False,fontsize=9)
    axes[1].barh(training_metrics.model,training_metrics.mae,color=['#aab8c2','#aab8c2','#e18a3b','#167d9a'])
    axes[1].invert_yaxis()
    axes[1].set(xlabel='Chicago held-out proxy MAE (minutes)',title='Training adds information across events')
    for i,value in enumerate(training_metrics.mae):axes[1].text(value+.4,i,f'{value:.1f}',va='center')
    axes[1].set_xlim(0,training_metrics.mae.max()+7)
    fig.tight_layout();efigure('17_training_logs')
    ''')
    md('''
    This source changes the answer to the classmates' training question: actual logged volume and running-day
    frequency can be studied. It still does not establish what a prescribed mileage increase would cause.
    Faster/healthier runners may train more, and unlogged activity remains unknown. The source was selected
    from social-network users with major-marathon participation; age groups are coarse estimates.
    The race-day outcome is a daily-log proxy, so the results should be treated as supplementary evidence
    until matched to official finish records. We preserve that distinction in the report and do not mix
    these prediction errors with the official-result models.
    ''')
    code('''
    training_results={
        'source':training_manifest,'raw_daily_rows':training_raw_rows,'athletes':len(training_athletes),
        'window_rows':len(logs),'flagged_days':int(pre_logs.flagged.sum()),
        'cohort_rows':len(training_cohort),'cohort_athletes':int(training_cohort.athlete.nunique()),
        'audit':training_audit.to_dict(orient='records'),'summary':training_summary.to_dict(orient='records'),
        'effects':training_effects.to_dict(orient='records'),'metrics':training_metrics.to_dict(orient='records'),
        'overlapping_test_athletes_removed':len(log_test_all)-len(log_test),
        'volume_frequency_correlation':float(training_correlations.loc['weekly_km','days_per_week'])
    }
    (EXP/'training_results.json').write_text(json.dumps(training_results,indent=2,default=float)+'\\n')
    runpy.run_path(str(ROOT/'scripts/render_report.py'),run_name='__main__')
    print('Completed six substantive datasets, the training-pilot audit, and the integrated report.')
    ''')
