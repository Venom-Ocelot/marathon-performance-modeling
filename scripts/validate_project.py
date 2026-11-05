"""Validate executed deliverables and independently recompute key quantities."""
from pathlib import Path
import base64
import hashlib
import io
import json
from html.parser import HTMLParser
import nbformat
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
s = json.loads((OUT / 'results.json').read_text())
nb = nbformat.read(ROOT / 'marathon_analysis.ipynb', as_version=4)
nbformat.validate(nb)
code = [c for c in nb.cells if c.cell_type == 'code']
assert [c.execution_count for c in code] == list(range(1, len(code)+1))
assert not [o for c in code for o in c.outputs if o.output_type == 'error']
for c in nb.cells:
    assert not [x for x in c.source if ord(x) < 32 and x not in '\n\r\t']

manifest = json.loads((ROOT / 'data/raw/manifest.json').read_text())
for filename, expected in manifest['files'].items():
    assert hashlib.sha256((ROOT / 'data/raw/berlin' / filename).read_bytes()).hexdigest() == expected
raw = pd.read_csv(ROOT / 'data/raw/berlin/Berlin_Marathon_data_1974_2019.csv', dtype={'AGE':'string','COUNTRY':'string'})
age = pd.to_numeric(raw.AGE,errors='coerce')
time = pd.to_timedelta(raw.TIME,errors='coerce').dt.total_seconds()/60
mask = raw.YEAR.between(1999,2018) & age.isin([20,30,35,40,45,50,55,60,65]) & time.gt(0) & raw.GENDER.isin(['male','female'])
d = raw.loc[mask,['YEAR','GENDER']].copy()
d['age'] = age.loc[mask]
d['minutes'] = time.loc[mask]
assert len(d) == s['n_analysis'] == 629545
assert raw.loc[raw.YEAR.eq(2019),'GENDER'].eq('male').all()
assert np.isclose(d.minutes.mean(),s['mean_minutes'])

# Rebuild standardized means through an explicit merge, independently of the notebook pivot.
weights = d.groupby(['age','GENDER']).size().div(len(d)).rename('weight').reset_index()
means = d.groupby(['YEAR','age','GENDER']).minutes.mean().rename('group_mean').reset_index()
weighted = means.merge(weights,on=['age','GENDER'],validate='many_to_one')
assert weighted.groupby('YEAR')['weight'].sum().sub(1).abs().max() < 1e-12
std = weighted.assign(product=weighted.group_mean*weighted.weight).groupby('YEAR')['product'].sum()
annual = pd.read_csv(OUT / 'tables/annual_weather_analysis.csv').set_index('YEAR')
assert np.allclose(std,annual.standardized_mean)
X = np.column_stack([np.ones(len(annual)),annual.AVG_TEMP_C,annual.index.to_numpy()-2009])
coef = np.linalg.lstsq(X,annual.standardized_mean,rcond=None)[0]
assert np.isclose(coef[1],s['primary_tests'][2]['estimate'])

train,test = d[d.YEAR<=2013],d[d.YEAR>=2014]
assert len(train)==s['n_train'] and len(test)==s['n_test']
assert set(train.YEAR).isdisjoint(test.YEAR)
metric = pd.DataFrame(s['prediction_metrics']).set_index('model')
assert np.isclose(np.abs(test.minutes-train.minutes.mean()).mean(),metric.loc['Training mean','mae'])
group_means = train.groupby(['age','GENDER']).minutes.mean().rename('prediction').reset_index()
check = test.merge(group_means,on=['age','GENDER'],validate='many_to_one')
assert np.isclose(np.abs(check.minutes-check.prediction).mean(),metric.loc['Age + sex','mae'])

class ReportParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.images=[]; self.ids=set(); self.anchors=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        if tag=='img':
            assert a.get('alt');self.images.append(a['src'])
        if tag=='a' and a.get('href','').startswith('#'): self.anchors.append(a['href'][1:])
html = (ROOT / 'berlin_story.html').read_text()
parser=ReportParser();parser.feed(html)
assert all(a in parser.ids for a in parser.anchors)
assert len(parser.images)==5
for src in parser.images:
    assert src.startswith('data:image/png;base64,')
    Image.open(io.BytesIO(base64.b64decode(src.split(',',1)[1]))).verify()
assert '629,545' in html and '6.5 minutes' in html and '34.9 minutes' in html
assert 'increases average absolute error' in html
assert len(list((OUT/'figures').glob('*.png')))==10

# Compare exact local token bytes to authored artifacts without printing any credential.
token_file=Path.home()/'.kaggle/access_token'
if token_file.exists():
    token=token_file.read_bytes().strip()
    assert len(token)>8
    for folder in [ROOT,ROOT/'scripts',ROOT/'outputs']:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in {'.ipynb','.html','.py','.txt','.json'}:
                assert token not in p.read_bytes(), 'Credential unexpectedly found in an artifact.'
print(f'PASS: {len(code)} executed code cells; Berlin independent calculations; embedded baseline charts; credential absence.')

# Expanded-source hashes and integrated artifact integrity.
e=json.loads((OUT/'expanded/results.json').read_text())
for folder in ['chicago','fall_marathons','boston_2015_2017','boston_2024','training_pilot']:
    base=ROOT/'data/raw'/folder
    source=json.loads((base/'manifest.json').read_text())
    for filename,expected in source['files'].items():
        assert hashlib.sha256((base/filename).read_bytes()).hexdigest()==expected
integrated=(ROOT/'marathon_story.html').read_text()
parser=ReportParser();parser.feed(integrated)
assert len(parser.images)==7 and all(a in parser.ids for a in parser.anchors)
for src in parser.images:
    Image.open(io.BytesIO(base64.b64decode(src.split(',',1)[1]))).verify()
assert 'Six datasets analyzed' in integrated
for number in ['852,085','1,624,823','17,169','27.8','15.1','13.0']:
    assert number in integrated,number
assert len(list((OUT/'expanded/figures').glob('*.png')))==7

# Independently reconstruct Chicago standardization, using row-level weights and sums.
c=pd.read_csv(ROOT/'data/raw/chicago/Chicago Marathon Results.csv',usecols=['Year','Age','Gender','Finish'])
c=c.loc[c.Age.between(20,69)&c.Gender.isin(['M','F'])&c.Finish.gt(0)].copy()
c['decade']=c.Age//10*10;c['minutes']=c.Finish/60
reference=c[c.Year!=2007]
weights=reference.groupby(['decade','Gender']).size().div(len(reference)).rename('weight')
cell=c.groupby(['Year','decade','Gender']).minutes.agg(['sum','size']).join(weights)
chi_check=(cell['sum']/cell['size']*cell.weight).groupby('Year').sum()
chi_annual=pd.read_csv(OUT/'expanded/tables/chicago_annual.csv').set_index('Year')
assert np.allclose(chi_check,chi_annual.standardized_mean)
assert len(reference)==e['audit'][0]['analysis_rows']==852085
main=chi_annual.loc[chi_annual.index!=2007]
x=np.column_stack([np.ones(len(main)),main.temp_c,main.index.to_numpy()-2009])
slope=np.linalg.lstsq(x,main.standardized_mean,rcond=None)[0][1]
assert np.isclose(5*slope,e['chicago_results'][0]['estimate'])
assert main.rh_pct.between(0,100).all() and len(main)==26

# The Boston join and split identities are checked from raw files, not saved cohorts.
a=pd.read_csv(ROOT/'data/raw/boston_2024/Athletes.csv',dtype={'Zip':'string'})
w=pd.read_csv(ROOT/'data/raw/boston_2024/Weather.csv',dtype={'Zip':'string'})
for frame in [a,w]:frame['Zip']=frame.Zip.str.zfill(5)
w['date']=pd.to_datetime(w.Date,format='%m/%d/%y')
pre=w[w.date<'2024-04-15']
assert pre.date.nunique()==14 and pre.groupby('Zip').date.nunique().eq(14).all()
valid_zips=set(pre.Zip)
assert (~a.Zip.isin(valid_zips)).sum()==e['home_unmatched']==29
conflict=a.Bib.duplicated(keep=False)
assert conflict.sum()==e['home_ambiguous_rows']==22
home=a[a.Zip.isin(valid_zips)&~conflict]
assert len(home)==e['home_n']==17169 and home.Bib.is_unique
assert (home['First Half']+home['Second Half']).eq(home.Finish).all()
assert np.allclose(home['Second Half']/home['First Half']-1,home['Percent Change'])

# Independently reproduce the simple previous-time forecast on matched identities.
parts=[]
for year in [2015,2016,2017]:
    b=pd.read_csv(ROOT/f'data/raw/boston_2015_2017/marathon_results_{year}.csv',
                  usecols=['Name','M/F','Age','Country','Official Time','10K'])
    b['year']=year
    b['finish']=pd.to_timedelta(b['Official Time'],errors='coerce').dt.total_seconds()/60
    b['ten']=pd.to_timedelta(b['10K'],errors='coerce').dt.total_seconds()/60
    b['namekey']=b.Name.fillna('').str.normalize('NFKC').str.upper().str.replace(r'\s+',' ',regex=True).str.strip()
    b['birth_proxy']=year-b.Age
    parts.append(b)
b=pd.concat(parts,ignore_index=True)
keys=['namekey','M/F','Country','birth_proxy']
valid=b[b.finish.gt(0)&b.Age.between(18,90)&b['M/F'].isin(['M','F'])&b.Name.notna()&b.Country.notna()]
l=valid[valid.year==2016];r=valid[valid.year==2017]
l=l[~l.duplicated(keys,keep=False)];r=r[~r.duplicated(keys,keep=False)]
pairs=l.merge(r,on=keys,suffixes=('_prior','_next'),validate='one_to_one')
assert len(pairs)==e['history_test_n']==4922
history_baseline=next(x for x in e['history_metrics'] if x['sample']=='All strict matches' and x['model']=='Repeat previous finish')
assert np.isclose(np.abs(pairs.finish_next-pairs.finish_prior).mean(),history_baseline['mae'])
checkpoint=b[b.ten.gt(0)&b.finish.gt(b.ten)&b.Age.between(18,90)&b['M/F'].isin(['M','F'])]
tr=checkpoint[checkpoint.year<2017];te=checkpoint[checkpoint.year==2017]
mean_remaining=(tr.finish-tr.ten).mean()
check_mae=np.abs((te.finish-te.ten)-mean_remaining).mean()
cp_baseline=next(x for x in e['checkpoint_metrics'] if x['sample']=='All eligible 2017' and x['model']=='Training remaining-time mean')
assert len(te)==cp_baseline['n'] and np.isclose(check_mae,cp_baseline['mae'])

# Exported evidence should contain aggregates, not direct runner identifiers.
for pth in (OUT/'expanded/tables').glob('*.csv'):
    columns=pd.read_csv(pth,nrows=0).columns.str.lower().tolist()
    assert not set(columns)&{'name','zip','bib','runner_key','city','state'},pth.name
assert not e['home_result']['ci_low']>0 and not e['home_result']['ci_high']<0
selection=pd.read_csv(OUT/'expanded/tables/dataset_selection.csv')
assert len(selection)==9 and selection.decision.eq('Analyze').sum()==5
print('PASS: expansion source hashes; Chicago independent coefficients; Boston ZIP/conflict audit; previous-time/checkpoint baselines; aggregate-only exports; integrated report links and images.')

# Training source: verify exact source, strictly pre-race windows, no athlete overlap,
# and independently recompute the Boston-mean baseline from raw day records.
t=json.loads((OUT/'expanded/training_results.json').read_text())
tfolder=ROOT/'data/raw/training_logs'
for filename,expected in t['source']['files'].items():
    with (tfolder/filename).open('rb') as handle:
        assert hashlib.file_digest(handle,'sha256').hexdigest()==expected
windows=[];raw_count=0
for chunk in pd.read_csv(tfolder/'run_ww_2019_d.csv',usecols=['datetime','athlete','distance','duration','gender','age_group','major'],chunksize=250000):
    raw_count+=len(chunk)
    for race,date in [('BOSTON 2019','2019-04-15'),('CHICAGO 2019','2019-10-13')]:
        start=str((pd.Timestamp(date)-pd.Timedelta(days=56)).date())
        select=chunk.major.str.contains(race,regex=False,na=False)&chunk.datetime.between(start,date)
        v=chunk.loc[select].copy();v['race']=race;v['event_date']=date;windows.append(v)
logs=pd.concat(windows,ignore_index=True)
assert raw_count==t['raw_daily_rows']==13290380
assert not logs.duplicated(['athlete','race','datetime']).any()
pre=logs[logs.datetime<logs.event_date].copy()
pre['bad']=(pre.distance.isna()|pre.duration.isna()|pre.distance.lt(0)|pre.duration.lt(0)
            |(pre.distance.eq(0)^pre.duration.eq(0))
            |(pre.distance.gt(0)&~(pre.duration/pre.distance).between(2,20)))
assert len(pre)==56*5618 and pre.bad.sum()==t['flagged_days']
pre_group=pre.groupby(['athlete','race']).agg(days=('datetime','nunique'),km=('distance','sum'),bad=('bad','sum'))
pre_group=pre_group[(pre_group.days==56)&(pre_group.km>0)&(pre_group.bad==0)].reset_index()
day=logs[logs.datetime==logs.event_date]
day=day[day.distance.between(40,46)&day.duration.between(120,480)&day.gender.isin(['M','F'])&day.age_group.notna()]
cohort=day.merge(pre_group[['athlete','race']],on=['athlete','race'],validate='one_to_one')
assert len(cohort)==t['cohort_rows']==5247
tr=cohort[cohort.race=='BOSTON 2019'];te_all=cohort[cohort.race=='CHICAGO 2019']
te=te_all[~te_all.athlete.isin(set(tr.athlete))]
assert len(te_all)-len(te)==t['overlapping_test_athletes_removed']==55
assert set(tr.athlete).isdisjoint(te.athlete) and len(te)==1022
baseline=next(r for r in t['metrics'] if r['model']=='Boston training-sample mean')
assert np.isclose(np.abs(te.duration-tr.duration.mean()).mean(),baseline['mae'])
assert 'logged race-day' in integrated and 'certified finish' in integrated
print('PASS: real training-log source hash; raw day count; race-day exclusion; consistency filters; cohort counts; disjoint athlete test; independently recomputed proxy baseline.')
