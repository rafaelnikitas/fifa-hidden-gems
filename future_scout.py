from role_trends import *
from strong_features import strong_history_panel
from catboost import CatBoostClassifier,Pool
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
import time,hashlib,importlib.metadata

YEARS=[2020,2021,2022,2023]
BASE='Player model';STATIC_MODEL='Player + current role';LINEAR='Player + linear trend';NEW='Player + learned future';FIFA='FIFA potential';CAL='Learned FIFA control'
METHODS=[FIFA,CAL,BASE,STATIC_MODEL,LINEAR,NEW]
CB_PARAMS=dict(iterations=450,depth=5,learning_rate=.045,l2_leaf_reg=12,border_count=64,
    loss_function='Logloss',random_seed=42,thread_count=4,verbose=False,allow_writing_files=False)

def model_frame(f,cols):return f[cols].astype('float32').fillna(-999.)

def make_panel(raw,role):
    clean,original,metadata_audit=canonical_ids(raw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
        panel,sets,_=strong_history_panel(clean)
    lookup=raw.set_index('row_id')
    for c in ['short_name','club_name','league_name','nationality_name']:
        panel[c]=panel.row_id.map(lookup[c])
    feat=role['features'].copy();feat.index=raw.row_id
    for c in STATIC+DYNAMIC+LEARNED:panel[c]=panel.row_id.map(feat[c])
    for name,positive in [('quality80',panel.observed_peak3.ge(80)),('jump',panel.delta3_observed.ge(11)),('star85',panel.observed_peak3.ge(85))]:
        panel['event_'+name]=np.where(positive,1.,np.where(panel.complete3,0.,np.nan))
    cols=sets['Pattern + FIFA']
    variants={BASE:cols,STATIC_MODEL:cols+STATIC,LINEAR:cols+STATIC+DYNAMIC,NEW:cols+STATIC+LEARNED}
    assert len(cols)==431
    return panel,variants,metadata_audit

def dated_weights(train):
    counts=train.groupby('player_id').size()
    w=np.where(train.overall.lt(70),3.,1.)/train.player_id.map(counts).to_numpy()
    y=train.event_quality80.to_numpy()
    extra=1+np.clip((train.delta3_observed.to_numpy()-11)/10,0,2)
    w*=np.where(y==1,4*extra,1.)
    return w/np.mean(w)

def ordered(f,method):
    if method==FIFA:return f.sort_values([method,'overall','age','player_id'],ascending=[False,False,True,True],kind='mergesort')
    return f.sort_values([method,'player_id'],ascending=[False,True],kind='mergesort')

def measure(f,method,k=50,target='quality80'):
    r=ordered(f,method);g=r.head(min(k,len(r)));event='event_'+target
    found=int(g[event].eq(1).sum());unknown=int(g[event].isna().sum());total=int(r[event].eq(1).sum())
    return {'year':int(f.edition.iloc[0]),'method':method,'target':target,'k':k,'eligible':len(r),
            'confirmed_total':total,'found':found,'selected':len(g),'unresolved':unknown,
            'precision_lower':found/max(len(g),1),'precision_upper':(found+unknown)/max(len(g),1),
            'confirmed_recall':found/total if total else np.nan}

def fit_models(panel,variants,years=YEARS,reuse=False):
    outputs=[];audits=[];bundles={};tick=time.monotonic()
    for year in years:
        cache=OUT/f'fit_{year}.joblib'
        saved=None
        if reuse and cache.exists():
            out,bb,aa=joblib.load(cache)
            if 'Player + future role' in out:
                out=out.rename(columns={'Player + future role':LINEAR});bb[LINEAR]=bb.pop('Player + future role')
                for item in aa:
                    if item['method']=='Player + future role':item['method']=LINEAR
            if all(name in bb for name in variants):
                if 'peak_seen' not in out:out['peak_seen']=out.row_id.map(panel.set_index('row_id').peak_seen)
                outputs.append(out);audits+=aa;bundles[year]=bb;continue
            saved=(out,bb,aa)
        train=panel.loc[panel.fit_domain&panel.outcome_end.le(year)&panel.event_quality80.notna()].copy()
        test=panel.loc[panel.eligible&panel.edition.eq(year)&panel.potential.notna()].copy()
        assert len(test)>0 and train.outcome_end.max()<=year
        w=dated_weights(train);y=train.event_quality80.astype(int)
        meta=['row_id','player_id','edition','short_name','age','overall','potential','position_group','primary_position','nationality_name',
              'club_name','league_name','club_team_id','league_id','peak_seen','complete3','observed_peak3','delta3_observed','event_quality80','event_jump','event_star85']+STATIC+DYNAMIC+LEARNED
        out=test[meta].copy();out[FIFA]=out.potential;bb={};aa=[]
        if saved:
            old,bb,aa=saved
            assert np.array_equal(old.row_id,out.row_id)
            for name in list(bb)+[CAL]:
                if name in old:out[name]=old[name].to_numpy()
        baseline_features=['overall','age','potential','potential_gap']+['role_'+r for r in ROLE_CORES]
        control=LGBMClassifier(n_estimators=220,learning_rate=.04,num_leaves=7,max_depth=3,min_child_samples=100,
            reg_lambda=20,reg_alpha=1,n_jobs=4,verbosity=-1,random_state=42,deterministic=True,force_col_wise=True)
        if CAL not in out:
            control.fit(model_frame(train,baseline_features),y,sample_weight=w)
            out[CAL]=control.predict(model_frame(test,baseline_features),raw_score=True)
            bb[CAL]={'model':control,'columns':baseline_features}
        for name,cols in variants.items():
            if name in bb:continue
            model=CatBoostClassifier(**CB_PARAMS)
            model.fit(model_frame(train,cols),y,sample_weight=w)
            out[name]=model.predict(model_frame(test,cols),prediction_type='RawFormulaVal')
            bb[name]={'model':model,'columns':cols}
            aa.append({'year':year,'method':name,'features':len(cols),'training_rows':len(train),'positives':int(y.sum()),
                       'first_origin':int(train.edition.min()),'last_origin':int(train.edition.max()),'last_label':int(train.outcome_end.max())})
        joblib.dump((out,bb,aa),cache,compress=3);out.to_csv(OUT/f'forecasts_{year}.csv',index=False)
        outputs.append(out);audits+=aa;bundles[year]=bb
        print(f'{year}: {len(train):,} dated profiles, {int(y.sum()):,} positives; {time.monotonic()-tick:.1f}s',flush=True)
    forecasts=pd.concat(outputs,ignore_index=True);audit=pd.DataFrame(audits)
    forecasts.to_csv(OUT/'all_rankings.csv',index=False);audit.to_csv(OUT/'training_audit.csv',index=False)
    metrics=pd.DataFrame([measure(f,m,k,t) for year,f in forecasts.groupby('edition') if year in YEARS
                         for m in METHODS for k in [10,25,50,100,200,500] for t in ['quality80','jump','star85']])
    metrics.to_csv(OUT/'performance.csv',index=False)
    return forecasts,metrics,bundles,audit

def weighted_hit(order,event,weight,k):
    w=weight[order];before=np.cumsum(w)-w
    included=np.minimum(w,np.maximum(0,k-before))
    return float((included*event[order]).sum())

def uncertainty(forecasts,draws=500):
    f=forecasts.loc[forecasts.edition.isin(YEARS)].copy();ids,iv=np.unique(f.player_id,return_inverse=True)
    rng=np.random.default_rng(301);index=f.index.to_numpy();local={v:i for i,v in enumerate(index)}
    arrays={}
    for year,g in f.groupby('edition'):
        loc=np.array([local[i] for i in g.index]);orders={m:np.argsort(-g[m].to_numpy(),kind='stable') for m in METHODS}
        # Match the exact deterministic tie policy used by the original rankings.
        orders={m:np.array([g.index.get_loc(i) for i in ordered(g,m).index]) for m in METHODS}
        event=g.event_quality80.eq(1).to_numpy(float);arrays[year]=(loc,orders,event)
    pairs=[(NEW,BASE),(NEW,STATIC_MODEL),(NEW,FIFA),(NEW,CAL),(BASE,FIFA)]
    observed={(year,a,b):measure(g,a)['found']-measure(g,b)['found'] for year,g in f.groupby('edition') for a,b in pairs}
    draws_by={key:[] for key in observed}
    for _ in range(draws):
        weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))[iv]
        for year,(loc,orders,event) in arrays.items():
            hits={m:weighted_hit(o,event,weights[loc],50) for m,o in orders.items()}
            for a,b in pairs:draws_by[(year,a,b)].append(hits[a]-hits[b])
    output=[]
    for a,b in pairs:
        total=np.sum([draws_by[(year,a,b)] for year in YEARS],axis=0)
        for year in YEARS:
            arr=draws_by[(year,a,b)];output.append({'year':year,'model':a,'baseline':b,'extra_hits':observed[(year,a,b)],
                'low':np.quantile(arr,.025),'high':np.quantile(arr,.975),'draws':draws,'scope':'Fixed models; player IDs resampled jointly across four editions; full rankings reranked at 50.'})
        output.append({'year':'Pooled','model':a,'baseline':b,'extra_hits':sum(observed[(year,a,b)] for year in YEARS),
                       'low':np.quantile(total,.025),'high':np.quantile(total,.975),'draws':draws,
                       'scope':'Conditional on these four cohorts, fitted models and observed records; not uncertainty over future football eras.'})
    out=pd.DataFrame(output);out.to_csv(OUT/'paired_player_bootstrap.csv',index=False);return out

def named_cases(panel,forecasts,role,bundles):
    f=forecasts.loc[forecasts.edition.eq(2023)].copy()
    for m,c in [(BASE,'ml_rank'),(NEW,'future_rank'),(FIFA,'fifa_rank')]:
        r=ordered(f,m);f[c]=r.index.to_series().map(pd.Series(np.arange(1,len(r)+1),index=r.index)).reindex(f.index)
    # Case choice is descriptive after evaluating every player. Include a loss or false alarm.
    confirmed=f.loc[f.event_quality80.eq(1)].copy();added=confirmed.loc[confirmed.ml_rank.le(50)&confirmed.fifa_rank.gt(50)].sort_values('ml_rank')
    if len(added)<2:added=pd.concat([added,confirmed.sort_values('ml_rank')]).drop_duplicates('player_id')
    pieces=[]
    if len(added):pieces.append(added.head(2).assign(case='Confirmed 80+'))
    false=f.loc[f.complete3&f.event_quality80.eq(0)].sort_values('ml_rank').head(1).assign(case='False alarm')
    if len(false):pieces.append(false)
    missed=confirmed.loc[confirmed.ml_rank.gt(50)].sort_values('fifa_rank').head(1).assign(case='Missed 80+')
    if len(missed):pieces.append(missed)
    cases=pd.concat(pieces).drop_duplicates('player_id').head(4)
    model=bundles[2023][BASE];profile=panel.set_index('row_id').reindex(cases.row_id)
    sv=model['model'].get_feature_importance(Pool(model_frame(profile,model['columns'])),type='ShapValues')
    raw=model['model'].predict(model_frame(profile,model['columns']),prediction_type='RawFormulaVal')
    np.testing.assert_allclose(sv[:,:-1].sum(axis=1)+sv[:,-1],raw,atol=1e-6)
    contrib=[]
    for i,(_,r) in enumerate(cases.iterrows()):
        for c,value in zip(model['columns'],sv[i,:-1]):
            group=feature_group(c)
            contrib.append({'player_id':int(r.player_id),'short_name':r.short_name,'feature':c,'shap':float(value),'group':group})
    neighbors=[];z=role['z'];raw_index=panel.set_index('row_id')
    meta=forecasts.loc[forecasts.edition.eq(2023)]
    for _,r in cases.iterrows():
        pool=panel.loc[panel.edition.eq(2023)&panel.position_group.eq(r.position_group)&panel.player_id.ne(r.player_id)&panel.overall.ge(75)]
        if r.position_group=='GK':
            gcols=[c for c in SKILLS if c.startswith('goalkeeping_')]
            seed=panel.loc[panel.edition.eq(2015)&panel.position_group.eq('GK'),gcols]
            scale=(seed.sub(seed.mean(axis=1),axis=0)).std().clip(lower=3).to_numpy()
            values=pool[gcols].to_numpy();values=(values-values.mean(axis=1,keepdims=True))/scale
            focal=panel.loc[panel.row_id.eq(r.row_id),gcols].to_numpy()[0];focal=(focal-focal.mean())/scale
            ds=np.sqrt(np.mean((values-focal)**2,axis=1));dimensions=5
        else:
            ds=np.sqrt(np.mean((z[pool.row_id.to_numpy()]-z[int(r.row_id)])**2,axis=1));dimensions=28
        for j in np.argsort(ds)[:4]:
            n=pool.iloc[j];neighbors.append({'player_id':int(r.player_id),'short_name':r.short_name,'neighbor_id':int(n.player_id),'neighbor':n.short_name,
                                           'neighbor_overall':int(n.overall),'distance':float(ds[j]),'edition':2023,'role':r.position_group,'dimensions':dimensions})
    # Global explanation uses a representative dated sample, not only successes.
    subset=panel.loc[panel.edition.eq(2023)&panel.eligible].sample(min(1200,int((panel.edition.eq(2023)&panel.eligible).sum())),random_state=SEED)
    allsv=model['model'].get_feature_importance(Pool(model_frame(subset,model['columns'])),type='ShapValues')[:,:-1]
    importance=pd.DataFrame({'feature':model['columns'],'mean_abs_shap':np.abs(allsv).mean(axis=0),'method':BASE})
    other=bundles[2023][NEW]
    newshap=other['model'].get_feature_importance(Pool(model_frame(subset,other['columns'])),type='ShapValues')[:,:-1]
    importance=pd.concat([importance,pd.DataFrame({'feature':other['columns'],'mean_abs_shap':np.abs(newshap).mean(axis=0),'method':NEW})],ignore_index=True)
    cases.to_csv(OUT/'player_cases.csv',index=False);pd.DataFrame(contrib).to_csv(OUT/'player_shap.csv',index=False)
    pd.DataFrame(neighbors).to_csv(OUT/'player_neighbors.csv',index=False);importance.to_csv(OUT/'shap_importance.csv',index=False)
    return cases,pd.DataFrame(contrib),pd.DataFrame(neighbors),importance

def feature_group(c):
    if c in LEARNED:return 'Future role'
    if c in STATIC:return 'Current role'
    if c in ['potential','potential_gap']:return 'FIFA potential'
    if any(x in c for x in ['lag','change','past_','history','slope','acceleration','growth_consistency','recent_peak','recent_floor']):return 'Past development'
    if c.startswith(('team_','league_','club_','position_','rating_vs_','age_vs_')):return 'Team / peer context'
    if c in ['age','height_cm','weight_kg','age_to_21','age_rating_interaction','first_observed_age','editions_since_first_seen']:return 'Age / physique'
    return 'Current attributes'

def execute_future_study(reuse=True):
    source=OUT/'normalized_ratings.joblib'
    if source.exists():raw=joblib.load(source);source_audit=pd.DataFrame()
    else:
        raw,source_audit,paths=load_ratings();joblib.dump(raw,source,compress=3)
    cache=OUT/'prepared_role_panel.joblib'
    if reuse and cache.exists():
        raw,role,panel,variants,metadata_audit=joblib.load(cache)
        if 'learned_audit' not in role:
            learned,la,ls,lc=learn_profile_forecasts(raw,role['z'],role['profiles'])
            role['features']=pd.concat([role['features'],learned],axis=1)
            role.update(learned_audit=la,learned_forecasts=ls,forecast_coefficients=lc)
            for c in LEARNED:panel[c]=panel.row_id.map(learned.set_axis(raw.row_id)[c])
            variants={BASE:variants[BASE],STATIC_MODEL:variants[STATIC_MODEL],LINEAR:variants['Player + future role'],NEW:variants[BASE]+STATIC+LEARNED}
            joblib.dump((raw,role,panel,variants,metadata_audit),cache,compress=3)
    else:
        role=build_role_panel(raw);panel,variants,metadata_audit=make_panel(raw,role)
        joblib.dump((raw,role,panel,variants,metadata_audit),cache,compress=3)
    validation=validate_time(raw,role)
    role['profile_history'].to_csv(OUT/'role_profiles.csv',index=False);role['trend_audit'].to_csv(OUT/'trend_time_audit.csv',index=False)
    role['profile_forecasts'].to_csv(OUT/'role_forecast_accuracy.csv',index=False);metadata_audit.to_csv(OUT/'league_name_audit.csv',index=False)
    role['learned_audit'].to_csv(OUT/'learned_profile_time_audit.csv',index=False);role['learned_forecasts'].to_csv(OUT/'learned_profile_accuracy.csv',index=False)
    role['forecast_coefficients'].to_csv(OUT/'learned_forecast_coefficients.csv',index=False)
    forecasts,metrics,bundles,audit=fit_models(panel,variants,YEARS+[2026],reuse=reuse)
    assert audit.last_label.le(audit.year).all() and audit.last_origin.le(audit.year-3).all()
    assert forecasts.overall.lt(70).all() and not forecasts.duplicated(['edition','player_id']).any()
    if reuse:
        indexed=panel.set_index('row_id')
        for year,g in forecasts.groupby('edition'):
            sample=g.head(8);profiles=indexed.reindex(sample.row_id)
            for name in [BASE,STATIC_MODEL,LINEAR,NEW]:
                bundle=bundles[year][name]
                score=bundle['model'].predict(model_frame(profiles,bundle['columns']),prediction_type='RawFormulaVal')
                np.testing.assert_allclose(score,sample[name],atol=1e-6,rtol=1e-6)
    intervals=uncertainty(forecasts)
    if reuse and (OUT/'profile_shift_uncertainty.csv').exists():shifts=pd.read_csv(OUT/'profile_shift_uncertainty.csv')
    else:shifts=profile_shift_bootstrap(role);shifts.to_csv(OUT/'profile_shift_uncertainty.csv',index=False)
    cases,contributions,neighbors,importance=named_cases(panel,forecasts,role,bundles)
    primary=metrics.loc[metrics.k.eq(50)&metrics.target.eq('quality80')]
    totals=primary.groupby('method').found.sum().astype(int).to_dict()
    mature=forecasts.loc[forecasts.edition.isin(YEARS)]
    unique={m:mature.loc[np.concatenate([ordered(f,m).head(50).index.to_numpy() for _,f in mature.groupby('edition')])].loc[lambda x:x.event_quality80.eq(1),'player_id'].nunique() for m in METHODS}
    forecast_check=role['learned_forecasts'].loc[role['learned_forecasts'].origin.isin(YEARS)]
    future_gain=totals[NEW]-totals[STATIC_MODEL]
    rare=metrics.loc[metrics.k.eq(50)&metrics.target.eq('star85')].groupby('method').found.sum().astype(int).to_dict()
    strict=pd.DataFrame([measure(f,m) for y,f in forecasts.loc[forecasts.edition.isin(YEARS)&forecasts.peak_seen.lt(70)].groupby('edition') for m in METHODS])
    strict.to_csv(OUT/'never_previously70_performance.csv',index=False)
    summary={'hits_at50':totals,'star85_hits':rare,'never70_at50':strict.groupby('method').found.sum().astype(int).to_dict(),
        'confirmed80_events':int(primary.groupby('year').confirmed_total.first().sum()),
        'unique80_players':{k:int(v) for k,v in unique.items()},'years_future_beats_static':int((primary.pivot(index='year',columns='method',values='found')[NEW]>primary.pivot(index='year',columns='method',values='found')[STATIC_MODEL]).sum()),
        'extra_future_vs_static':int(future_gain),'profile_forecast_wins':int(forecast_check.improvement.gt(0).sum()),'profile_forecast_comparisons':len(forecast_check),
        'feature_counts':{k:len(v) for k,v in variants.items()},'geometry_variance':role['normalizer']['explained_variance'],
        'validation':{**validation,'dated_training_labels':True,'native_shap_additivity':True},
        'retrospective_scope':'Previously inspected ratings archive. These are dated replays, not independent prospective validation.',
        'packages':{m:importlib.metadata.version(m) for m in ['numpy','pandas','scikit-learn','lightgbm','catboost','plotly','matplotlib']}}
    (OUT/'findings.json').write_text(json.dumps(summary,indent=2))
    (OUT/'experiment_protocol.json').write_text(json.dumps({'primary_outcome':'Reaches 80+ within the next three editions from current <70; any age; no prior 85+',
        'primary_budget':50,'primary_feature_test':[NEW,STATIC_MODEL],'secondary_test':[NEW,BASE],
        'features':variants,'parameters':CB_PARAMS,'trend_window':4,'trend_shrinkage':.5,'historical_geometry':2015,
        'missing_followup':'Confirmed positive when observed; negative requires all three records; remaining outcomes unresolved.'},indent=2))
    ordered(forecasts.loc[forecasts.edition.eq(2026)],NEW).head(100).to_csv(OUT/'fc26_snapshot_watchlist.csv',index=False)
    return dict(raw=raw,role=role,panel=panel,variants=variants,forecasts=forecasts,metrics=metrics,bundles=bundles,
                audit=audit,intervals=intervals,shifts=shifts,cases=cases,contributions=contributions,neighbors=neighbors,importance=importance,summary=summary)

def explain_player(study,player_id,edition=2023,method=BASE):
    row=study['panel'].loc[lambda p:p.player_id.eq(player_id)&p.edition.eq(edition)]
    if len(row)!=1:raise ValueError('Choose one recorded player and edition.')
    bundle=study['bundles'][edition][method];x=model_frame(row,bundle['columns'])
    values=bundle['model'].get_feature_importance(Pool(x),type='ShapValues')[0]
    score=bundle['model'].predict(x,prediction_type='RawFormulaVal')[0]
    np.testing.assert_allclose(values.sum(),score,atol=1e-6)
    out=pd.DataFrame({'feature':bundle['columns'],'value':row.iloc[0][bundle['columns']].to_numpy(),'shap':values[:-1]})
    out.attrs.update(expected=float(values[-1]),score=float(score),player_id=int(player_id),edition=int(edition),method=method)
    return out.loc[out.shap.abs().sort_values(ascending=False).index]

if __name__=='__main__':
    result=execute_future_study(reuse=os.environ.get('FUTURE_REUSE','1')=='1')
    print(json.dumps(result['summary'],indent=2))
