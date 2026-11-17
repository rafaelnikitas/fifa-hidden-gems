from future_scout import *
import re

ABLATIONS=['FIFA inputs only','Without team / peer context','Without player history']

def feature_removal_intervals(forecasts,draws=500):
    ids,iv=np.unique(forecasts.player_id,return_inverse=True)
    rng=np.random.default_rng(904);arrays={};methods=[BASE]+ABLATIONS
    for year,g in forecasts.groupby('edition'):
        loc=forecasts.index.get_indexer(g.index)
        orders={m:np.array([g.index.get_loc(i) for i in ordered(g,m).index]) for m in methods}
        arrays[year]=(loc,orders,g.event_quality80.eq(1).to_numpy(float))
    samples={m:[] for m in ABLATIONS}
    for _ in range(draws):
        weights=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))[iv]
        delta={m:0. for m in ABLATIONS}
        for loc,orders,event in arrays.values():
            hits={m:weighted_hit(o,event,weights[loc],50) for m,o in orders.items()}
            for m in ABLATIONS:delta[m]+=hits[BASE]-hits[m]
        for m in ABLATIONS:samples[m].append(delta[m])
    rows=[]
    for m in ABLATIONS:
        effect=sum(measure(g,BASE)['found']-measure(g,m)['found'] for _,g in forecasts.groupby('edition'))
        rows.append({'model':BASE,'baseline':m,'extra_hits':effect,'low':np.quantile(samples[m],.025),'high':np.quantile(samples[m],.975),'draws':draws,'scope':'Conditional on fitted models and these historical cohorts; player IDs resampled jointly across editions.'})
    result=pd.DataFrame(rows);result.to_csv(OUT/'feature_removal_intervals.csv',index=False)
    return result

def scouting_feature_group(c):
    if c in ['potential','potential_gap']:
        return 'FIFA potential'
    if c.startswith(('team_','league_','club_','position_','rating_vs_team','age_vs_team')) or c.endswith('_peer_percentile'):
        return 'Team / peer context'
    if re.search(r'_(lag|change)[1-4]$',c) or '_past_' in c or c.startswith(('past_','recent_','rating_below_','rating_above_')) or c in ['history_length','rating_acceleration','core_growth_ahead_of_rating','technical_growth_ahead_of_rating','previous_peak_gap','first_observed_age','editions_since_first_seen']:
        return 'Player history'
    if c in ['age','height_cm','weight_kg','age_to_21','age_rating_interaction']:
        return 'Age / physique'
    return 'Current skills'

def add_feature_evidence(study,reuse=False):
    p=study['panel'];base=study['variants'][BASE]
    core=['overall','age','potential','potential_gap']+['role_'+r for r in ROLE_CORES]
    variants={ABLATIONS[0]:core,ABLATIONS[1]:[c for c in base if scouting_feature_group(c)!='Team / peer context'],ABLATIONS[2]:[c for c in base if scouting_feature_group(c)!='Player history']}
    feature_dict=pd.DataFrame([{'feature':c,'group':scouting_feature_group(c)} for c in base])
    feature_dict.to_csv(OUT/'feature_dictionary.csv',index=False)
    outputs=[];audits=[];fitted={};tick=time.monotonic()
    signature=hashlib.sha256(json.dumps({'variants':variants,'parameters':CB_PARAMS},sort_keys=True).encode()).hexdigest()
    for year in YEARS:
        cache=OUT/f'feature_ablation_{year}.joblib'
        if reuse and cache.exists():
            saved=joblib.load(cache)
            if saved.get('signature')==signature:
                outputs.append(saved['forecasts']);audits+=saved['audit'];fitted[year]=saved['models'];continue
        train=p.loc[p.fit_domain&p.outcome_end.le(year)&p.event_quality80.notna()]
        test=p.loc[p.eligible&p.edition.eq(year)&p.potential.notna()]
        out=study['forecasts'].loc[study['forecasts'].edition.eq(year)].copy()
        assert np.array_equal(out.row_id.to_numpy(),test.row_id.to_numpy())
        models={};audit=[]
        for name,cols in variants.items():
            if name==ABLATIONS[0] and (OUT/'same_learner_fifa_models.joblib').exists():
                bundle=joblib.load(OUT/'same_learner_fifa_models.joblib')[year]
                assert bundle['columns']==cols
                model=bundle['model']
            else:
                model=CatBoostClassifier(**CB_PARAMS)
                model.fit(model_frame(train,cols),train.event_quality80.astype(int),sample_weight=dated_weights(train))
            out[name]=model.predict(model_frame(test,cols),prediction_type='RawFormulaVal')
            models[name]={'model':model,'columns':cols}
            audit.append({'year':year,'method':name,'features':len(cols),'training_rows':len(train),'positives':int(train.event_quality80.sum()),'last_origin':int(train.edition.max()),'last_label':int(train.outcome_end.max())})
        joblib.dump({'signature':signature,'forecasts':out,'models':models,'audit':audit},cache,compress=3)
        outputs.append(out);audits+=audit;fitted[year]=models
        print(f'Feature controls {year}: {time.monotonic()-tick:.1f}s',flush=True)
    forecasts=pd.concat(outputs,ignore_index=True)
    methods=[FIFA]+ABLATIONS+[BASE]
    metrics=pd.DataFrame([measure(g,m,k) for _,g in forecasts.groupby('edition') for m in methods for k in [10,25,50,100,200,500]])
    metrics.to_csv(OUT/'feature_ablation_performance.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'feature_ablation_time_audit.csv',index=False)
    assert all(a['last_label']<=a['year'] and a['last_origin']<=a['year']-3 for a in audits)
    forecasts.to_csv(OUT/'feature_ablation_rankings.csv',index=False)
    sample=p.loc[p.edition.eq(2023)&p.eligible].sample(min(1200,int((p.edition.eq(2023)&p.eligible).sum())),random_state=SEED)
    bundle=study['bundles'][2023][BASE]
    values=bundle['model'].get_feature_importance(Pool(model_frame(sample,base)),type='ShapValues')
    scores=bundle['model'].predict(model_frame(sample,base),prediction_type='RawFormulaVal')
    np.testing.assert_allclose(values.sum(axis=1),scores,atol=1e-6)
    np.savez_compressed(OUT/'feature_shap_2023.npz',row_id=sample.row_id.to_numpy(),columns=np.array(base),values=sample[base].to_numpy(float),shap=values[:,:-1],expected=values[:,-1],score=scores)
    importance=pd.DataFrame({'feature':base,'group':[scouting_feature_group(c) for c in base],'mean_abs_shap':np.abs(values[:,:-1]).mean(axis=0)})
    importance.to_csv(OUT/'player_feature_importance.csv',index=False)
    discoveries=[]
    for year,g in forecasts.groupby('edition'):
        mr=pd.Series(np.arange(1,len(g)+1),index=ordered(g,BASE).index)
        fr=pd.Series(np.arange(1,len(g)+1),index=ordered(g,FIFA).index)
        a=g.loc[g.event_quality80.eq(1),['player_id','short_name','edition','overall','potential','observed_peak3','position_group']].copy()
        a['ml_rank']=mr.reindex(a.index);a['fifa_rank']=fr.reindex(a.index)
        a['selection']=np.select([a.ml_rank.le(50)&a.fifa_rank.le(50),a.ml_rank.le(50),a.fifa_rank.le(50)],['Both','ML only','FIFA only'],default='Neither')
        discoveries.append(a)
    discoveries=pd.concat(discoveries);discoveries.to_csv(OUT/'confirmed_discovery_comparison.csv',index=False)
    case_values=study['panel'].set_index('row_id').reindex(study['cases'].row_id)
    case_values[['player_id','edition','short_name']+base].to_csv(OUT/'case_feature_values.csv',index=False)
    intervals=feature_removal_intervals(forecasts)
    result={'variants':variants,'metrics':metrics,'forecasts':forecasts,'audit':pd.DataFrame(audits),'importance':importance,'sample':sample,'shap':values[:,:-1],'expected':values[:,-1],'scores':scores,'columns':base,'discoveries':discoveries,'intervals':intervals}
    (OUT/'feature_evidence_summary.json').write_text(json.dumps({'top50_hits':metrics.loc[metrics.k.eq(50)].groupby('method').found.sum().astype(int).to_dict(),'features':{k:len(v) for k,v in variants.items()},'confirmed_selection_overlap':discoveries.selection.value_counts().to_dict(),'scope':'Fixed settings and dated labels; retrospective feature-removal comparison. Correlated features can substitute for each other.'},indent=2))
    study['feature_evidence']=result
    return study

if __name__=='__main__':
    study=execute_future_study(reuse=True)
    study=add_feature_evidence(study,reuse=True)
    print((OUT/'feature_evidence_summary.json').read_text())
