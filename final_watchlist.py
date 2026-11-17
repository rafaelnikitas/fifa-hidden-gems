from story_sections import *
from datetime import date,datetime
from zoneinfo import ZoneInfo

RUN_DATE=os.environ.get('FIFA_ANALYSIS_DATE',datetime.now(ZoneInfo('Europe/Athens')).date().isoformat())
BUILTIN_SNAPSHOT_DATE='2025-09-21'

def snapshot_checks(frame):
    required=list(dict.fromkeys(['player_id','short_name','overall','potential','age','height_cm','weight_kg','player_positions','preferred_foot','club_name','league_name','club_position','weak_foot','skill_moves','pace','shooting','passing','dribbling','defending','physic']+SKILLS))
    missing=sorted(set(required)-set(frame.columns))
    if missing:raise ValueError('The current snapshot is missing required source fields: '+', '.join(missing))
    if frame.player_id.duplicated().any():raise ValueError('The current snapshot must contain one record per exact player ID.')
    for c in ['overall','potential','age','height_cm']:
        if pd.to_numeric(frame[c],errors='coerce').notna().mean()<.95:raise ValueError('Insufficient current coverage for '+c+'. No historical or invented potential is substituted.')
    for c in ['short_name','club_name','league_name','club_position']:
        if frame[c].notna().mean()<.9:raise ValueError('Insufficient current coverage for '+c+'.')
    if not pd.to_numeric(frame.overall,errors='coerce').between(1,99).all():raise ValueError('Overall ratings must be between 1 and 99.')
    return True

def load_current_snapshot(study):
    configured=os.environ.get('FIFA_CURRENT_SNAPSHOT_CSV','').strip()
    automatic=list(INPUT_ROOT.rglob('current_scouting_snapshot.csv')) if INPUT_ROOT.exists() else []
    if configured or automatic:
        path=Path(configured) if configured else automatic[0]
        if not configured and len(automatic)!=1:raise ValueError('Attach only one current_scouting_snapshot.csv.')
        current=pd.read_csv(path,low_memory=False);snapshot_checks(current)
        if 'edition' not in current:
            if 'fifa_version' not in current:raise ValueError('The current snapshot needs edition or fifa_version.')
            current['edition']=pd.to_numeric(current.fifa_version).astype(int)+2000
        if current.edition.nunique()!=1:raise ValueError('The current snapshot must contain a single game edition.')
        asof=os.environ.get('FIFA_CURRENT_SNAPSHOT_DATE','').strip()
        if not asof and 'snapshot_date' in current and current.snapshot_date.nunique()==1:asof=str(current.snapshot_date.iloc[0])
        if not asof:raise ValueError('Supply an ISO snapshot_date or FIFA_CURRENT_SNAPSHOT_DATE. The run date is not substituted for the data date.')
        asof=date.fromisoformat(asof).isoformat()
        if asof>RUN_DATE:raise ValueError('The snapshot date is later than the analysis run date.')
        current['source']=os.environ.get('FIFA_CURRENT_SOURCE_URL',path.name)
        meta={'snapshot_date':asof,'edition':int(current.edition.iloc[0]),'source':str(current.source.iloc[0]),'custom_snapshot':True}
        raw=pd.concat([study['raw'].loc[study['raw'].edition.lt(meta['edition'])],current],ignore_index=True)
        for c in study['raw'].columns:
            if c not in raw:raw[c]=np.nan
        for c in NUMERIC:raw[c]=pd.to_numeric(raw[c],errors='coerce')
        raw=raw.sort_values(['player_id','edition']).reset_index(drop=True);raw['row_id']=np.arange(len(raw))
        raw['primary_position']=raw.player_positions.fillna('Unknown').str.split(',').str[0].str.strip()
        mapping={'GK':'GK','CB':'CB','LB':'FB','RB':'FB','LWB':'FB','RWB':'FB','CDM':'MF','CM':'MF','CAM':'MF','LM':'Wing','RM':'Wing','LW':'Wing','RW':'Wing','ST':'FW','CF':'FW'}
        raw['position_group']=raw.primary_position.map(mapping).fillna('Other')
        clean,_,_=canonical_ids(raw)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',pd.errors.PerformanceWarning)
            panel,_,_=strong_history_panel(clean)
        lookup=raw.set_index('row_id')
        for c in ['short_name','long_name','club_name','league_name','nationality_name']:
            panel[c]=panel.row_id.map(lookup[c])
    else:
        panel=study['panel'];meta={'snapshot_date':BUILTIN_SNAPSHOT_DATE,'edition':2026,'source':'rovnez / FC26_20250921.csv','custom_snapshot':False}
    trained=[year for year in study['bundles'] if int(year)<=meta['edition'] and BASE in study['bundles'][year]]
    meta['model_edition']=max(trained)
    meta['snapshot_age_days']=(date.fromisoformat(RUN_DATE)-date.fromisoformat(meta['snapshot_date'])).days
    return panel,meta

def readable_value(feature,value):
    if pd.isna(value):return 'not recorded'
    value=float(value)
    if 'percentile' in feature or '_pct_' in feature:return f'{100*value:.1f} percentile'
    if feature=='height_cm':return f'{value:.0f} cm'
    if feature=='weight_kg':return f'{value:.0f} kg'
    if feature=='age':return f'{value:.0f} years'
    if feature in ['club_named_starter','club_substitute','club_reserve']:return 'yes' if value>.5 else 'no'
    if abs(value-round(value))<1e-6:return f'{value:.0f}'
    return f'{value:.2f}'

def build_final_watchlist(study,top_n=20):
    panel,meta=load_current_snapshot(study);edition=meta['edition'];bundle=study['bundles'][meta['model_edition']][BASE]
    pool=panel.loc[panel.eligible&panel.edition.eq(edition)&panel.potential.notna()].copy()
    pool[BASE]=bundle['model'].predict(model_frame(pool,bundle['columns']),prediction_type='RawFormulaVal')
    pool[FIFA]=pool.potential
    if not meta['custom_snapshot']:
        saved=study['forecasts'].loc[study['forecasts'].edition.eq(edition)].set_index('player_id')
        np.testing.assert_allclose(pool[BASE],saved.reindex(pool.player_id)[BASE],atol=1e-6,rtol=1e-6)
    ranked=ordered(pool,BASE).copy();ranked['ml_rank']=np.arange(1,len(ranked)+1)
    fifa=ordered(pool,FIFA);fr=pd.Series(np.arange(1,len(fifa)+1),index=fifa.player_id)
    ranked['fifa_rank']=ranked.player_id.map(fr);top=ranked.head(top_n).copy()
    shap=bundle['model'].get_feature_importance(Pool(model_frame(top,bundle['columns'])),type='ShapValues')
    np.testing.assert_allclose(shap.sum(axis=1),top[BASE],atol=1e-6)
    rows=[];reasons=[];potcols=[j for j,c in enumerate(bundle['columns']) if c in ['potential','potential_gap']]
    for i,(_,player) in enumerate(top.iterrows()):
        values=shap[i,:-1];positive=np.flatnonzero(values>0);negative=np.flatnonzero(values<0)
        supports=positive[np.argsort(-values[positive])[:2]];brakes=negative[np.argsort(values[negative])[:1]]
        text=lambda j:f'{nice(bundle["columns"][j])}: {readable_value(bundle["columns"][j],player[bundle["columns"][j]])} ({values[j]:+.2f})'
        reasons.append({'player_id':int(player.player_id),'support':'; '.join(text(j) for j in supports),'brake':'; '.join(text(j) for j in brakes)})
        for j,c in enumerate(bundle['columns']):rows.append({'player_id':int(player.player_id),'player':player.short_name,'ml_rank':int(player.ml_rank),'feature':c,'feature_label':nice(c),'value':None if pd.isna(player[c]) else float(player[c]),'shap':float(values[j]),'group':'FIFA potential + headroom' if j in potcols else 'Other player / context inputs'})
    top=top.merge(pd.DataFrame(reasons),on='player_id',how='left',validate='one_to_one')
    top['snapshot_date']=meta['snapshot_date'];top['snapshot_edition']=edition
    top['raw_score']=top[BASE];top['expected_score']=shap[:,-1]
    top['potential_contribution']=shap[:,potcols].sum(axis=1);top['other_contribution']=shap[:,:-1].sum(axis=1)-top.potential_contribution
    top['past_editions_available']=top.overall_past_observations.sub(1).fillna(0).astype(int)
    exported=['ml_rank','player_id','short_name','age','position_group','club_name','overall','potential','fifa_rank','raw_score','expected_score','potential_contribution','other_contribution','past_editions_available','support','brake','snapshot_date','snapshot_edition']
    top[exported].to_csv(OUT/'latest_model_shortlist.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'latest_shortlist_shap.csv',index=False)
    audit={'snapshot':meta,'eligible_players':len(pool),'top_n':len(top),'future_target_editions':[edition+1,edition+2,edition+3],'model':BASE,'features':len(bundle['columns']),'hyperparameter_search_performed':False,'probability_calibration_performed':False,'shap_additivity_checked':True,'outcomes':'Unconfirmed. No future ratings are used in the current shortlist.','parameters':bundle['model'].get_params()}
    (OUT/'latest_shortlist_audit.json').write_text(json.dumps(audit,indent=2))
    study['watchlist']={'top':top,'shap':pd.DataFrame(rows),'meta':meta,'audit':audit}
    return study

def current_shortlist_html(study):
    w=study['watchlist'];top=w['top'];meta=w['meta'];edition=meta['edition']
    dated=meta['snapshot_age_days']>90
    status='Dated snapshot · not a live current-season shortlist' if dated else 'Latest attached snapshot · prospective outcomes unconfirmed'
    header=f'<div class="future-finding"><b>{html.escape(status)}</b><br>Data as of {meta["snapshot_date"]} · FC{edition%100:02d} · {w["audit"]["eligible_players"]:,} eligible players · model: {BASE}.<br>Target: 80+ in editions {edition+1}–{edition+3}; all listed outcomes remain unconfirmed.</div>'
    rows=[]
    for _,r in top.iterrows():
        rows.append('<tr><td>'+str(int(r.ml_rank))+'</td><td><b>'+html.escape(r.short_name)+'</b><br><small>'+html.escape(str(r.club_name))+'</small></td><td>'+f'{r.age:.0f} · {r.position_group}'+'</td><td>'+f'{r.overall:.0f} / {r.potential:.0f}'+'</td><td>'+str(int(r.fifa_rank))+'</td><td>'+html.escape(r.support)+'</td><td>'+html.escape(r.brake)+'</td></tr>')
    table='<div style="overflow:auto"><table class="future-table latest-list"><thead><tr><th>ML rank</th><th>Player / club</th><th>Age / role</th><th>OVR / POT</th><th>FIFA rank</th><th>Main supporting signals · SHAP</th><th>Main opposing signal · SHAP</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    details=[]
    for _,r in top.iterrows():
        c=w['shap'].loc[w['shap'].player_id.eq(r.player_id)].copy();c=c.loc[c.shap.abs().nlargest(8).index]
        t=pd.DataFrame({'Feature':c.feature_label,'Value':[readable_value(f,v) for f,v in zip(c.feature,c.value)],'Contribution':c.shap.map(lambda x:f'{x:+.3f}')})
        details.append(f'<details><summary>#{int(r.ml_rank)} · {html.escape(r.short_name)} — score detail</summary><p>FIFA potential and headroom contribute {r.potential_contribution:+.3f}; the other inputs contribute {r.other_contribution:+.3f}. Expected score {r.expected_score:+.3f} + all contributions = {r.raw_score:+.3f}. Earlier snapshots available within the previous four editions: {int(r.past_editions_available)}.</p>'+t.to_html(index=False,border=0,classes='future-table')+'<p class="future-note">Eight largest feature contributions shown. The complete decomposition is exported. Contributions describe the fitted model, not development causes or star probabilities.</p></details>')
    return header+table+''.join(details)

def plot_current_priorities(study):
    top=study['watchlist']['top'].head(12);fig,ax=plt.subplots(figsize=(12,6.2));y=np.arange(len(top))
    for i,(_,r) in enumerate(top.iterrows()):ax.plot([r.ml_rank,r.fifa_rank],[i,i],color=PALE,lw=3,zorder=1)
    ax.scatter(top.ml_rank,y,color=TEAL,s=55,label='Player model',zorder=3);ax.scatter(top.fifa_rank,y,color=GOLD,s=55,label='FIFA potential',zorder=3)
    ax.set_xscale('log');ax.set_yticks(y,top.short_name);ax.invert_yaxis();ax.set_xlabel('Rank in the same eligible pool · lower is reviewed earlier · logarithmic scale')
    ax.legend(frameon=False,loc='lower right');fig.suptitle('Where the model looks first in the latest complete snapshot',fontsize=17,fontweight='bold')
    finish(fig,'11_latest_shortlist_ranks.png',f'Data as of {study["watchlist"]["meta"]["snapshot_date"]}. Current overall below 70; all ages. These are unconfirmed prospective selections, not observed discoveries.')

def plot_current_player_reasons(study):
    w=study['watchlist'];top=w['top'].head(3);fig,axs=plt.subplots(1,len(top),figsize=(14,5))
    for ax,(_,r) in zip(np.atleast_1d(axs),top.iterrows()):
        c=w['shap'].loc[w['shap'].player_id.eq(r.player_id)].copy();c=c.loc[c.shap.abs().nlargest(6).index].sort_values('shap')
        ax.barh([x[:38] for x in c.feature_label],c.shap,color=np.where(c.shap.ge(0),TEAL,RED));ax.axvline(0,color=GREY)
        ax.set_title(f'#{int(r.ml_rank)} {r.short_name}\nOVR {r.overall:.0f} · POT {r.potential:.0f}',fontweight='bold',fontsize=11);ax.set_xlabel('SHAP contribution',fontsize=11)
    fig.suptitle('Why these profiles receive early scouting attention',fontsize=18,fontweight='bold')
    finish(fig,'12_latest_player_reasons.png','Six largest native SHAP contributions per player. The model includes FIFA potential; the other inputs change how that potential is prioritised.')

def selection_summary_html(study):
    metrics=study['metrics'].loc[lambda q:q.k.eq(50)&q.target.eq('quality80')].groupby('method').found.sum()
    control=int(study['feature_evidence']['metrics'].loc[lambda q:q.k.eq(50)&q.method.eq('FIFA inputs only')].found.sum())
    rows=[{'Candidate':FIFA,'Inputs':'Potential ranking','Confirmed 80+ selections':int(metrics[FIFA])},{'Candidate':'FIFA inputs only · CatBoost','Inputs':'11 · age / rating / potential / role','Confirmed 80+ selections':control}]
    for name in [BASE,STATIC_MODEL,LINEAR,NEW]:rows.append({'Candidate':name,'Inputs':str(len(study['variants'][name])),'Confirmed 80+ selections':int(metrics[name])})
    table=pd.DataFrame(rows).to_html(index=False,border=0,classes='future-table')
    stages=pd.DataFrame([['Validation','Four chronological expanding-window replays; every training outcome ends by the decision edition.'],['Selection criterion','Confirmed 80+ selections at top 50, with precision, recall, unique players, misses and conditional intervals alongside it.'],['Hyperparameters','One fixed CatBoost configuration for the player-feature comparisons. No automated hyperparameter search or early-stopping selection on the evaluation cohorts.'],['Fixed settings','450 iterations; depth 5; learning rate 0.045; L2 12; border count 64; seed 42.'],['Training weights','Sub-70 emphasis, balancing repeated player histories and bounded extra weight for rare large improvements.'],['Calibration','Not performed. The outputs are raw ranking scores, rather than calibrated probabilities.'],['Latest-snapshot refit','The selected specification is fitted to all eligible mature labels available by the forecast edition. The bundled FC26 model uses 66,811 training profiles from origins 2015–2023.'] ],columns=['Step','What I used'])
    return table+stages.to_html(index=False,border=0,classes='future-table')

def case_comparison_html(study):
    frame=study['cases'][['short_name','overall','potential','ml_rank','fifa_rank','observed_peak3','case']].copy()
    frame.columns=['Player','Overall','Potential','ML rank','FIFA rank','Peak in window','Outcome']
    return frame.to_html(index=False,border=0,classes='future-table',float_format=lambda x:f'{x:.0f}')

def newer_source_audit_html():
    rows=[['Current model-compatible archive','2025-09-21','Recorded','Used for the dated shortlist'],['EAFC26 Player Database · Flynn','2026-04-02','No potential field','Not substituted'],['EA FC 27 · EA API community export','2026-09-12','No potential; height / weight empty','Not substituted'],['FC27 · FIFPlay community export','2026-10-11','Potential empty in all 10,762 rows; duplicate IDs and mostly generic league labels','Not substituted']]
    frame=pd.DataFrame(rows,columns=['Source checked','Snapshot / file date','Compatibility','Use'])
    return '<details><summary>Source freshness and compatibility</summary>'+frame.to_html(index=False,border=0,classes='future-table')+'<p>Checked 2026-11-16. A file dated later is not enough: the model needs current potential, identity, age, physical data, skills and team context. Potential is not silently carried over from an older edition.</p></details>'
