from future_room import *
from feature_evidence import *
from matplotlib.patches import Rectangle

def plot_decision_clock(study):
    fig,ax=plt.subplots(figsize=(12,3.6))
    items=[(2,2015,2020,INK,'Training profiles · outcomes finish by 2023'),(1,2015,2023,TEAL,'Player history and context known at the decision'),(0,2023,2026,GOLD,'Evaluation · the next three editions')]
    for y,start,end,color,label in items:
        ax.barh(y,end-start,left=start,height=.34,color=color)
        ax.text(2014.7,y,label,ha='right',va='center',fontsize=10)
    ax.axvline(2023,color=INK,ls='--');ax.text(2023,2.55,'2023 decision',ha='center',fontweight='bold')
    ax.set(xlim=(2014.7,2026.4),ylim=(-.6,2.8),yticks=[],xticks=np.arange(2015,2027),xlabel='Game edition');ax.grid(axis='y',visible=False)
    fig.suptitle('Every shortlist uses information available at the time',fontsize=18,fontweight='bold')
    finish(fig,'01_prediction_clock.png','The same expanding-window rule is applied to each forecast edition. Player IDs link records; names and IDs are excluded from the model.')

def main_metrics(study):
    return study['metrics'].loc[lambda q:q.k.eq(50)&q.target.eq('quality80')&q.method.isin([BASE,FIFA])].copy()

def plot_model_comparison(study):
    q=main_metrics(study);fig,axs=plt.subplots(1,2,figsize=(13,5));x=np.arange(4)
    for offset,name in [(-.18,FIFA),(.18,BASE)]:
        v=q.loc[q.method.eq(name)].set_index('year').reindex(YEARS).found
        bars=axs[0].bar(x+offset,v,width=.34,color=COLORS[name],label=name)
        for b,n in zip(bars,v):axs[0].text(b.get_x()+b.get_width()/2,n+.16,str(int(n)),ha='center',fontsize=11)
    axs[0].set(xticks=x,xticklabels=YEARS,ylabel='Confirmed future 80+ players in the top 50',ylim=(0,11));axs[0].legend(frameon=False,fontsize=9)
    axs[0].set_title('50 reviews in every edition')
    for name in [FIFA,BASE]:
        a=study['metrics'].loc[lambda m:m.method.eq(name)&m.target.eq('quality80')].groupby('k').found.sum()
        axs[1].plot(a.index,a.values,lw=2.5,marker='o',color=COLORS[name],label=name)
    axs[1].axvline(50,color=GREY,ls='--');axs[1].set(xlabel='Reviews per edition',ylabel='Confirmed 80+ selections across four lists')
    axs[1].set_title('The comparison across review budgets')
    fig.suptitle('The same scouting budget finds 22 future 80+ selections instead of 14',fontsize=17,fontweight='bold')
    finish(fig,'02_model_comparison.png','2020–2023. Every candidate starts below 70. Repeated players count once per annual shortlist; unresolved outcomes retain their slots.')

def plot_sampling_uncertainty(study):
    q=study['intervals'].loc[lambda x:x.model.eq(BASE)&x.baseline.eq(FIFA)]
    fig,ax=plt.subplots(figsize=(11,4.5))
    ax.errorbar(q.extra_hits,np.arange(len(q)),xerr=np.vstack([q.extra_hits-q.low,q.high-q.extra_hits]),fmt='o',color=TEAL,capsize=5,markersize=8)
    ax.axvline(0,color=INK,lw=1);ax.set_yticks(np.arange(len(q)),q.year.astype(str));ax.invert_yaxis();ax.set_xlabel('Additional confirmed 80+ selections · ML minus FIFA potential')
    fig.suptitle('The pooled gain is +8; the conditional 95% interval is +1 to +15',fontsize=17,fontweight='bold')
    finish(fig,'03_sampling_uncertainty.png','500 paired player-cluster bootstrap draws. Shared player IDs are resampled jointly; each full annual ranking is reselected at 50.')

def plot_feature_removal(study):
    e=study['feature_evidence'];q=e['metrics'].loc[lambda x:x.k.eq(50)].groupby('method').found.sum()
    methods=[FIFA,ABLATIONS[0],ABLATIONS[1],ABLATIONS[2],BASE]
    labels=['FIFA potential','Age, rating, potential + position','Without team / peer context','Without player history','Full player model']
    fig,axs=plt.subplots(1,2,figsize=(14,5.2),gridspec_kw={'width_ratios':[1.25,1]})
    bars=axs[0].barh(np.arange(5),[q[m] for m in methods],color=[GOLD,GREY,GREY,GREY,TEAL])
    axs[0].set_yticks(np.arange(5),labels,fontsize=10);axs[0].invert_yaxis();axs[0].set_xlim(0,max(q)*1.18)
    for b,m in zip(bars,methods):axs[0].text(q[m]+.3,b.get_y()+b.get_height()/2,str(int(q[m])),va='center',fontweight='bold')
    axs[0].set_xlabel('Confirmed 80+ selections · 200 reviews');axs[0].set_title('Retrain after removing a feature group')
    a=study['intervals'].loc[lambda x:x.model.eq(BASE)&x.baseline.eq(FIFA)&x.year.astype(str).eq('Pooled'),['baseline','extra_hits','low','high']]
    a=pd.concat([a,e['intervals'][['baseline','extra_hits','low','high']]],ignore_index=True)
    axs[1].errorbar(a.extra_hits,np.arange(len(a)),xerr=np.vstack([a.extra_hits-a.low,a.high-a.extra_hits]),fmt='o',color=INK,capsize=4,markersize=7)
    axs[1].axvline(0,color=GREY);axs[1].set_yticks(np.arange(4),['vs FIFA potential','vs FIFA inputs only','vs no context','vs no player history'],fontsize=10);axs[1].invert_yaxis()
    axs[1].set_xlabel('Full model: additional confirmed selections');axs[1].set_title('Paired conditional 95% intervals')
    fig.suptitle('Team and peer context make the largest difference in the removal check',fontsize=17,fontweight='bold')
    finish(fig,'04_feature_removal.png','All learned controls use the same target, dated training rows, weights and CatBoost settings. Removal losses overlap because features interact.')

def plot_feature_shap(study):
    e=study['feature_evidence'];imp=e['importance'].nlargest(12,'mean_abs_shap');cols=e['columns'];sample=e['sample'];rng=np.random.default_rng(42)
    fig,ax=plt.subplots(figsize=(12,6.5));cmap=plt.get_cmap('coolwarm')
    for i,c in enumerate(imp.feature):
        j=cols.index(c);v=sample[c];colour=v.rank(pct=True).to_numpy();sh=e['shap'][:,j];valid=v.notna().to_numpy()
        jitter=rng.normal(0,.11,len(sh));ax.scatter(sh[valid],i+jitter[valid],c=colour[valid],vmin=0,vmax=1,cmap=cmap,s=11,alpha=.5,edgecolors='none',rasterized=True)
        if (~valid).any():ax.scatter(sh[~valid],i+jitter[~valid],c=GREY,s=10,alpha=.4)
    ax.axvline(0,color=GREY,lw=1);ax.set_yticks(np.arange(len(imp)),[nice(c) for c in imp.feature]);ax.invert_yaxis()
    ax.set_xlabel('Contribution to the player-model score ← lower priority | higher priority →');ax.grid(axis='y',visible=False)
    sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,1));cb=fig.colorbar(sm,ax=ax,pad=.025,fraction=.028);cb.set_ticks([0,1],labels=['Low','High']);cb.set_label('Feature value within the sample')
    fig.suptitle('Potential still matters. Context changes the ranking.',fontsize=18,fontweight='bold')
    finish(fig,'05_feature_shap.png','Native SHAP · 1,200 randomly sampled eligible 2023 profiles. Features ordered by mean absolute contribution. Grey points indicate missing values.')

def plot_feature_patterns(study):
    e=study['feature_evidence'];fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ax,c in zip(axs,['team_top11_rating','position_age_pct_overall']):
        j=e['columns'].index(c);x=e['sample'][c].to_numpy();y=e['shap'][:,j];ok=np.isfinite(x);x=x[ok];y=y[ok]
        if c.endswith('pct_overall'):x=x*100
        ax.scatter(x,y,s=13,alpha=.2,color=BLUE,rasterized=True)
        t=pd.DataFrame({'value':x,'shap':y});t['bin']=pd.qcut(t.value,8,duplicates='drop')
        med=t.groupby('bin',observed=True)[['value','shap']].median();ax.plot(med.value,med.shap,color=TEAL,lw=2.3,marker='o',markersize=5)
        ax.axhline(0,color=GREY,lw=1);ax.set_xlabel('Club top-11 rating' if c=='team_top11_rating' else 'Rating rank among the same age + position · percentile')
        ax.set_ylabel('Contribution to the model score');ax.set_title('The club around the player' if c=='team_top11_rating' else 'A low overall rating in the right peer group')
    fig.suptitle('The extra signal is relative to the player’s surroundings',fontsize=17,fontweight='bold')
    finish(fig,'06_feature_patterns.png','Each point is one 2023 profile; lines connect median contributions in value bins. These show fitted-model associations, not effects of changing clubs or ranks.')

def plot_discovery_ranks(study):
    d=study['feature_evidence']['discoveries'];fig,ax=plt.subplots(figsize=(12,6.5));limit=max(d.ml_rank.max(),d.fifa_rank.max())*1.6
    ax.add_patch(Rectangle((50,.75),limit-50,49.25,color=TEAL,alpha=.06,zorder=0));ax.add_patch(Rectangle((.75,50),49.25,limit-50,color=GOLD,alpha=.06,zorder=0))
    for label,color,size in [('Neither',GREY,24),('Both',INK,60),('FIFA only',GOLD,75),('ML only',TEAL,75)]:
        g=d.loc[d.selection.eq(label)];ax.scatter(g.fifa_rank,g.ml_rank,s=size,color=color,alpha=.75,label=f'{label} · {len(g)}',zorder=3)
    ax.axhline(50,color=INK,ls='--',lw=1);ax.axvline(50,color=INK,ls='--',lw=1)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(.75,limit);ax.set_ylim(limit,.75)
    ax.set_xlabel('FIFA potential rank · logarithmic scale');ax.set_ylabel('Player-model rank · logarithmic scale')
    selected=d.loc[d.edition.eq(2023)&d.short_name.isin(['M. Svilar','M. Kerkez','A. Güler'])].copy()
    extra=d.loc[d.selection.eq('ML only')].assign(gain=lambda g:g.fifa_rank/g.ml_rank).sort_values('gain',ascending=False)
    selected=pd.concat([selected,extra]).drop_duplicates('player_id').head(6)
    offsets=[(12,-18),(12,14),(-70,15),(10,-20),(10,18),(-90,-15)]
    for ((_,r),offset) in zip(selected.iterrows(),offsets):
        ax.annotate(f'{r.short_name} · {str(r.edition)[-2:]}',(r.fifa_rank,r.ml_rank),xytext=offset,textcoords='offset points',fontsize=9,color=INK,arrowprops={'arrowstyle':'-','color':GREY,'lw':.8},bbox={'boxstyle':'round,pad=.25','fc':'white','ec':'none','alpha':.92})
    ax.legend(loc='lower right',frameon=True,fontsize=9);fig.suptitle('The model finds 11 breakouts FIFA misses; FIFA finds 3 the model misses',fontsize=16,fontweight='bold')
    finish(fig,'07_discovery_ranks.png','All 101 confirmed 80+ player-edition events. Dashed lines mark top 50. Names illustrate the ranking differences; repeated players can appear in different editions.')

def comparison_table(study):
    q=main_metrics(study);rows=[]
    for m in [BASE,FIFA]:
        g=q.loc[q.method.eq(m)];hits=int(g.found.sum());selected=int(g.selected.sum());total=int(g.confirmed_total.sum())
        rows.append({'Ranking':m,'Confirmed 80+':hits,'Distinct players':study['summary']['unique80_players'][m],'Reviews':selected,'Confirmed precision':f'{hits/selected:.1%}','Confirmed recall':f'{hits/total:.1%}'})
    return pd.DataFrame(rows).to_html(index=False,border=0,classes='future-table')

def annual_table(study):
    q=main_metrics(study);rows=[]
    for year,g in q.groupby('year'):
        r={'Edition':int(year),'Eligible players':int(g.eligible.iloc[0]),'Confirmed 80+':int(g.confirmed_total.iloc[0])}
        for method,short in [(BASE,'ML'),(FIFA,'FIFA')]:
            a=g.loc[g.method.eq(method)].iloc[0];r[short+' found / 50']=int(a.found);r[short+' precision']=f'{a.precision_lower:.1%}';r[short+' recall']=f'{a.confirmed_recall:.1%}';r[short+' unresolved']=int(a.unresolved)
        rows.append(r)
    return '<details><summary>Annual precision, recall and outcome counts</summary><div class="future-audit">'+pd.DataFrame(rows).to_html(index=False,border=0,classes='future-table')+'</div></details>'

def section_text(key,study):
    s=study['summary'];e=study['feature_evidence'];hits=s['hits_at50'];counts=e['metrics'].loc[lambda x:x.k.eq(50)].groupby('method').found.sum()
    if key=='question':
        return '''<p>FIFA potential is a natural place to start a scouting shortlist. I wanted to test whether the same review budget could find more future quality when I added a player's skills, development and surroundings.</p>
        <p>I ranked players with <b>overall below 70, at any age</b>, and checked whether they reached <b>80+ in the next three editions</b>. Previously observed 85+ players were excluded. I compared 50 selections per edition across 2020–2023.</p>
        <p>The model uses FIFA potential as one input. The question is whether the additional information improves the order in which players deserve attention.</p>'''
    if key=='timing':
        return '''<p>I trained each model using only outcomes available at its decision date. For a 2023 shortlist, training profiles end in 2020, with their three-edition outcomes known by 2023. The shortlist is then evaluated using 2024–2026 ratings.</p>
        <p>A recorded 80+ rating confirms a positive outcome. A negative requires all three future records. Players with incomplete follow-up stay unresolved: disappearing from the dataset does not count as stagnation.</p>'''
    if key=='comparison':
        return f'''<div class="future-finding"><b>The player model finds {hits[BASE]} confirmed future 80+ selections; FIFA potential finds {hits[FIFA]}.</b> That is eight additional hits from the same 200 review slots, or {hits[BASE]/hits[FIFA]-1:.0%} more confirmed hits.</div>
        {comparison_table(study)}
        <p>The model matches or exceeds potential in each annual top-50 list: 4 versus 1, 5 versus 5, 4 versus 3, and 9 versus 5. The budget curve shows how the comparison changes as the shortlist grows.</p>
        <p class="future-note">Some players recur across editions. Confirmed precision uses all review slots; confirmed recall uses the 101 documented positive player-edition outcomes. Unresolved players remain in both rankings.</p>'''
    if key=='uncertainty':
        a=study['intervals'].loc[lambda x:x.model.eq(BASE)&x.baseline.eq(FIFA)&x.year.astype(str).eq('Pooled')].iloc[0]
        return f'''<p>I resampled whole player IDs across all four editions and rebuilt the top-50 lists in each draw. This preserves the connection between repeated observations of the same player.</p>
        <p>The pooled difference is <b>+8 confirmed hits</b>, with a conditional 95% interval of <b>{a.low:+.0f} to {a.high:+.0f}</b>. Individual seasons have wider intervals. This calculation holds the fitted models and historical cohorts fixed; it does not include model-selection uncertainty or variation across future seasons.</p>'''
    if key=='removal':
        ci=e['intervals'].set_index('baseline')
        return f'''<p>I retrained the same model after removing two groups of information. The target, training dates, weights and CatBoost settings stayed fixed.</p>
        <p>Without <b>team and peer context</b>, confirmed hits fall from {hits[BASE]} to <b>{int(counts[ABLATIONS[1]])}</b>. Without <b>player history</b>, they fall to <b>{int(counts[ABLATIONS[2]])}</b>. A model using only age, current rating, potential, headroom and position finds <b>{int(counts[ABLATIONS[0]])}</b>.</p>
        <p>The conditional interval for the context difference is <b>{ci.loc[ABLATIONS[1],'low']:+.0f} to {ci.loc[ABLATIONS[1],'high']:+.0f}</b> hits. The history difference is less precise: <b>{ci.loc[ABLATIONS[2],'low']:+.0f} to {ci.loc[ABLATIONS[2],'high']:+.0f}</b>.</p>
        <p>The largest drop comes from removing context: club strength, standing within the squad, and comparisons with players of the same position and age. The removal losses overlap because the remaining features are retrained together.</p>'''
    if key=='features':
        return '''<p><b>Potential remains the strongest individual feature.</b> The added information changes how that potential is interpreted. Among the leading inputs are club top-11 strength, listed squad status, rating rank for the same age and position, and position-specific skill rank.</p>
        <p>The chart shows how each feature contributes to the fitted ranking. Points to the right raise a player's score; colour shows whether the feature value is relatively low or high. Importance is measured on a random sample of 1,200 eligible 2023 players, rather than only successful discoveries.</p>
        <p>Club strength and peer standing help distinguish players with similar overall ratings. These are predictive associations: a contribution from club strength does not establish that joining a stronger club causes development.</p>'''
    if key=='ranks':
        return '''<p>The difference between the lists is visible at player level. <b>Eleven confirmed breakouts appear only in the ML top 50; three appear only in FIFA's.</b> Another eleven appear in both. The net gain is eight selections.</p>
        <p>The rank plot includes all 101 confirmed player-edition breakouts. Players near the top right were prioritised by ML and ranked much lower by FIFA potential. Players near the bottom left show the reverse.</p>'''
    if key=='similarity':
        ev=100*sum(s['geometry_variance'])
        return f'''<p>I also compared players by the shape of their attributes. The coordinates were fitted on 2015 outfield profiles and then held fixed. The two displayed dimensions retain <b>{ev:.1f}%</b> of profile variation.</p>
        <p>Similar profiles can still have different outcomes. The map combines similarity with the two shortlists and the later rating reveal. Nearest-player comparisons use all 28 normalised skill dimensions within the same position group; goalkeeper comparisons use five goalkeeping attributes.</p>'''
    if key=='explorer':
        return '''<p>The scouting replay lets me compare rankings within an edition, position or nationality. Filters rerank the full eligible pool. Clicking a player reveals the recorded career and an available individual explanation.</p>'''
    if key=='cases':
        t=study['cases'][['short_name','overall','potential','ml_rank','fifa_rank','observed_peak3','case']].copy();t.columns=['Player','Overall','Potential','ML rank','FIFA rank','Peak in window','Outcome']
        return '''<p>In 2023, <b>M. Kerkez</b> and <b>M. Svilar</b> both started at 69 and later reached 82. The model ranked them fifth and thirtieth; FIFA potential ranked them fifty-second and 376th.</p>
        <p>The same model ranked <b>Pablo Torre</b> first, but his observed peak in the window was 74. It also placed <b>A. Güler</b> sixty-second and missed him at top 50. These examples include discoveries, a false alarm and a miss.</p>'''+t.to_html(index=False,border=0,classes='future-table',float_format=lambda x:f'{x:.0f}')+'''
        <p>Individual SHAP values connect these rankings to specific inputs. They sum, with the expected score, to the fitted raw score. The displayed bars are the six largest contributions; the score is a ranking index rather than a calibrated probability.</p>'''
    if key=='findings':
        return f'''<div class="future-finding"><b>At a fixed review budget, adding player information improves the historical 80+ shortlist.</b> The gain is {hits[BASE]} versus {hits[FIFA]} confirmed selections. Team and peer context produce the largest loss when removed.</div>
        <p>The stricter pool of players never previously rated 70+ gives <b>{s['never70_at50'][BASE]} versus {s['never70_at50'][FIFA]}</b> confirmed selections. For the rarer 85+ outcome, the model and potential each find <b>{s['star85_hits'][BASE]}</b>; there are only ten documented 85+ player-edition events in these cohorts.</p>
        <p>I also tested forecast role-profile features. That extension finds {hits[NEW]} future 80+ selections at top 50, below the player model's {hits[BASE]}. The improvement over potential comes from the player model; the role forecast did not improve this shortlist.</p>
        <p>The scouting value is a better order for reviewing low-rated players. The next evaluation is a frozen pipeline on later-maturing cohorts, with match performance and playing time alongside game ratings.</p>'''
    raise KeyError(key)

def methods_html(study):
    dictionary=pd.read_csv(OUT/'feature_dictionary.csv');counts=dictionary.groupby('group').size().rename('Inputs').reset_index().rename(columns={'group':'Feature group'})
    return '''<details><summary>Data and methods</summary>
    <p>Men's annual FIFA15–FC26 snapshots. Exact player IDs link editions. Historical snapshots use the first available update; FC26 is the September 2025 snapshot. FC25 supplies future ratings but lacks potential.</p>
    <p>The player model has 431 inputs: current skills, age and physique, potential, past development, and contemporaneous team/peer context. Context uses edition-specific club and league IDs where available. Peer groups use exact age and primary position, or two-year age bands and a broader role. Percentiles are shrunk toward 0.5 for small groups. Names, player IDs, nationality and future outcomes are excluded from the predictors.</p>'''+counts.to_html(index=False,border=0,classes='future-table')+'''
    <p>CatBoost: 450 iterations, depth 5, learning rate 0.045, L2 12, border count 64, seed 42. Training includes sub-80 origins with no previously observed 85+ rating; labels must finish by the forecast edition. Sub-70 origins receive extra weight, repeated player histories are balanced, and rare large positive jumps receive bounded extra weight. All feature-removal models use the same settings and weights. Past club and peer changes remain in the context block; player history contains the player's own rating/skill changes, trends and observed career coverage.</p>
    <p>FIFA potential ranks descending; ties use higher current overall, younger age, then player ID. ML ties use player ID. Potential describes a longer horizon than the three-edition target; the comparison is at the same review budget.</p>
    <p>Precision is confirmed positives divided by all selections. The corresponding upper bound treats unresolved selections as positive. Recall uses documented positive outcomes. Source dropout can leave outcomes unknown. This is a retrospective replay on a previously explored archive, rather than untouched prospective validation. Bootstrap intervals condition on the fitted models and the four historical cohorts.</p>
    <p>The similarity transform, scaling and PCA are fitted in 2015. The outfield map excludes goalkeepers; scouting rankings include them. The interactive FC26 watchlist is based on the September 2025 source snapshot and has no mature three-edition outcomes.</p>
    <p>Sources: <a href="https://www.kaggle.com/datasets/stefanoleone992/ea-sports-fc-24-complete-player-dataset">Stefano Leone · FIFA15–FC24 · CC0</a>; <a href="https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats">Davis Nyagami · FC25 · Apache 2.0</a>; <a href="https://www.kaggle.com/datasets/rovnez/fc-26-fifa-26-player-data">rovnez · FC26 · CC BY 4.0</a>. Earlier research on skill evolution: <a href="https://dtai.cs.kuleuven.be/events/MLSA19/papers/verstraete_MLSA19.pdf">Verstraete et al., 2019</a>. Individual explanations use <a href="https://catboost.ai/docs/en/concepts/shap-values">native CatBoost SHAP</a>.</p>
    </details>'''

STORY=[('question','1. A shortlist before the breakthrough',[]),('timing','2. The prediction timeline',['01_prediction_clock.png']),('comparison','3. More discoveries from the same budget',['02_model_comparison.png']),('uncertainty','4. Uncertainty around the gain',['03_sampling_uncertainty.png']),('removal','5. The information behind the improvement',['04_feature_removal.png']),('features','6. The signals learned by the model',['05_feature_shap.png','06_feature_patterns.png']),('ranks','7. The players each approach finds',['07_discovery_ranks.png']),('similarity','8. Similar profiles, different futures',['08_player_similarity.png']),('explorer','9. An interactive scouting replay',[]),('cases','10. Discoveries, false alarms and misses',['09_player_careers.png','10_player_explanations.png']),('findings','11. What I found',[])]

def reading_room(room):
    return '<div class="future-room"><iframe title="Interactive hidden-gem scouting replay" sandbox="allow-scripts" style="width:100%;height:1650px;border:0" srcdoc="'+html.escape(room.read_text(),quote=True)+'"></iframe></div>'

def export_scouting_article(study,room):
    parts=[]
    for key,title,figures in STORY:
        part='<section><h2>'+title+'</h2>'+section_text(key,study)
        for name in figures:part+='<figure><img alt="'+html.escape(title)+'" src="data:image/png;base64,'+base64.b64encode((OUT/name).read_bytes()).decode()+'"></figure>'
        if key=='comparison':part+=annual_table(study)
        if key=='explorer':part+=reading_room(room)
        if key=='findings':part+=methods_html(study)
        parts.append(part+'</section>')
    css='''body{margin:0;background:#f2f5f6;color:#173441;font:16px/1.75 system-ui,-apple-system,Arial,sans-serif}header{background:#173441;color:white;padding:43px max(24px,calc((100% - 1140px)/2)) 36px}header small{letter-spacing:2px;text-transform:uppercase;color:#9ed8c7}header h1{font-size:37px;line-height:1.2;max-width:1000px}header p{color:#d5e4e8;max-width:1050px}main{max-width:1140px;margin:24px auto;background:#fff;padding:28px 40px;border:1px solid #dce5e9;border-radius:10px}h2{font-size:25px;line-height:1.4;margin:36px 0 17px}section{padding-bottom:15px}figure{margin:23px 0}figure img{width:100%;height:auto}a{color:#008777}@media(max-width:650px){main{padding:16px}header{padding:28px 18px}header h1{font-size:28px}}'''
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Beyond FIFA Potential: Finding Hidden Gems</title><style>'+css+'</style>'+READER_CSS+'</head><body><header><small>Football scouting · FIFA15–FC26</small><h1>Beyond FIFA potential: finding hidden gems before the breakthrough</h1><p>I compared a player model with FIFA potential to find future 80+ players while they were still rated below 70.</p></header><main>'+''.join(parts)+'</main></body></html>'
    path=OUT/'Beyond_FIFA_Potential_Article.html';path.write_text(page);return path
