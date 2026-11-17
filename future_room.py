from future_story import *
from plotly.offline import get_plotlyjs

def room_payload(study,reuse=True):
    cache=OUT/'room_payload.json'
    if reuse and cache.exists():
        saved=json.loads(cache.read_text())
        if saved.get('schema_version')==3:return saved
    raw=study['raw'];role=study['role'];forecast=study['forecasts'].copy();coords=role['coordinates']
    forecast['nationality_name']=forecast.nationality_name.fillna('Unknown');forecast['club_name']=forecast.club_name.fillna('No recorded club')
    km=KMeans(n_clusters=6,n_init=10,random_state=42).fit(role['z'][raw.edition.eq(2015)&raw.position_group.ne('GK')])
    clusters=km.predict(role['z'])
    forecast['x']=coords[forecast.row_id.to_numpy(),0];forecast['y']=coords[forecast.row_id.to_numpy(),1]
    forecast['cluster']=clusters[forecast.row_id.to_numpy()]
    methods=METHODS;rankcols=[]
    for j,m in enumerate(methods):
        c='rank'+str(j);rankcols.append(c);forecast[c]=0
        for year,g in forecast.groupby('edition'):
            r=ordered(g,m);forecast.loc[r.index,c]=np.arange(1,len(r)+1)
    cols=['player_id','edition','short_name','age','overall','potential','position_group','nationality_name','club_name',
          'x','y','event_quality80','complete3','observed_peak3','delta3_observed','cluster','learned_future_advantage','peak_seen']+rankcols
    def clean(x):
        if isinstance(x,(np.bool_,bool)):return bool(x)
        if pd.isna(x):return None
        if isinstance(x,(np.integer,int)):return int(x)
        if isinstance(x,(np.floating,float)):return round(float(x),5)
        return str(x)
    players=[[clean(x) for x in row] for row in forecast[cols].itertuples(index=False,name=None)]
    ids=set(forecast.player_id);histories={str(int(i)):[[int(r.edition),int(r.overall),None if pd.isna(r.potential) else int(r.potential)] for r in g.itertuples()] for i,g in raw.loc[raw.player_id.isin(ids)].groupby('player_id')}
    explanations={};features=study['panel'].set_index('row_id')
    for year,g in forecast.groupby('edition'):
        chosen=set()
        for m in [BASE,NEW,FIFA]:chosen.update(ordered(g,m).head(120).row_id)
        chosen.update(g.loc[g.event_quality80.eq(1),'row_id'])
        sample=features.reindex(sorted(chosen))
        for name in [BASE,NEW]:
            bundle=study['bundles'][int(year)][name];mat=model_frame(sample,bundle['columns'])
            shap=bundle['model'].get_feature_importance(Pool(mat),type='ShapValues')
            scores=bundle['model'].predict(mat,prediction_type='RawFormulaVal')
            np.testing.assert_allclose(shap.sum(axis=1),scores,atol=1e-6)
            for i,(_,r) in enumerate(sample.iterrows()):
                a=shap[i,:-1];top=np.argsort(-np.abs(a))[:7]
                explanations[f'{int(year)}:{int(r.player_id)}:{methods.index(name)}']={
                    'expected':round(float(shap[i,-1]),5),'score':round(float(scores[i]),5),
                    'features':[[nice(bundle['columns'][j]),round(float(a[j]),5)] for j in top],
                    'other':round(float(a.sum()-a[top].sum()),5)}
    neighbors={str(int(i)):g[['neighbor','neighbor_overall','distance','dimensions']].to_dict('records') for i,g in study['neighbors'].groupby('player_id')}
    pca=role['normalizer']['pca'];skills=role['normalizer']['skills'];axes=[]
    for v in pca.components_:
        pos=[nice(skills[j]) for j in np.argsort(-v)[:3]];neg=[nice(skills[j]) for j in np.argsort(v)[:3]]
        axes.append({'positive':pos,'negative':neg})
    payload={'schema_version':3,'columns':cols,'players':players,'methods':methods,'years':YEARS+[2026],'base':methods.index(BASE),'future':methods.index(NEW),'fifa':methods.index(FIFA),
             'histories':histories,'explanations':explanations,'neighbors':neighbors,'variance':role['normalizer']['explained_variance'],
             'axis_loadings':axes,'cluster_names':['Profile '+str(i+1) for i in range(6)]}
    cache.write_text(json.dumps(payload,separators=(',',':'),ensure_ascii=False));return payload

def export_room(study,reuse=True):
    payload=room_payload(study,reuse)
    packed=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':'),ensure_ascii=False).encode(),compresslevel=9)).decode()
    template=globals().get('ROOM_TEMPLATE')
    logic=globals().get('ROOM_LOGIC')
    if template is None:template=(Path(__file__).parent/'scout_room.html').read_text()
    if logic is None:logic=(Path(__file__).parent/'scout_room.js').read_text()
    page=template.replace('__PLOTLY__',get_plotlyjs()).replace('__PACKED__',packed).replace('__LOGIC__',logic)
    path=OUT/'Beyond_FIFA_Potential_Scout_Room.html';path.write_text(page);return path
