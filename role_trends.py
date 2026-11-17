from ratings_loader import *
from sklearn.decomposition import PCA
import joblib,json,warnings

ROLES=['CB','FB','MF','Wing','FW']
LEAGUES=[13,16,19,31,53]
ROLE_LABELS={'CB':'Centre-backs','FB':'Full-backs','MF':'Central midfielders','Wing':'Wide players','FW':'Forwards','GK':'Goalkeepers'}
OUT=Path(os.environ.get('FUTURE_OUTPUT','results'));OUT.mkdir(exist_ok=True,parents=True)
SEED=42

def canonical_ids(raw):
    p=raw.copy()
    original=p[['row_id','club_name','league_name']].copy()
    native=p.club_team_id.notna()&p.league_id.notna()
    p.loc[native,'club_name']='club_'+p.loc[native,'club_team_id'].astype(int).astype(str)
    p.loc[native,'league_name']='league_'+p.loc[native,'league_id'].astype(int).astype(str)
    p.loc[~native&p.club_name.isna(),'club_name']=''
    p.loc[~native&p.league_name.isna(),'league_name']=''
    audit=raw.groupby(['edition','league_id']).league_name.nunique().reset_index(name='name_variants')
    audit=audit.loc[audit.name_variants.gt(1)]
    return p,original,audit

def normalize_shapes(raw):
    # The coordinate system is anchored to the first edition, before any forecast.
    seed=raw.loc[raw.edition.eq(2015)&raw.position_group.isin(ROLES)]
    skills=[c for c in SKILLS if not c.startswith('goalkeeping_') and seed[c].notna().mean()>.95]
    def design(f):
        a=(f.age.to_numpy(float)-25)/10;o=(f.overall.to_numpy(float)-70)/10
        return np.column_stack([np.ones(len(f)),a,a*a,o,o*o]+[f.position_group.eq(r).to_numpy(float) for r in ROLES])
    med=seed[skills].median().to_numpy();x=design(seed);y=seed[skills].fillna(dict(zip(skills,med))).to_numpy(float)
    penalty=np.eye(x.shape[1])*20;penalty[0,0]=0
    beta=np.linalg.solve(x.T@x+penalty,x.T@y)
    resid=y-x@beta;scale=np.maximum(resid.std(axis=0),3.)
    def transform(f):
        z=(f[skills].fillna(dict(zip(skills,med))).to_numpy(float)-design(f)@beta)/scale
        z-=z.mean(axis=1,keepdims=True)
        return np.clip(z,-5,5)
    z=transform(raw)
    pc=PCA(n_components=2,svd_solver='full').fit(transform(seed))
    coordinates=pc.transform(z)
    bundle={'skills':skills,'medians':med,'beta':beta,'scale':scale,'pca':pc,'normalizer_fit_end':2015,
            'explained_variance':pc.explained_variance_ratio_.tolist()}
    return z,coordinates,bundle

def cohort_profiles(raw,z,skills):
    f=raw.copy();f['_league_id']=f.league_id
    aliases={'Premier League':13,'Ligue 1 McDonald\'s':16,'Bundesliga':19,'Serie A Enilive':31,'LALIGA EA SPORTS':53}
    missing=f._league_id.isna();f.loc[missing,'_league_id']=f.loc[missing,'league_name'].map(aliases)
    # Both groups use the same age bins and fixed five leagues. Candidates <70 never define prototypes.
    f['_age_bin']=np.where(f.age.le(25),0,1)
    f['_band']=np.select([f.overall.between(80,89),f.overall.between(70,79)],['elite','reference'],default='omit')
    mask=f.age.between(20,32)&f.position_group.isin(ROLES)&f._league_id.isin(LEAGUES)&f._band.ne('omit')
    rows=f.loc[mask,['player_id','edition','position_group','_league_id','_age_bin','_band']].copy()
    zz=['z'+str(i) for i in range(z.shape[1])]
    rows[zz]=z[mask]
    keys=['edition','position_group','_league_id','_age_bin','_band']
    grouped=rows.groupby(keys)[zz].mean();n=rows.groupby(keys).size()
    records=[];profiles={};support={}
    # Equal weight to each supported league/age cell. A minimum of two elite and reference players.
    for (year,role),part in rows.groupby(['edition','position_group']):
        cells=[];counts=[]
        for league in LEAGUES:
            for age in [0,1]:
                ke=(year,role,league,age,'elite');kr=(year,role,league,age,'reference')
                if ke in n and kr in n and n.loc[ke]>=2 and n.loc[kr]>=2:
                    cells.append(grouped.loc[ke].to_numpy()-grouped.loc[kr].to_numpy())
                    counts.append(int(n.loc[ke]+n.loc[kr]))
        if len(cells)<3:continue
        mu=np.mean(cells,axis=0);profiles[(int(year),role)]=mu
        support[(int(year),role)]={'cells':len(cells),'profiles':sum(counts)}
        for skill,value in zip(skills,mu):records.append({'edition':int(year),'role':role,'skill':skill,'elite_premium':float(value),'cells':len(cells),'profiles':sum(counts)})
    return profiles,support,pd.DataFrame(records),rows

STATIC=['role_fit_now','role_distance_now','role_fit_available','role_reference_cells']
DYNAMIC=['role_fit_forecast','role_distance_forecast','future_fit_advantage','trend_alignment','role_trend_speed','role_trend_coherence','role_trend_years','role_trend_available']
LEARNED=['learned_future_fit','learned_future_distance','learned_future_advantage','learned_future_alignment',
         'learned_change_size','learned_forecast_training_r2','learned_forecast_transitions','learned_forecast_available']

def learn_profile_forecasts(raw,z,profiles):
    values={c:np.full(len(raw),np.nan) for c in LEARNED};values[LEARNED[-1]][:]=0
    audit=[];scores=[];coefficients=[]
    def predictors(h):
        h=np.asarray(h[-4:]);t=np.arange(len(h),dtype=float);t-=t.mean()
        slope=(t[:,None]*(h-h.mean(axis=0))).sum(axis=0)/max(t@t,1e-8)
        return np.column_stack([np.mean(h[-3:],axis=0)-h[-1],slope,h[-1]-h[-2]])
    for year in sorted(raw.edition.unique()):
        xx=[];yy=[];origins=[]
        for r in ROLES:
            for origin in range(2016,int(year)):
                past=[profiles[(s,r)] for s in range(max(2015,origin-3),origin+1) if (s,r) in profiles]
                if len(past)<2 or (origin+1,r) not in profiles:continue
                xx.append(predictors(past));yy.append(profiles[(origin+1,r)]-past[-1]);origins.append(origin)
        if not xx:continue
        x=np.vstack(xx);y=np.concatenate(yy);scale=np.maximum(np.std(x,axis=0),.01);xs=x/scale
        coef=np.linalg.solve(xs.T@xs+20*np.eye(3),xs.T@y)/scale
        fitted=x@coef;r2=1-np.sum((y-fitted)**2)/max(np.sum(y*y),1e-8)
        coefficients.append({'origin':int(year),'toward_recent_mean':coef[0],'past_slope':coef[1],'last_change':coef[2],
                             'last_training_profile':int(max(origins)+1),'past_role_transitions':len(origins),'training_r2':r2})
        for r in ROLES:
            if (int(year),r) not in profiles:continue
            h=[profiles[(s,r)] for s in range(max(2015,int(year)-3),int(year)+1) if (s,r) in profiles]
            if len(h)<2:continue
            mu=h[-1].copy();forecast_history=[a.copy() for a in h]
            for _ in range(3):
                step=predictors(forecast_history)@coef
                forecast_history.append(forecast_history[-1]+np.clip(step,-.25,.25))
            future=forecast_history[-1];change=future-mu
            loc=np.flatnonzero(raw.edition.eq(year)&raw.position_group.eq(r));a=z[loc];an=np.maximum(np.linalg.norm(a,axis=1),1e-8)
            distance=np.sqrt(np.mean((a-mu)**2,axis=1));fd=np.sqrt(np.mean((a-future)**2,axis=1))
            vals=[(a@future)/(an*max(np.linalg.norm(future),1e-8)),fd,distance-fd,
                  (a@change)/(an*max(np.linalg.norm(change),1e-8)),np.sqrt(np.mean(change*change)),
                  r2,len(origins),1.]
            for c,v in zip(LEARNED,vals):values[c][loc]=v
            audit.append({'origin':int(year),'role':r,'last_training_profile':int(max(origins)+1),'first_training_origin':int(min(origins)),
                          'last_training_origin':int(max(origins)),'forecast_edition':int(year)+3})
            if (int(year)+3,r) in profiles:
                actual=profiles[(int(year)+3,r)];a0=np.sqrt(np.mean((actual-mu)**2));a1=np.sqrt(np.mean((actual-future)**2))
                scores.append({'origin':int(year),'role':r,'target_edition':int(year)+3,'static_rmse':a0,'trend_rmse':a1,'improvement':a0-a1})
    return pd.DataFrame(values,index=raw.index).astype('float32'),pd.DataFrame(audit),pd.DataFrame(scores),pd.DataFrame(coefficients)

def add_role_features(raw,z,profiles,support):
    values={c:np.full(len(raw),np.nan) for c in STATIC+DYNAMIC}
    values['role_fit_available'][:]=0;values['role_trend_available'][:]=0
    audit=[];forecasts=[]
    for (year,role),mu in profiles.items():
        loc=np.flatnonzero(raw.edition.eq(year)&raw.position_group.eq(role));a=z[loc];an=np.linalg.norm(a,axis=1)
        distance=np.sqrt(np.mean((a-mu)**2,axis=1))
        similarity=(a@mu)/(np.maximum(an,1e-8)*max(np.linalg.norm(mu),1e-8))
        values['role_fit_now'][loc]=similarity;values['role_distance_now'][loc]=distance
        values['role_fit_available'][loc]=1;values['role_reference_cells'][loc]=support[(year,role)]['cells']
        years=[y for y in range(max(2015,year-3),year+1) if (y,role) in profiles]
        if len(years)<3:continue
        history=np.vstack([profiles[(y,role)] for y in years]);t=np.asarray(years,float);t-=t.mean()
        slope=(t[:,None]*(history-history.mean(axis=0))).sum(axis=0)/(t@t)
        predicted=history.mean(axis=0)+t[:,None]*slope
        coherence=1-np.sum((history-predicted)**2)/max(np.sum((history-history.mean(axis=0))**2),1e-8)
        # Fixed half-strength extrapolation. This is specified before viewing the scouting scores.
        forecast=mu+1.5*slope
        fd=np.sqrt(np.mean((a-forecast)**2,axis=1));fn=max(np.linalg.norm(forecast),1e-8)
        values['role_fit_forecast'][loc]=(a@forecast)/(np.maximum(an,1e-8)*fn)
        values['role_distance_forecast'][loc]=fd
        values['future_fit_advantage'][loc]=distance-fd
        values['trend_alignment'][loc]=(a@slope)/(np.maximum(an,1e-8)*max(np.linalg.norm(slope),1e-8))
        values['role_trend_speed'][loc]=np.linalg.norm(slope)/np.sqrt(z.shape[1])
        values['role_trend_coherence'][loc]=coherence;values['role_trend_years'][loc]=len(years);values['role_trend_available'][loc]=1
        audit.append({'origin':year,'role':role,'first_profile':min(years),'last_profile':max(years),'profile_years':len(years),'forecast_edition':year+3})
        if (year+3,role) in profiles:
            actual=profiles[(year+3,role)]
            static_err=np.sqrt(np.mean((actual-mu)**2));future_err=np.sqrt(np.mean((actual-forecast)**2))
            forecasts.append({'origin':year,'role':role,'target_edition':year+3,'static_rmse':static_err,'trend_rmse':future_err,'improvement':static_err-future_err})
    return pd.DataFrame(values,index=raw.index).astype('float32'),pd.DataFrame(audit),pd.DataFrame(forecasts)

def build_role_panel(raw):
    z,coords,normalizer=normalize_shapes(raw)
    profiles,support,history,cohorts=cohort_profiles(raw,z,normalizer['skills'])
    features,audit,forecast=add_role_features(raw,z,profiles,support)
    learned,learned_audit,learned_scores,coefficients=learn_profile_forecasts(raw,z,profiles)
    features=pd.concat([features,learned],axis=1)
    return {'z':z,'coordinates':coords,'normalizer':normalizer,'profiles':profiles,'support':support,'profile_history':history,
            'cohort_rows':cohorts,'features':features,'trend_audit':audit,'profile_forecasts':forecast,
            'learned_audit':learned_audit,'learned_forecasts':learned_scores,'forecast_coefficients':coefficients}

def validate_time(raw,role):
    truncated=raw.loc[raw.edition.le(2020)].reset_index(drop=True)
    small=build_role_panel(truncated)
    ids=raw.loc[raw.edition.eq(2020),'row_id']
    a=role['features'].loc[raw.edition.eq(2020)].set_axis(ids.to_numpy())
    b=small['features'].loc[truncated.edition.eq(2020)].set_axis(truncated.loc[truncated.edition.eq(2020),'row_id'].to_numpy())
    pd.testing.assert_frame_equal(a.sort_index(),b.sort_index(),check_dtype=False,atol=1e-6,rtol=1e-6)
    assert role['trend_audit'].last_profile.le(role['trend_audit'].origin).all()
    assert role['normalizer']['normalizer_fit_end']==2015
    assert role['learned_audit'].last_training_profile.le(role['learned_audit'].origin).all()
    return {'future_removal_invariance':True,'normalizer_and_geometry_fit_2015':True,'role_profiles_known_at_origin':True,
            'candidates_excluded_from_prototypes':True}

def profile_shift_bootstrap(role,draws=400):
    f=role['cohort_rows'];f=f.loc[f.edition.isin([2015,2023])].copy();cols=[c for c in f if c.startswith('z')]
    f['_cell']=f._league_id.astype(int).astype(str)+'_'+f._age_bin.astype(str)
    output=[];rng=np.random.default_rng(79)
    for r in ROLES:
        g=f.loc[f.position_group.eq(r)].copy();groups=[(y,b,c) for y in [2015,2023] for b in ['elite','reference'] for c in g._cell.unique()]
        common=[c for c in g._cell.unique() if all(len(g.loc[g.edition.eq(y)&g._band.eq(b)&g._cell.eq(c)])>=2 for y in [2015,2023] for b in ['elite','reference'])]
        if not common:continue
        ids,iv=np.unique(g.player_id,return_inverse=True);arrays=g[cols].to_numpy();masks={(y,b,c):np.flatnonzero(g.edition.eq(y)&g._band.eq(b)&g._cell.eq(c)) for y,b,c in groups if c in common}
        def difference(w):
            means={}
            for y in [2015,2023]:
                pairs=[]
                for c in common:
                    a=masks[(y,'elite',c)];b=masks[(y,'reference',c)]
                    if w[a].sum()==0 or w[b].sum()==0:continue
                    pairs.append(np.average(arrays[a],axis=0,weights=w[a])-np.average(arrays[b],axis=0,weights=w[b]))
                means[y]=np.mean(pairs,axis=0)
            return means[2023]-means[2015]
        observed=difference(np.ones(len(g)));boot=[]
        for _ in range(draws):
            w=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))[iv];boot.append(difference(w))
        boot=np.asarray(boot);se=np.std(boot,axis=0,ddof=1)
        from scipy.stats import norm
        for j,skill in enumerate(role['normalizer']['skills']):
            output.append({'role':r,'skill':skill,'shift':observed[j],'low':np.quantile(boot[:,j],.025),'high':np.quantile(boot[:,j],.975),
                           'p_approx':2*norm.sf(abs(observed[j])/max(se[j],1e-8)),'matched_cells':len(common),'players':len(ids)})
    out=pd.DataFrame(output);p=out.p_approx.to_numpy();order=np.argsort(p);q=np.empty(len(p));q[order]=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];out['q_bh']=np.minimum(q,1.)
    return out
