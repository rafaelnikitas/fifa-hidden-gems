from ratings_loader import *
import warnings
from types import SimpleNamespace
CFG=SimpleNamespace(min_age=16,max_age=99,training_max_age=99,rating_cap=69,star_rating=85)
def strong_player_engineering(raw):
    p=raw.sort_values(['player_id','edition']).reset_index(drop=True).copy()
    p['peak_seen']=p.groupby('player_id').overall.cummax()
    p['is_candidate']=p.age.between(CFG.min_age,CFG.max_age)&p.overall.le(CFG.rating_cap)&p.peak_seen.le(CFG.rating_cap)
    p['is_training']=p.age.between(CFG.min_age,CFG.training_max_age)&p.overall.lt(CFG.star_rating)&p.peak_seen.lt(CFG.star_rating)
    p['role_quality']=np.nan
    for role,cols in ROLE_CORES.items():
        mask=p.position_group.eq(role);p.loc[mask,'role_quality']=p.loc[mask,cols].mean(axis=1)
    for name,cols in [('technical',TECHNICAL),('athletic',ATHLETIC),('mental',MENTAL)]:
        p[name+'_quality']=p[cols].mean(axis=1)
        p[name+'_headroom']=p[name+'_quality']-p.overall
    p['role_headroom']=p.role_quality-p.overall
    p['skill_spikiness']=p[SKILLS].quantile(.9,axis=1)-p[SKILLS].median(axis=1)
    p['attributes_75plus']=p[SKILLS].ge(75).sum(axis=1)
    p['attributes_80plus']=p[SKILLS].ge(80).sum(axis=1)
    p['weakest_role_skill']=np.nan
    for role,cols in ROLE_CORES.items():
        mask=p.position_group.eq(role);p.loc[mask,'weakest_role_skill']=p.loc[mask,cols].min(axis=1)
    p['role_balance']=p.weakest_role_skill-p.role_quality
    p['age_to_21']=21-p.age
    p['age_rating_interaction']=(21-p.age)*p.overall/100
    for role in ROLE_CORES:p['role_'+role]=p.position_group.eq(role).astype(int)
    p['right_foot']=p.preferred_foot.str.lower().eq('right').astype(int)
    p['age_band']=np.floor(p.age/2)*2
    peers=p.groupby(['edition','position_group','age_band'],observed=True)
    for c in ['role_quality','technical_quality','athletic_quality','mental_quality']:
        # Ranks use the full contemporaneous player pool, without future labels.
        p[c+'_peer_percentile']=peers[c].rank(pct=True)
    tracked=['overall','role_quality','technical_quality','athletic_quality','mental_quality',
             'movement_reactions','skill_ball_control','power_stamina']
    index=p.set_index(['player_id','edition'])
    for lag in [1,2]:
        keys=pd.MultiIndex.from_arrays([p.player_id,p.edition-lag],names=['player_id','edition'])
        values=index[tracked].reindex(keys)
        for c in tracked:
            p[c+f'_lag{lag}']=values[c].to_numpy()
            p[c+f'_change{lag}']=p[c]-p[c+f'_lag{lag}']
    p['history_length']=1+p.overall_lag1.notna().astype(int)+(p.overall_lag1.notna()&p.overall_lag2.notna()).astype(int)
    p['rating_acceleration']=p.overall_change1-(p.overall_lag1-p.overall_lag2)
    p['core_growth_ahead_of_rating']=p.role_quality_change1-p.overall_change1
    p['technical_growth_ahead_of_rating']=p.technical_quality_change1-p.overall_change1
    p['potential_gap']=p.potential-p.overall
    p['under_radar']=p.potential.lt(80)
    current=['overall','age','height_cm','weight_kg','weak_foot','skill_moves','pace','shooting','passing','dribbling','defending','physic']+SKILLS+['right_foot']+['role_'+r for r in ROLE_CORES]
    engineered=['role_quality','technical_quality','athletic_quality','mental_quality','role_headroom','technical_headroom',
        'athletic_headroom','mental_headroom','skill_spikiness','attributes_75plus','attributes_80plus','role_balance',
        'weakest_role_skill','age_to_21','age_rating_interaction']+[c+'_peer_percentile' for c in ['role_quality','technical_quality','athletic_quality','mental_quality']]
    history=[c+suffix for c in tracked for suffix in ['_lag1','_change1','_lag2','_change2']]+['history_length','rating_acceleration','core_growth_ahead_of_rating','technical_growth_ahead_of_rating']
    sets={'Snapshot skills':current,'Hidden-gem features':current+engineered+history,
          'Skills + EA potential':current+engineered+history+['potential','potential_gap']}
    for name,cols in sets.items():
        assert len(cols)==len(set(cols))
        assert not any(c.startswith('future') or c in ['player_id','edition','short_name','club_name','league_name'] for c in cols)
        if name!='Skills + EA potential':assert not any('potential' in c for c in cols)
    return p,sets

def strong_current_context(raw):
    p, old_sets = strong_player_engineering(raw)
    # No age ceiling. A previously established 85+ player is not a new discovery.
    p['eligible'] = p.overall.lt(70) & p.peak_seen.lt(85)
    p['training_pool'] = p.overall.lt(85) & p.peak_seen.lt(85)
    club = p.club_name.fillna('').astype(str).str.strip().str.casefold()
    league = p.league_name.fillna('').astype(str).str.strip().str.casefold()
    p['_club'] = club.where(club.ne(''), 'missing_player_' + p.player_id.astype(str))
    p['_league'] = league.where(league.ne(''), 'missing_league_' + p.player_id.astype(str))
    valid_club = club.ne('') & league.ne('')
    keys = ['edition', '_league', '_club']
    grouped = p.groupby(keys, observed=True, sort=False)
    teams = grouped.agg(team_roster_n=('overall', 'size'), team_mean_rating=('overall', 'mean'),
        team_median_rating=('overall', 'median'), team_rating_std=('overall', 'std'),
        team_best_rating=('overall', 'max'), team_mean_age=('age', 'mean'))
    top11 = p.sort_values('overall', ascending=False).groupby(keys, observed=True, sort=False).head(11)
    teams['team_top11_rating'] = top11.groupby(keys, observed=True).overall.mean()
    teams['team_u21_share'] = p.assign(_u21=p.age.le(21).astype(float)).groupby(keys, observed=True)._u21.mean()
    teams['team_80plus_count'] = p.assign(_elite=p.overall.ge(80).astype(float)).groupby(keys, observed=True)._elite.sum()
    teams = teams.reset_index()
    leagues = teams.groupby(['edition', '_league'], observed=True)
    teams['team_strength_league_percentile'] = leagues.team_top11_rating.rank(pct=True, method='average')
    teams['league_team_count'] = leagues.team_top11_rating.transform('size')
    teams['league_mean_team_strength'] = leagues.team_top11_rating.transform('mean')
    teams['league_best_team_strength'] = leagues.team_top11_rating.transform('max')
    teams['team_strength_vs_league'] = teams.team_top11_rating - teams.league_mean_team_strength
    teams['team_strength_vs_league_best'] = teams.team_top11_rating - teams.league_best_team_strength
    teams['team_strength_global_percentile'] = teams.groupby('edition').team_top11_rating.rank(pct=True)
    p = p.merge(teams, on=keys, how='left', validate='many_to_one', sort=False)
    p = p.sort_values(['player_id', 'edition']).reset_index(drop=True)
    context = list(teams.columns.difference(keys, sort=False))
    p['rating_vs_team_top11'] = p.overall - p.team_top11_rating
    p['rating_vs_team_mean'] = p.overall - p.team_mean_rating
    p['age_vs_team_mean'] = p.age - p.team_mean_age
    p['team_roster_log_n'] = np.log1p(p.team_roster_n)
    p['league_level_numeric'] = pd.to_numeric(p.league_level, errors='coerce')
    p['previous_peak_gap'] = p.peak_seen - p.overall
    p['first_observed_age'] = p.groupby('player_id').age.transform('first')
    p['editions_since_first_seen'] = p.edition - p.groupby('player_id').edition.transform('first')
    context += ['rating_vs_team_top11', 'rating_vs_team_mean', 'age_vs_team_mean', 'team_roster_log_n',
                'league_level_numeric', 'previous_peak_gap', 'first_observed_age', 'editions_since_first_seen']
    rating_fields = list(dict.fromkeys(['overall', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'role_quality', 'technical_quality', 'athletic_quality', 'mental_quality'] + SKILLS))
    # Higher percentile means a higher attribute; average ties, no player-ID ordering.
    comparisons = [
        ('team', ['edition', '_league', '_club'], rating_fields),
        ('position', ['edition', 'primary_position'], rating_fields),
        ('position_age', ['edition', 'primary_position', 'age'], rating_fields),
        ('team_role', ['edition', '_league', '_club', 'position_group'],
            ['overall', 'role_quality', 'technical_quality', 'athletic_quality', 'mental_quality', 'movement_reactions']),
        ('league_position', ['edition', '_league', 'primary_position'],
            ['overall', 'role_quality', 'technical_quality', 'athletic_quality', 'mental_quality', 'pace']),
        ('league_position_age', ['edition', '_league', 'primary_position', 'age'],
            ['overall', 'role_quality', 'technical_quality', 'athletic_quality', 'mental_quality', 'pace']),
    ]
    for prefix, by, columns in comparisons:
        g = p.groupby(by, observed=True, sort=False)
        group_n = g.overall.transform('size')
        p[prefix + '_peer_log_n'] = np.log1p(group_n)
        context.append(prefix + '_peer_log_n')
        for c in columns:
            counts = g[c].transform('count')
            ranks = g[c].rank(method='average', pct=False)
            # Shrink tiny peer groups toward 0.5 rather than claiming a 1/1 leader.
            percentile = (ranks - .5) / counts
            p[prefix + '_pct_' + c] = .5 + (percentile - .5) * counts / (counts + 5)
            context.append(prefix + '_pct_' + c)
            if prefix in ['team', 'team_role']:
                sums = g[c].transform('sum')
                peer_mean = (sums - p[c]) / (counts - p[c].notna().astype(int)).replace(0, np.nan)
                p[prefix + '_gap_' + c] = p[c] - peer_mean
                context.append(prefix + '_gap_' + c)
        if prefix == 'team_role':
            p['team_role_rating_depth_gap'] = p.overall - g.overall.transform('max')
            context.append('team_role_rating_depth_gap')
    p['team_skill_above_rating_rank'] = p.team_pct_role_quality - p.team_pct_overall
    p['position_age_skill_above_rating_rank'] = p.position_age_pct_role_quality - p.position_age_pct_overall
    context += ['team_skill_above_rating_rank', 'position_age_skill_above_rating_rank']
    lineup = p.club_position.fillna('').astype(str)
    for name, mask in [('named_starter', ~lineup.isin(['SUB', 'RES', ''])),
                       ('substitute', lineup.eq('SUB')), ('reserve', lineup.eq('RES'))]:
        p['club_' + name] = mask.astype(float).where(lineup.ne(''))
        context.append('club_' + name)
    # A missing club ID (FC25) does not manufacture a transfer.
    index = p.set_index(['player_id', 'edition'])
    tracking = ['team_top11_rating', 'team_strength_league_percentile', 'league_mean_team_strength',
        'rating_vs_team_mean', 'team_pct_overall', 'team_pct_role_quality', 'team_role_pct_role_quality',
        'position_age_pct_role_quality', 'team_role_rating_depth_gap']
    for lag in [1, 2]:
        lookup = pd.MultiIndex.from_arrays([p.player_id, p.edition-lag], names=['player_id', 'edition'])
        historical = index[tracking + ['club_team_id', 'league_id']].reindex(lookup)
        for c in tracking:
            p[c + f'_context_change{lag}'] = p[c].to_numpy() - historical[c].to_numpy()
            context.append(c + f'_context_change{lag}')
        for name in ['club_team_id', 'league_id']:
            now = pd.to_numeric(p[name], errors='coerce').to_numpy()
            then = pd.to_numeric(historical[name], errors='coerce').to_numpy()
            p[name + f'_changed{lag}'] = np.where(np.isfinite(now) & np.isfinite(then), (now != then).astype(float), np.nan)
            context.append(name + f'_changed{lag}')
    context = list(dict.fromkeys(context))
    club_features = [c for c in context if c.startswith(('team_', 'rating_vs_team', 'age_vs_team', 'club_', 'league_'))]
    p.loc[~valid_club.to_numpy(), club_features] = np.nan
    base_features = old_sets['Hidden-gem features']
    sets = {'Skills only': base_features, 'Skills + context': base_features + context,
            'Context + EA': base_features + context + ['potential', 'potential_gap']}
    for name, cols in sets.items():
        assert len(cols) == len(set(cols))
        assert not set(cols).intersection(['player_id', 'row_id', 'edition', 'club_name', 'league_name', 'short_name'])
        if name != 'Context + EA':
            assert not any('potential' in c for c in cols)
    p[list(dict.fromkeys(sum(sets.values(), [])))] = p[list(dict.fromkeys(sum(sets.values(), [])))].astype('float32')
    return p, sets, context

def strong_history_panel(raw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
        p, old_sets, context = strong_current_context(raw)
    p = p.copy()
    index = p.set_index(['player_id', 'edition'])
    new = {}
    tracked = ['overall', 'role_quality', 'technical_quality', 'athletic_quality', 'mental_quality',
               'team_top11_rating', 'movement_reactions', 'skill_ball_control', 'power_stamina']
    histories = {c: [p[c].to_numpy()] for c in tracked}
    for lag in range(1, 5):
        keys = pd.MultiIndex.from_arrays([p.player_id, p.edition-lag], names=['player_id', 'edition'])
        past = index[tracked].reindex(keys)
        for c in tracked:
            values = past[c].to_numpy()
            histories[c].append(values)
            if lag >= 3:
                new[c+f'_lag{lag}'] = values
                new[c+f'_change{lag}'] = p[c].to_numpy()-values
    for c, arrays in histories.items():
        values = np.column_stack(arrays)
        finite = np.isfinite(values)
        n = finite.sum(axis=1)
        t = -np.arange(5, dtype=float)[None, :]
        sx = np.where(finite, t, 0).sum(axis=1)
        sy = np.where(finite, values, 0).sum(axis=1)
        sxx = np.where(finite, t*t, 0).sum(axis=1)
        sxy = np.where(finite, t*values, 0).sum(axis=1)
        denominator = n*sxx-sx*sx
        new[c+'_past_slope'] = np.divide(n*sxy-sx*sy, denominator,
            out=np.full(len(p), np.nan), where=denominator>0)
        new[c+'_past_observations'] = n
    ratings = np.column_stack(histories['overall'])
    new['rating_below_recent_peak'] = np.nanmax(ratings, axis=1)-p.overall.to_numpy()
    new['rating_above_recent_floor'] = p.overall.to_numpy()-np.nanmin(ratings, axis=1)
    new['past_growth_skill_ahead'] = new['role_quality_past_slope']-new['overall_past_slope']
    new['recent_growth_consistency'] = np.divide(
        (np.diff(ratings, axis=1)<0).sum(axis=1),
        np.isfinite(np.diff(ratings, axis=1)).sum(axis=1),
        out=np.full(len(p), np.nan), where=np.isfinite(np.diff(ratings, axis=1)).sum(axis=1)>0)
    p = pd.concat([p, pd.DataFrame(new, index=p.index).astype('float32')], axis=1)
    p['eligible'] = p.overall.lt(70) & p.peak_seen.lt(85)
    p['fit_domain'] = p.overall.lt(80) & p.peak_seen.lt(85)
    p['country'] = p.nationality_name.fillna('Unknown').astype(str)
    for h in range(1, 4):
        keys = pd.MultiIndex.from_arrays([p.player_id, p.edition+h], names=['player_id', 'edition'])
        p[f'next_rating_{h}'] = index.overall.reindex(keys).to_numpy()
    future = p[[f'next_rating_{h}' for h in range(1,4)]]
    p['observed_peak3'] = future.max(axis=1)
    p['complete3'] = future.notna().all(axis=1)
    p['delta3_observed'] = p.observed_peak3-p.overall
    p['delta3'] = p.delta3_observed.where(p.complete3)
    p['outcome_end'] = p.edition+3
    sets = {'Pattern': old_sets['Skills + context']+list(new),
            'Pattern + FIFA': old_sets['Context + EA']+list(new)}
    for name, cols in sets.items():
        assert len(cols)==len(set(cols))
        assert not any(c.startswith(('future_', 'next_', 'delta', 'observed_peak')) for c in cols)
        assert not set(cols).intersection(['player_id','row_id','edition','country','nationality_name','short_name','club_name'])
        if name=='Pattern': assert not any('potential' in c for c in cols)
    return p, sets, context+list(new)
