from pathlib import Path
import os
import numpy as np
import pandas as pd
INPUT_ROOT=Path(os.environ.get('FIFA_INPUT_ROOT','/kaggle/input' if Path('/kaggle/input').exists() else 'data'))
SKILLS = [
    'attacking_crossing','attacking_finishing','attacking_heading_accuracy','attacking_short_passing','attacking_volleys',
    'skill_dribbling','skill_curve','skill_fk_accuracy','skill_long_passing','skill_ball_control',
    'movement_acceleration','movement_sprint_speed','movement_agility','movement_reactions','movement_balance',
    'power_shot_power','power_jumping','power_stamina','power_strength','power_long_shots',
    'mentality_aggression','mentality_interceptions','mentality_positioning','mentality_vision','mentality_penalties','mentality_composure',
    'defending_marking_awareness','defending_standing_tackle','defending_sliding_tackle',
    'goalkeeping_diving','goalkeeping_handling','goalkeeping_kicking','goalkeeping_positioning','goalkeeping_reflexes',
]

NUMERIC = ['overall','potential','age','height_cm','weight_kg','weak_foot','skill_moves','international_reputation',
           'pace','shooting','passing','dribbling','defending','physic'] + SKILLS

def locate(name,hint):
    found=[p for p in INPUT_ROOT.rglob(name) if hint in str(p)]
    if len(found)!=1: raise FileNotFoundError(f'Attach {hint}: expected one {name}, found {len(found)} beneath {INPUT_ROOT}.')
    return found[0]

def load_ratings():
    base_path = locate('male_players.csv', 'ea-sports-fc-24-complete-player-dataset')
    fc25_path = locate('male_players.csv', 'ea-sports-fc-25-database-ratings-and-stats')
    fc26_path = locate('FC26_20250921.csv', 'fc-26-fifa-26-player-data')
    base = pd.read_csv(base_path, low_memory=False)
    base['edition'] = 2000 + pd.to_numeric(base['fifa_version']).astype(int)
    base = base.sort_values(['edition','fifa_update','player_id']).drop_duplicates(['player_id','edition'])
    fc26 = pd.read_csv(fc26_path, low_memory=False)
    fc26['edition'] = 2026
    fc25 = pd.read_csv(fc25_path, low_memory=False)
    names = {
        'Name':'short_name','OVR':'overall','PAC':'pace','SHO':'shooting','PAS':'passing','DRI':'dribbling','DEF':'defending','PHY':'physic',
        'Age':'age','Position':'player_positions','Weak foot':'weak_foot','Skill moves':'skill_moves','Preferred foot':'preferred_foot',
        'Team':'club_name','League':'league_name','Nation':'nationality_name',
        'Acceleration':'movement_acceleration','Sprint Speed':'movement_sprint_speed','Positioning':'mentality_positioning',
        'Finishing':'attacking_finishing','Shot Power':'power_shot_power','Long Shots':'power_long_shots','Volleys':'attacking_volleys',
        'Penalties':'mentality_penalties','Vision':'mentality_vision','Crossing':'attacking_crossing','Free Kick Accuracy':'skill_fk_accuracy',
        'Short Passing':'attacking_short_passing','Long Passing':'skill_long_passing','Curve':'skill_curve','Dribbling':'skill_dribbling',
        'Agility':'movement_agility','Balance':'movement_balance','Reactions':'movement_reactions','Ball Control':'skill_ball_control',
        'Composure':'mentality_composure','Interceptions':'mentality_interceptions','Heading Accuracy':'attacking_heading_accuracy',
        'Def Awareness':'defending_marking_awareness','Standing Tackle':'defending_standing_tackle','Sliding Tackle':'defending_sliding_tackle',
        'Jumping':'power_jumping','Stamina':'power_stamina','Strength':'power_strength','Aggression':'mentality_aggression',
        'GK Diving':'goalkeeping_diving','GK Handling':'goalkeeping_handling','GK Kicking':'goalkeeping_kicking',
        'GK Positioning':'goalkeeping_positioning','GK Reflexes':'goalkeeping_reflexes',
    }
    fc25 = fc25.rename(columns=names)
    fc25['player_id'] = pd.to_numeric(fc25['url'].str.extract(r'/([0-9]+)/?$')[0], errors='raise').astype(int)
    assert fc25.player_id.is_unique, 'Duplicate FC25 player IDs need review.'
    fc25['edition'] = 2025
    fc25['long_name'] = fc25.short_name
    fc25['height_cm'] = pd.to_numeric(fc25['Height'].str.extract(r'^(\d+)')[0], errors='coerce')
    fc25['weight_kg'] = pd.to_numeric(fc25['Weight'].str.extract(r'^(\d+)')[0], errors='coerce')
    keep = ['player_id','edition','short_name','long_name','player_positions','preferred_foot','club_name','league_name','nationality_name','club_team_id','league_id','league_level','club_position','club_joined_date','club_loaned_from','club_contract_valid_until_year'] + NUMERIC
    frames, audit = [], []
    for frame, source in [(base,'FIFA15–FC24 / SoFIFA'),(fc25,'FC25 / EA ratings'),(fc26,'FC26 / SoFIFA')]:
        for c in keep:
            if c not in frame:
                frame[c] = np.nan
        for c in NUMERIC:
            frame[c] = pd.to_numeric(frame[c], errors='coerce')
        frame['player_id'] = pd.to_numeric(frame.player_id, errors='raise').astype(int)
        frame['source'] = source
        for year, group in frame.groupby('edition'):
            audit.append({'edition':int(year),'players':len(group),'source':source,
                          'mean_skill_fields_present':group[SKILLS].notna().sum(axis=1).mean(),
                          'potential_present':group.potential.notna().mean()})
        frames.append(frame[list(dict.fromkeys(keep + ['source']))])
    panel = pd.concat(frames, ignore_index=True).sort_values(['player_id','edition']).reset_index(drop=True)
    assert not panel.duplicated(['player_id','edition']).any()
    assert panel.overall.between(1,99).all()
    assert sorted(panel.edition.unique()) == list(range(2015,2027)), 'The study needs all twelve editions, 2015–2026.'
    fc24_ids = set(base.loc[base.edition.eq(2024),'player_id'])
    common25 = set(fc25.player_id).intersection(fc24_ids)
    assert len(common25) / len(fc25) > 0.5, 'FC25 IDs do not align with historical FIFA IDs.'
    overlapping = base.loc[base.edition.eq(2024),['player_id','overall']].merge(fc25[['player_id','overall']],on='player_id')
    assert overlapping[['overall_x','overall_y']].corr().iloc[0,1] > 0.75, 'Check the cross-source identity bridge.'
    panel['primary_position'] = panel.player_positions.fillna('Unknown').str.split(',').str[0].str.strip()
    positions = {'GK':'GK','CB':'CB','LB':'FB','RB':'FB','LWB':'FB','RWB':'FB','CDM':'MF','CM':'MF','CAM':'MF',
                 'LM':'Wing','RM':'Wing','LW':'Wing','RW':'Wing','ST':'FW','CF':'FW'}
    panel['position_group'] = panel.primary_position.map(positions).fillna('Other')
    panel['row_id'] = np.arange(len(panel))
    return panel, pd.DataFrame(audit), [base_path,fc25_path,fc26_path]

ROLE_CORES={
    'FW':['attacking_finishing','mentality_positioning','skill_ball_control','movement_reactions','power_shot_power'],
    'Wing':['movement_acceleration','movement_sprint_speed','skill_dribbling','skill_ball_control','mentality_vision'],
    'MF':['attacking_short_passing','skill_long_passing','mentality_vision','skill_ball_control','movement_reactions'],
    'CB':['defending_marking_awareness','defending_standing_tackle','mentality_interceptions','power_strength','movement_reactions'],
    'FB':['power_stamina','attacking_crossing','movement_acceleration','movement_sprint_speed','defending_standing_tackle'],
    'GK':['goalkeeping_diving','goalkeeping_handling','goalkeeping_kicking','goalkeeping_positioning','goalkeeping_reflexes'],
    'Other':['skill_ball_control','movement_reactions','power_stamina'],
}

TECHNICAL=['attacking_short_passing','skill_ball_control','skill_dribbling','mentality_vision']

ATHLETIC=['movement_acceleration','movement_sprint_speed','movement_agility','power_stamina']

MENTAL=['movement_reactions','mentality_composure','mentality_positioning','mentality_vision']
