from future_scout import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from IPython.display import display,Image
from sklearn.cluster import KMeans
import base64,html,gzip

INK='#173441';TEAL='#008777';GOLD='#c48b29';BLUE='#427fa3';RED='#bc625e';GREY='#a7b6bd';PALE='#e3eaed'
COLORS={FIFA:GOLD,CAL:GREY,BASE:INK,STATIC_MODEL:BLUE,LINEAR:'#b3a4bf',NEW:TEAL}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.13,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})


def nice(c):
    aliases={'overall':'Current overall','potential':'FIFA potential','potential_gap':'Potential headroom','age':'Age',
        'team_role_rating_depth_gap':'Gap to strongest same-role teammate','team_pct_role_quality':'Skill rank in club',
        'core_growth_ahead_of_rating':'Core skills growing ahead of rating','role_headroom':'Position skills above overall',
        'role_fit_now':'Current elite-profile similarity','role_distance_now':'Current elite-profile distance',
        'learned_future_fit':'Forecast profile similarity','learned_future_advantage':'Forecast fit advantage',
        'learned_future_distance':'Forecast profile distance','learned_future_alignment':'Alignment with forecast change',
        'learned_change_size':'Forecast change magnitude','learned_forecast_training_r2':'Historical forecast fit',
        'learned_forecast_transitions':'Past role transitions','learned_forecast_available':'Forecast available'}
    aliases.update({'position_age_pct_overall':'Rating rank · exact age + position','team_strength_global_percentile':'Club strength percentile',
        'team_top11_rating':'Club top-11 rating','team_median_rating':'Club median rating','league_best_team_strength':'Strongest club in league',
        'league_mean_team_strength':'Typical league club strength','club_named_starter':'Listed as starter',
        'role_quality_peer_percentile':'Role-skill rank · position + age band','mental_quality_peer_percentile':'Mental-skill rank · position + age band','height_cm':'Height',
        'position_age_pct_role_quality':'Role-skill rank · exact age + position','age_vs_team_mean':'Age relative to teammates',
        'position_pct_overall':'Rating rank within position','skill_fk_accuracy':'Free-kick accuracy','overall_past_slope':'Past rating trend',
        'team_roster_log_n':'Club squad size','age_rating_interaction':'Age and current rating'})
    if c in aliases:return aliases[c]
    return c.replace('attacking_','').replace('movement_','').replace('mentality_','').replace('defending_','').replace('skill_','').replace('power_','').replace('_',' ').capitalize()

def finish(fig,name,note=None):
    if note:fig.text(.015,.015,note,ha='left',va='bottom',fontsize=9,color='#637c88')
    fig.tight_layout(rect=[0,.06 if note else .01,1,.94]);fig.savefig(OUT/name,dpi=170,bbox_inches='tight')
    display(Image(filename=str(OUT/name)));plt.close(fig)

def plot_similarity(study):
    raw=study['raw'];role=study['role'];all_players=study['forecasts'].loc[study['forecasts'].edition.eq(2023)].copy()
    f=all_players.loc[all_players.position_group.ne('GK')].copy()
    co=role['coordinates'];rng=np.random.default_rng(SEED);sample=f.sample(min(1500,len(f)),random_state=SEED)
    fig,ax=plt.subplots(figsize=(12,6.4));xy=co[sample.row_id.to_numpy()];ax.scatter(xy[:,0],xy[:,1],s=8,color=GREY,alpha=.25,rasterized=True)
    hits=f.loc[f.event_quality80.eq(1)];xy=co[hits.row_id.to_numpy()];ax.scatter(xy[:,0],xy[:,1],s=45,color=TEAL,label='Eventually reaches 80+',zorder=3)
    ml=ordered(all_players,BASE).head(50).loc[lambda x:x.position_group.ne('GK')];xy=co[ml.row_id.to_numpy()];ax.scatter(xy[:,0],xy[:,1],s=75,facecolors='none',edgecolors=INK,lw=1.2,label='Player-model top 50',zorder=4)
    fifa=ordered(all_players,FIFA).head(50).loc[lambda x:x.position_group.ne('GK')];xy=co[fifa.row_id.to_numpy()];ax.scatter(xy[:,0],xy[:,1],s=100,facecolors='none',edgecolors=GOLD,lw=1,label='FIFA top 50',zorder=3)
    offsets=[(18,28),(-150,32),(25,-35),(-140,-35)]
    named=study['cases'].loc[study['cases'].position_group.ne('GK')]
    for (_,r),offset in zip(named.iterrows(),offsets):
        point=co[int(r.row_id)];color=TEAL if r.event_quality80==1 else RED
        ax.scatter(*point,s=65,color=color,zorder=6)
        ax.annotate(f'{r.short_name}\n{r.overall:.0f} → {r.observed_peak3:.0f} · ML #{r.ml_rank:.0f}',point,xytext=offset,textcoords='offset points',
                    fontsize=9,fontweight='bold',color=color,bbox=dict(boxstyle='round,pad=.35',facecolor='white',edgecolor=PALE,alpha=.95),
                    arrowprops=dict(arrowstyle='-',color=color,lw=1),zorder=7)
    ev=role['normalizer']['explained_variance'];ax.set_xlabel(f'Principal component 1 · {100*ev[0]:.1f}% of profile variation');ax.set_ylabel(f'Principal component 2 · {100*ev[1]:.1f}%')
    ax.legend(loc='upper right',fontsize=9,frameon=True,facecolor='white');ax.set_title('Similar profiles, different futures · the 2023 scouting pool',fontsize=18,fontweight='bold',pad=18)
    finish(fig,'08_player_similarity.png','Outfield projection of the all-position shortlists. Coordinates fitted in 2015; future colours are revealed after prediction. Neighbors use all 28 dimensions.')

def plot_careers(study):
    cases=study['cases'];raw=study['raw'];fig,axs=plt.subplots(1,len(cases),figsize=(14,4.8),sharey=True)
    for ax,(_,r) in zip(np.atleast_1d(axs),cases.iterrows()):
        g=raw.loc[raw.player_id.eq(r.player_id)].sort_values('edition');color=TEAL if r.event_quality80==1 else RED
        ax.plot(g.edition,g.overall,'o-',color=color,lw=2);ax.axvline(2023,color=INK,ls='--');ax.axhline(80,color=GOLD,ls=':',lw=1.5)
        ax.axvspan(2023.05,2026.1,color=PALE,alpha=.5);ax.set_xticks([2021,2023,2026]);ax.set_xlim(min(2021,g.edition.min())-.4,2026.5)
        ax.set_title(f'{r.short_name}\nOVR {r.overall:.0f} · POT {r.potential:.0f}',fontweight='bold',fontsize=11)
        ax.text(.04,.04,f'ML #{r.ml_rank:.0f} | FIFA #{r.fifa_rank:.0f}\n{r.case}',transform=ax.transAxes,fontsize=9,color=color)
    np.atleast_1d(axs)[0].set_ylabel('Observed game rating');np.atleast_1d(axs)[0].set_ylim(45,90)
    fig.suptitle('The career reveal includes discoveries, a false alarm and a miss',fontsize=18,fontweight='bold')
    finish(fig,'09_player_careers.png','All examples use 2023 information. Cases are selected after evaluation to explain results; full rankings and denominators remain available.')

def plot_case_shap(study):
    cases=study['cases'].head(3);allc=study['contributions'];fig,axs=plt.subplots(1,len(cases),figsize=(14,4.6))
    for ax,(_,r) in zip(np.atleast_1d(axs),cases.iterrows()):
        c=allc.loc[allc.player_id.eq(r.player_id)].copy();c=c.loc[c.shap.abs().nlargest(6).index].sort_values('shap')
        ax.barh([nice(x)[:38] for x in c.feature],c.shap,color=np.where(c.shap.ge(0),TEAL,RED));ax.axvline(0,color=GREY)
        ax.set_title(r.short_name,fontweight='bold');ax.set_xlabel('SHAP contribution')
    fig.suptitle('Why these players moved up the scouting list',fontsize=18,fontweight='bold')
    finish(fig,'10_player_explanations.png','Six largest individual feature attributions, not the complete score. Ranking scores are not calibrated probabilities of becoming a star.')

READER_CSS='<style>\n.code_cell>.input,.jp-Cell-inputWrapper{display:none!important}.output_prompt,.input_prompt{display:none!important}\n.code_cell:has(>.input):not(:has(.output_area)){display:none!important}\n.future-note{font:13px/1.65 system-ui;color:#637c88}.future-finding{border-left:4px solid #008777;background:#ecf7f3;padding:16px 20px;margin:18px 0;color:#173441;font:16px/1.65 system-ui}\n.future-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:13px;margin:19px 0}.future-cards>div{padding:18px;background:#f3f6f7;border-top:3px solid #008777;border-radius:8px}.future-cards b{display:block;font-size:30px;line-height:1.5}.future-cards span{font-size:12px;color:#637c88}\n.future-table{border-collapse:collapse;width:100%;font-size:13px;margin:14px 0}.future-table td,.future-table th{padding:10px;border-bottom:1px solid #dce5e9;text-align:left}.future-table th{background:#f1f5f6}.future-audit{overflow:auto;max-height:430px}\ndetails{padding:14px 18px;background:#f5f7f8;border:1px solid #dce5e9;margin:16px 0;border-radius:8px}summary{cursor:pointer;font-weight:600;color:#173441}\n@media(max-width:650px){.future-cards{grid-template-columns:1fr}}\n</style>'
