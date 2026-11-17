from pathlib import Path
import argparse,io,re,tokenize
import nbformat as nbf

ROOT=Path(__file__).parent

def inline(name):
    source=(ROOT/name).read_text().split("\nif __name__=='__main__':")[0]
    names=['ratings_loader','strong_features','role_trends','future_scout','future_story','future_room','feature_evidence','story_sections','final_watchlist']
    prefixes=tuple('from '+n+' import' for n in names)
    source='\n'.join(line for line in source.splitlines() if not line.startswith(prefixes))
    return tokenize.untokenize([t for t in tokenize.generate_tokens(io.StringIO(source).readline) if t.type!=tokenize.COMMENT])

def build(story_path=None,clear_outputs=False):
    story_path=Path(story_path) if story_path else ROOT/'Beyond_FIFA_Potential_Story.md'
    story=story_path.read_text();blocks={m.group(1):m.group(2).strip() for m in re.finditer(r'<!-- story:([a-z_]+) -->\s*([\s\S]*?)(?=<!-- story:|\Z)',story)}
    path=ROOT/'FIFA_Beyond_Potential_Hidden_Gems.ipynb'
    old=nbf.read(path,4) if path.exists() and not clear_outputs else None
    old_codes={c.source:c for c in old.cells if c.cell_type=='code'} if old else {}
    cells=[]
    def md(key):
        cell=nbf.v4.new_markdown_cell(blocks[key]);cell.metadata={'story_key':key,'tags':['editable-story']};cells.append(cell)
    def code(source):
        cell=nbf.v4.new_code_cell(source);cell.metadata={'jupyter':{'source_hidden':True},'collapsed':True,'hide_input':True,'tags':['hide-input']}
        if source in old_codes:cell.outputs=old_codes[source].outputs;cell.execution_count=old_codes[source].execution_count
        cells.append(cell)
    md('intro')
    files=['ratings_loader.py','strong_features.py','role_trends.py','future_scout.py','future_story.py','future_room.py','feature_evidence.py','story_sections.py','final_watchlist.py']
    implementation='ROOM_TEMPLATE='+repr((ROOT/'scout_room.html').read_text())+'\nROOM_LOGIC='+repr((ROOT/'scout_room.js').read_text())+'\n\n'+'\n\n'.join(inline(n) for n in files)
    code(implementation)
    code("""from IPython.display import display,HTML,Image
from contextlib import redirect_stdout,redirect_stderr
import io
display(HTML(READER_CSS))
execution_log=io.StringIO()
reuse=os.environ.get('FUTURE_REUSE','0')=='1'
with redirect_stdout(execution_log),redirect_stderr(execution_log):
    study=execute_future_study(reuse=reuse)
    study=add_feature_evidence(study,reuse=reuse)
    study=build_final_watchlist(study,top_n=20)
    room=export_room(study,reuse=reuse)
log_bytes=(OUT/'execution_log.txt').write_text(execution_log.getvalue())
""")
    sections=[('question',[]),('timing',['plot_decision_clock(study)']),('comparison',['display(HTML(comparison_table(study)))','plot_model_comparison(study)','display(HTML(annual_table(study)))']),('uncertainty',['plot_sampling_uncertainty(study)']),('removal',['plot_feature_removal(study)']),('features',['plot_feature_shap(study)','plot_feature_patterns(study)']),('ranks',['plot_discovery_ranks(study)']),('similarity',['plot_similarity(study)']),('explorer',['display(HTML(reading_room(room)))']),('cases',['display(HTML(case_comparison_html(study)))','plot_careers(study)','plot_case_shap(study)']),('selection',['display(HTML(selection_summary_html(study)))']),('findings',['display(HTML(methods_html(study)))']),('watchlist',['plot_current_priorities(study)','plot_current_player_reasons(study)','display(HTML(current_shortlist_html(study)))','display(HTML(newer_source_audit_html()))'])]
    for key,operations in sections:
        md(key)
        if operations:code('\n'.join(operations))
    nb=nbf.v4.new_notebook(cells=cells)
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'kaggle':{'accelerator':'none','dataSources':[{'sourceType':'datasetVersion','sourceId':6730330},{'sourceType':'datasetVersion','sourceId':9484665},{'sourceType':'datasetVersion','sourceId':13139703}],'isInternetEnabled':False,'isGpuEnabled':False,'language':'python','sourceType':'notebook'},'authoring':{'story_cells':'All commentary is stored in ordinary Markdown cells. Model execution does not replace it.','story_source':story_path.name,'html_export':'python export_article.py saved_notebook.ipynb --output article.html'}}
    nbf.validate(nb);nbf.write(nb,path);return path

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Build the notebook from editable story blocks and preserve unchanged code outputs.')
    p.add_argument('--story');p.add_argument('--clear-outputs',action='store_true')
    a=p.parse_args();print(build(a.story,a.clear_outputs))
