from pathlib import Path
import argparse,base64,html,json,re
import nbformat
import mistune

CSS='''body{margin:0;background:#f2f5f6;color:#173441;font:16px/1.75 system-ui,-apple-system,Arial,sans-serif}main{max-width:1140px;margin:28px auto;background:white;padding:28px 40px;border:1px solid #dce5e9;border-radius:10px}h1{font-size:36px;line-height:1.25;margin:12px 0 22px}h2{font-size:25px;line-height:1.35;margin:40px 0 16px}h3{font-size:20px}p{margin:13px 0}figure{margin:24px 0}figure img{width:100%;height:auto}a{color:#008777}table{border-collapse:collapse;width:100%;font-size:13px;margin:15px 0}td,th{padding:10px;border-bottom:1px solid #dce5e9;text-align:left;vertical-align:top}th{background:#f1f5f6}.future-table.latest-list{min-width:1020px;font-size:12px}.latest-list td:nth-child(2){min-width:140px}.latest-list td:nth-child(6){min-width:250px}.latest-list td:nth-child(7){min-width:200px}.future-note{font-size:13px;color:#637c88}.future-finding{border-left:4px solid #008777;background:#ecf7f3;padding:16px 20px;margin:18px 0}.future-audit{overflow:auto}details{padding:13px 17px;background:#f5f7f8;border:1px solid #dce5e9;margin:16px 0;border-radius:8px}summary{cursor:pointer;font-weight:600}iframe{max-width:100%}.toolbar{position:sticky;top:0;z-index:20;display:flex;gap:10px;align-items:center;padding:12px 20px;background:#173441;color:white;box-shadow:0 2px 10px #0002}.toolbar button{background:white;color:#173441;border:0;border-radius:5px;padding:8px 12px;cursor:pointer;font:600 13px system-ui}.toolbar small{margin-left:auto;color:#c2d9df}.story-prose[contenteditable=true]{outline:2px dashed #008777;outline-offset:6px;background:#fbfffd}.story-prose:focus{outline-style:solid}.computed{margin:18px 0}@media(max-width:650px){main{margin:10px;padding:17px}h1{font-size:28px}.toolbar{padding:9px}.toolbar small{display:none}}@media print{.toolbar{display:none}main{border:0;padding:0}}'''

EDITOR_JS='''let editing=false;const editButton=document.getElementById('edit-story');editButton.onclick=()=>{editing=!editing;document.querySelectorAll('.story-prose').forEach(e=>e.setAttribute('contenteditable',String(editing)));editButton.textContent=editing?'Finish editing':'Edit text';if(editing)document.querySelector('.story-prose').focus();};document.getElementById('save-story').onclick=()=>{editing=false;document.querySelectorAll('.story-prose').forEach(e=>e.setAttribute('contenteditable','false'));editButton.textContent='Edit text';const page='<!doctype html>'+document.documentElement.outerHTML;const url=URL.createObjectURL(new Blob([page],{type:'text/html;charset=utf-8'}));const link=document.createElement('a');link.href=url;link.download='Beyond_FIFA_Potential_Article_Edited.html';document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);};'''

def export_reader(notebook_path,output_path=None,story_path=None):
    notebook_path=Path(notebook_path);nb=nbformat.read(notebook_path,4)
    render=mistune.create_markdown(renderer=mistune.HTMLRenderer(escape=False),plugins=['table'])
    parts=[];stories=[];figures=0
    for i,cell in enumerate(nb.cells):
        if cell.cell_type=='markdown':
            key=cell.metadata.get('story_key','cell_'+str(i))
            stories.append('<!-- story:'+key+' -->\n'+cell.source.strip())
            parts.append('<section class="story-prose" data-story-key="'+html.escape(key)+'" contenteditable="false">'+render(cell.source)+'</section>')
        elif cell.cell_type=='code':
            for out in cell.get('outputs',[]):
                data=out.get('data',{})
                if 'image/png' in data:
                    figures+=1;parts.append('<figure class="computed"><img alt="Analysis figure '+str(figures)+'" src="data:image/png;base64,'+data['image/png']+'"></figure>')
                elif 'text/html' in data:
                    value=data['text/html']
                    if not value.strip() or value.lstrip().startswith('<style>'):continue
                    parts.append('<div class="computed">'+value+'</div>')
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Beyond FIFA potential: finding hidden gems</title><style>'+CSS+'</style></head><body><nav class="toolbar" aria-label="Article editing"><button id="edit-story">Edit text</button><button id="save-story">Save HTML</button><small>Story text is editable; charts and calculated tables stay fixed.</small></nav><main>'+''.join(parts)+'</main><script>'+EDITOR_JS+'</script></body></html>'
    output=Path(output_path) if output_path else notebook_path.with_name('Beyond_FIFA_Potential_Article.html')
    output.write_text(page,encoding='utf-8')
    if story_path is not None:Path(story_path).write_text('\n\n'.join(stories)+'\n',encoding='utf-8')
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Export the saved notebook with its current Markdown story and executed figures; no model fitting.')
    p.add_argument('notebook');p.add_argument('--output');p.add_argument('--story')
    args=p.parse_args();print(export_reader(args.notebook,args.output,args.story))
