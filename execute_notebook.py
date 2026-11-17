from pathlib import Path
import io,time
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

path=Path(__file__).parent/'FIFA_Beyond_Potential_Hidden_Gems.ipynb'
nb=nbformat.read(path,4);shell=InteractiveShell.instance();start=time.monotonic();n=0
for i,cell in enumerate(nb.cells):
    if cell.cell_type!='code':continue
    n+=1;print(f'RUN {n} · cell {i+1} · {time.monotonic()-start:.1f}s',flush=True)
    with capture_output(stdout=True,stderr=True,display=True) as captured:result=shell.run_cell(cell.source,store_history=False)
    cell.execution_count=n;outputs=[]
    if captured.stdout:outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr:outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for rich in captured.outputs:outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
    cell.outputs=outputs;error=result.error_before_exec or result.error_in_exec
    if error:
        cell.outputs.append(nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=[]))
        nbformat.write(nb,path);print(captured.stdout,captured.stderr,flush=True);raise error
    nbformat.write(nb,path)
nb.metadata['execution']={'runner':'IPython in process; every code cell in order','code_cells':n,'elapsed_seconds':round(time.monotonic()-start,2),'completed':True}
nbformat.validate(nb);nbformat.write(nb,path);print(f'PASS {n} code cells · {time.monotonic()-start:.1f}s · {path.stat().st_size:,} bytes',flush=True)

from export_article import export_reader
export_reader(path,path.parent/'results/Beyond_FIFA_Potential_Article.html',path.parent/'Beyond_FIFA_Potential_Story.md')
