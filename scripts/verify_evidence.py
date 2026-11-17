from pathlib import Path
import json
import nbformat
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
nb=nbformat.read(root/'FIFA_Beyond_Potential_Hidden_Gems.ipynb',4)
nbformat.validate(nb)
assert sum(c.cell_type=='markdown' for c in nb.cells)==14
assert sum(c.cell_type=='code' for c in nb.cells)==14
m=pd.read_csv(root/'results/performance.csv')
hits=m.loc[m.target.eq('quality80')&m.k.eq(50)].groupby('method').found.sum()
assert int(hits['Player model'])==22 and int(hits['FIFA potential'])==14
top=pd.read_csv(root/'results/latest_model_shortlist.csv').set_index('player_id')
shap=pd.read_csv(root/'results/latest_shortlist_shap.csv')
assert len(top)==20 and top.index.is_unique and top.overall.lt(70).all()
assert top.snapshot_date.eq('2025-09-21').all()
reconstructed=shap.groupby('player_id').shap.sum()+top.expected_score
np.testing.assert_allclose(reconstructed.sort_index(),top.raw_score.sort_index(),atol=1e-6)
np.testing.assert_allclose(top.expected_score+top.potential_contribution+top.other_contribution,top.raw_score,atol=1e-6)
assert json.loads((root/'results/latest_shortlist_audit.json').read_text())['shap_additivity_checked']
print('PASS: notebook structure, 22 vs 14 hits, eligible top 20 and additive SHAP evidence.')
