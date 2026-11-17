# Editing the story

The notebook's commentary is stored in ordinary Markdown cells above its charts and tables. The model does not regenerate or replace those cells when it runs.

## In Kaggle

1. Open the notebook and choose **Edit**.
2. Double-click the paragraph or heading above a chart.
3. Write your own explanation, goals or interpretation. Press **Shift + Enter** to render it.
4. Save a new version. For text-only edits, use **Quick Save** with **Save output for this version** to retain the existing figures without refitting the models.

Code cells remain collapsed. Calculated tables and figures stay separate from your narrative. Editing the words does not change the fitted model or its predictions.

## In the HTML article

Open `Beyond_FIFA_Potential_Article.html` in a browser. Click **Edit text**, then click a heading or paragraph and write directly. Click **Finish editing** to review it and **Save HTML** to download your edited copy.

Only story text is editable. Plots, calculated tables, dates reported with the shortlist, and player-level explanations remain fixed. Changes in this HTML copy do not modify the notebook.

## In the Markdown draft

`Beyond_FIFA_Potential_Story.md` contains the same story without embedded figures or model code. It can be edited in any text editor or used as a draft for Medium. Keep the hidden `<!-- story:... -->` section markers if you want to rebuild the notebook from it.

From the project folder:

```bash
python build_notebook.py --story Beyond_FIFA_Potential_Story.md
```

This updates the notebook's Markdown cells and preserves outputs for unchanged code cells. It does not train models. The `--clear-outputs` option explicitly clears them.

## Export after editing a saved notebook

After editing in Kaggle, download the `.ipynb`. Export that saved file so the article uses your current Markdown and the recorded figures:

```bash
python export_article.py FIFA_Beyond_Potential_Hidden_Gems.ipynb \
  --output Beyond_FIFA_Potential_Article.html \
  --story Beyond_FIFA_Potential_Story.md
```

This export does not execute or fit models. If you edit the story in more than one format, choose one as the source and export from it to keep the copies consistent.

## Refreshing the final shortlist

The supplied complete snapshot is dated **21 September 2025**. The last section reports this date explicitly and does not describe the list as a live November 2026 roster. Newer sources checked for this update lack usable current potential or other required fields; the checks are recorded in `results/newer_sources_checked.json`.

To score a newer complete snapshot, attach one file named `current_scouting_snapshot.csv` to Kaggle, in the SoFIFA-style schema used by the original source. Include the game `edition` and an ISO `snapshot_date`. The loader checks exact player IDs, current potential, age, physical data, skills and club context before scoring. It never silently substitutes old potential for missing current potential.

For a local file with another name, set `FIFA_CURRENT_SNAPSHOT_CSV` and `FIFA_CURRENT_SNAPSHOT_DATE` before running. The custom snapshot changes the final prospective shortlist; the earlier historical comparisons stay fixed.
