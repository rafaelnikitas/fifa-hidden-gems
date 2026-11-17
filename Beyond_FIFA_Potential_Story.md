<!-- story:intro -->
# Beyond FIFA potential: finding hidden gems before the breakthrough

I compared a player model with FIFA potential to find future 80+ players while they were still rated below 70.

<!-- story:question -->
## 1. A shortlist before the breakthrough

FIFA potential is a natural place to start a scouting shortlist. I wanted to test whether the same review budget could find more future quality when I added a player's skills, development and surroundings.

I ranked players with **overall below 70, at any age**, and checked whether they reached **80+ in the next three editions**. Previously observed 85+ players were excluded. I compared 50 selections per edition across 2020–2023.

**The ML model uses FIFA potential and potential headroom as predictors.** I compare this combination with ranking by potential alone. The extra information comes from player skills, development history, club strength and relative standing among peers.

<!-- story:timing -->
## 2. The prediction timeline

I trained each model using only outcomes available at its decision date. For a 2023 shortlist, training profiles end in 2020, with their three-edition outcomes known by 2023. The shortlist is then evaluated using 2024–2026 ratings.

A recorded 80+ rating confirms a positive outcome. A negative requires all three future records. Players with incomplete follow-up stay unresolved: disappearing from the dataset does not count as stagnation.

<!-- story:comparison -->
## 3. More discoveries from the same budget

**The player model finds 22 confirmed future 80+ selections; FIFA potential finds 14.** That is eight additional hits from the same 200 review slots, or 57% more confirmed hits.

The model matches or exceeds potential in each annual top-50 list: 4 versus 1, 5 versus 5, 4 versus 3, and 9 versus 5. The budget curve shows how the comparison changes as the shortlist grows.

Some players recur across editions. Confirmed precision uses all review slots; confirmed recall uses the 101 documented positive player-edition outcomes. Unresolved players remain in both rankings.

<!-- story:uncertainty -->
## 4. Uncertainty around the gain

I resampled whole player IDs across all four editions and rebuilt the top-50 lists in each draw. This preserves the connection between repeated observations of the same player.

The pooled difference is **+8 confirmed hits**, with a conditional 95% interval of **+1 to +15**. Individual seasons have wider intervals. This calculation holds the fitted models and historical cohorts fixed; it does not include model-selection uncertainty or variation across future seasons.

<!-- story:removal -->
## 5. The information behind the improvement

I retrained the same model after removing two groups of information. The target, training dates, weights and CatBoost settings stayed fixed.

Without **team and peer context**, confirmed hits fall from 22 to **11**. Without **player history**, they fall to **17**. A model using only age, current rating, potential, headroom and position finds **7**.

The conditional interval for the context difference is **+3 to +19** hits. The history difference is less precise: **-1 to +11**.

The largest drop comes from removing context: club strength, standing within the squad, and comparisons with players of the same position and age. The removal losses overlap because the remaining features are retrained together.

<!-- story:features -->
## 6. The signals learned by the model

**Potential remains the strongest individual feature.** The added information changes how that potential is interpreted. Among the leading inputs are club top-11 strength, listed squad status, rating rank for the same age and position, and position-specific skill rank.

The chart shows how each feature contributes to the fitted ranking. Points to the right raise a player's score; colour shows whether the feature value is relatively low or high. Importance is measured on a random sample of 1,200 eligible 2023 players, rather than only successful discoveries.

Club strength and peer standing help distinguish players with similar overall ratings. These are predictive associations: a contribution from club strength does not establish that joining a stronger club causes development.

<!-- story:ranks -->
## 7. The players each approach finds

The difference between the lists is visible at player level. **Eleven confirmed breakouts appear only in the ML top 50; three appear only in FIFA's.** Another eleven appear in both. The net gain is eight selections.

The rank plot includes all 101 confirmed player-edition breakouts. Players near the top right were prioritised by ML and ranked much lower by FIFA potential. Players near the bottom left show the reverse.

<!-- story:similarity -->
## 8. Similar profiles, different futures

I also compared players by the shape of their attributes. The coordinates were fitted on 2015 outfield profiles and then held fixed. The two displayed dimensions retain **34.1%** of profile variation.

Similar profiles can still have different outcomes. The map combines similarity with the two shortlists and the later rating reveal. Nearest-player comparisons use all 28 normalised skill dimensions within the same position group; goalkeeper comparisons use five goalkeeping attributes.

<!-- story:explorer -->
## 9. An interactive scouting replay

The scouting replay lets me compare rankings within an edition, position or nationality. Filters rerank the full eligible pool. Clicking a player reveals the recorded career and an available individual explanation.

<!-- story:cases -->
## 10. Discoveries, false alarms and misses

In 2023, **M. Kerkez** and **M. Svilar** both started at 69 and later reached 82. The model ranked them fifth and thirtieth; FIFA potential ranked them fifty-second and 376th.

The same model ranked **Pablo Torre** first, but his observed peak in the window was 74. It also placed **A. Güler** sixty-second and missed him at top 50. These examples include discoveries, a false alarm and a miss.

Individual SHAP values connect these rankings to specific inputs. They sum, with the expected score, to the fitted raw score. The displayed bars are the six largest contributions; the score is a ranking index rather than a calibrated probability.

<!-- story:selection -->
## 11. How I selected the final model

I treated this as a ranking problem: given the same 50-player review budget, which approach finds more confirmed future 80+ players? I compared FIFA potential, compact learned controls, the player model, and extensions with current or forecast role profiles.

The 431-input player model had the strongest top-50 result in the four historical replays. I kept its CatBoost settings fixed for the feature-removal comparisons: 450 iterations, depth 5, learning rate 0.045, L2 regularisation 12, border count 64 and seed 42. Shallower trees and regularisation limit the complexity of each fitted interaction.

**I did not run an automated hyperparameter search or use these evaluation cohorts for early stopping.** This is an exploratory model/feature comparison on an archive studied previously; its replay periods are not an untouched final test set.

**I did not calibrate the output probabilities.** Sample weights emphasise the scouting use case, so I use the raw model output only to order players. A high ranking is not a stated percentage chance of reaching 80+. If I wanted probabilities, I would fit a calibration step on a separate mature validation window and check reliability and a proper probability score.

<!-- story:findings -->
## 12. What I found

**At a fixed review budget, adding player information improves the historical 80+ shortlist.** The gain is 22 versus 14 confirmed selections. Team and peer context produce the largest loss when removed.

The stricter pool of players never previously rated 70+ gives **23 versus 14** confirmed selections. For the rarer 85+ outcome, the model and potential each find **5**; there are only ten documented 85+ player-edition events in these cohorts.

I also tested forecast role-profile features. That extension finds 17 future 80+ selections at top 50, below the player model's 22. The improvement over potential comes from the player model; the role forecast did not improve this shortlist.

The scouting value is a better order for reviewing low-rated players. The next evaluation is a frozen pipeline on later-maturing cohorts, with match performance and playing time alongside game ratings.

<!-- story:watchlist -->
## 13. The latest data-backed shortlist

I apply the selected player model to players still rated below 70, at any age, excluding previously observed 85+ players. The table shows the model's first 20 selections, their FIFA potential rank in the same eligible pool, and the inputs that most strongly support or oppose each model score.

FIFA potential is part of the ML model. Its value is interpreted together with the player's attributes, history, team and peers. The individual explanations separate the contribution of current potential and headroom from the contribution of all other inputs.

These are **unconfirmed prospective selections**. The data date and target editions are shown with the results. The bundled complete snapshot is dated **21 September 2025**; it is not a live November 2026 roster. The newer files checked below lack usable potential, so I do not replace it with invented or silently carried-forward values.

Each explanation uses native SHAP values from the selected model. Positive values raise the ranking score and negative values lower it. They describe the model's reasoning, rather than causes of future development.
