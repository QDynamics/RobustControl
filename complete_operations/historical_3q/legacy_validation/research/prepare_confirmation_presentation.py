"""Preserve pilot figures and make explicitly named confirmation entry points."""
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]
dest=ROOT/'figures/v12_pilot_16point'; dest.mkdir(exist_ok=False)
for name in ('interface_selection_v12','interface_cost_readout_v12'):
    for ext in ('.pdf','.png'): shutil.copy2(ROOT/'figures'/(name+ext),dest/(name+ext))
text=(ROOT/'research/present_v12.py').read_text(encoding='utf-8').replace('v12_map_v1','v12_confirmation_v1')
text=text.replace('"""Generate figures/tables directly from the frozen validation records."""','"""Generate final figures from the independently registered 128-point confirmation."""')
(ROOT/'research/present_v12_confirmation.py').write_text(text,encoding='utf-8')
text=(ROOT/'research/replay_v12_controls.py').read_text(encoding='utf-8').replace('v12_map_v1','v12_confirmation_v1')
text=text.replace('"""Portable independent control/channel replay. Requires a NEW output directory."""','"""Portable independent replay of the 128-point confirmation; NEW output required."""')
(ROOT/'research/replay_v12_confirmation.py').write_text(text,encoding='utf-8')
main=ROOT/'cnot_selection_v12.tex'; text=main.read_text(encoding='utf-8')
text=text.replace('a validation ensemble of 16 points comprising eight independent draws and their negatives','an independent confirmation ensemble of 128 points comprising 64 independent draws and their negatives')
text=text.replace('the paired-sample 95\\% interval for the gap to the runner-up includes zero','the paired-sample interval for the gap to the runner-up includes zero')
text=text.replace('Eight antithetic pairs are the independent units; intervals are exploratory and lack simultaneous coverage.','Sixty-four antithetic pairs are the independent units. Student intervals use a Bonferroni correction over all eligible candidate pairs within each cell (approximate 95\\% coverage), without simultaneous coverage across cells.')
text=text.replace('The candidates, library-selection rule, and validation points were fixed before the map was evaluated.','The library and noise grid were fixed before validation. A 16-point pilot was retained; the uniform 128-point confirmation was registered after that pilot, with no change to the controls or validation domain.')
main.write_text(text,encoding='utf-8')
supp=ROOT/'v12_supplement.tex'; text=supp.read_text(encoding='utf-8')
text=text.replace('Validation uses a new seed, 12092721, with eight independent draws and their negatives. Antithetic pairs, rather than individual points, are the eight independent units for uncertainty diagnostics. Every library member is evaluated at every registered point. The best mean among eligible members determines the cell color; a cell is unresolved if the paired 95\\% Student interval for the difference to its runner-up includes zero or the mean gap is at most $4\\times10^{-7}$. The critical value has seven degrees of freedom. These small-sample, post-ranking intervals are descriptive: they do not guarantee simultaneous coverage, account for every selection comparison, or certify a continuous parameter region. A color therefore denotes an empirical library ranking rather than a proven optimal controller.',r'''The 16-point pilot uses seed 12092721, with eight independent draws and their negatives. It left 21 of 42 complete-task comparisons unresolved. A separately registered confirmation protocol freezes the same library, objectives, hardware, and entire parameter grid, then evaluates 128 new points: 64 independent draws with seed 12092722 and their negatives. Every candidate and every cell receives this same confirmation budget; no favorable cell is preferentially sampled and no controls are reselected. The original pilot arrays and figures are retained separately. The final plots use only the new confirmation set.

Antithetic pairs, rather than individual points, are the 64 independent units. The smallest mean determines the cell color. For $k$ eligible candidates, the Student critical value is $t_{63,1-0.05/[2\binom{k}{2}]}$, accounting for all candidate pairs within the cell by a Bonferroni correction. A cell remains unresolved if the interval for its gap to the runner-up includes zero or if that gap is at most $4\times10^{-7}$. This is an approximate within-cell 95\% diagnostic based on the distribution of sample means; it does not establish exact coverage, simultaneous coverage across all 42 cells, or a continuous parameter region. A color denotes an empirical library ranking, not an optimal controller.''')
supp.write_text(text,encoding='utf-8')
print('Pilot preserved; confirmation presentation and replay entry points created')
