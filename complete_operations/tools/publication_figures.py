"""Regenerate quantitative displays using public inputs and exclusive outputs.

Historical source calculations remain unchanged. This wrapper redirects their
export directories and allows a standard Matplotlib font on non-Windows hosts.
It does not optimize controls or perform interval certification.
"""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]


def execute(source, replacements, out, argv=None):
    text=source.read_text(encoding='utf-8-sig')
    for before,after in replacements:
        if before not in text:raise ValueError('Expected producer text absent: '+before)
        text=text.replace(before,after)
    previous=sys.argv[:]
    if argv:sys.argv=argv
    try:exec(compile(text,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
    finally:sys.argv=previous
    return {'source':source.relative_to(ROOT).as_posix(),
            'original_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'executed_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'adaptations':'Output redirection and optional local font registration only'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path)
    out=ap.parse_args().out.resolve()
    if out.exists():raise SystemExit('Refusing existing output')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt,font_manager
    from matplotlib.figure import Figure
    import npj_qi_article_figures_20261118_v1 as article
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    register=out/'register_depth';register.mkdir()
    article.circuit_figure(register)
    legacy=ROOT/'historical_3q/legacy_validation'
    target=out/'legacy_library'
    records.append(execute(legacy/'formatting/build_v17_figures_v2.py',[
        ("OUT=BASE/'figures/v17_results_v2'",'OUT=Path('+repr(str(target))+')'),
        ("font_manager.fontManager.addfont(str(Path(r'C:\\Windows\\Fonts')/name))",
         "font_manager.fontManager.addfont(str(Path(r'C:\\Windows\\Fonts')/name)) if (Path(r'C:\\Windows\\Fonts')/name).is_file() else None")],target))
    cnot=ROOT/'historical_3q/cnot_tolerance'
    target=out/'cnot_tolerance'
    records.append(execute(cnot/'tools/plot_cnot_tolerance_20261012_v2.py',[
        ("OUT=ROOT/'output/pdf/cnot_tolerance_20261012_v1'",'OUT=Path('+repr(str(target))+')'),
        ("font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf')",
         "font_manager.fontManager.addfont('C:/Windows/Fonts/times.ttf') if Path('C:/Windows/Fonts/times.ttf').is_file() else None")],target))
    # The unified panel is analytic; no dataset is hidden in its geometry.
    target=out/'unified'
    generic_font=str(font_manager.findfont('STIXGeneral'))
    records.append(execute(legacy/'figures/build_unified_results.py',[
        ('FONT_PATHS = [Path(r"C:\\Windows\\Fonts") / name for name in ("times.ttf", "timesi.ttf", "timesbd.ttf")]',
         'FONT_PATHS = [Path('+repr(generic_font)+')] * 3')],target,
         ['build_unified_results.py','--output-dir',str(target)]))
    plt.close('all')
    with (out/'REPRODUCTION.json').open('x',encoding='utf-8') as stream:
        json.dump({'source_transformations':records,'scope':'Numerical/analytic figure regeneration; no refitting or interval recomputation.'},stream,indent=2)


if __name__=='__main__':main()
