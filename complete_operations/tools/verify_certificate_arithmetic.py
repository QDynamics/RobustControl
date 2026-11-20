"""Independent exact-arithmetic checks of replayed certificate records.

No imports from the certificate producers or their interval backend. This checks
serialized enclosures and proof inequalities; the producer replay supplies the
Hamiltonian/jet evaluation. It is not a second implementation of those jets.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from fractions import Fraction as Q
from pathlib import Path


def interval(r, bits):
    assert r['scale_bits'] == bits
    lo, hi = int(r['lo_scaled']), int(r['hi_scaled'])
    assert lo <= hi
    return lo, hi


def product(a, b):
    x = [u*v for u in a for v in b]
    return min(x), max(x)


def matrix_defect(j_records, y_records, bits):
    """Exact interval I-YJ, expressed at scale 2**(2*bits)."""
    n = len(j_records)
    assert len(y_records) == n and all(len(r) == n for r in j_records+y_records)
    j = [[interval(v,bits) for v in r] for r in j_records]
    y = [[interval(v,bits) for v in r] for r in y_records]
    s2 = 1 << (2*bits)
    defect = []
    for row in y:
        rr=[]
        i=len(defect)
        for c in range(n):
            terms=[product(row[k],j[k][c]) for k in range(n)]
            eye=s2 if i == c else 0
            rr.append((eye-sum(v[1] for v in terms),eye-sum(v[0] for v in terms)))
        defect.append(rr)
    bound=Q(max(sum(max(abs(lo),abs(hi)) for lo,hi in r) for r in defect),s2)
    return bound,defect


def check_neumann(j,y,bits,reported):
    q,defect=matrix_defect(j,y,bits)
    assert 0 <= q <= reported < 1, ('Neumann bound',str(q),str(reported))
    return q,defect


def check_inclusion(r,bits):
    boxes=r['coordinate_box']; images=r['krawczyk_image']; margins=r['strict_margins']
    assert len(boxes)==len(images)==len(margins)==37
    smallest=None
    for box,img,record in zip(boxes,images,margins):
        bl,bh=interval(box,bits); kl,kh=interval(img,bits)
        assert bl < kl <= kh < bh, 'Krawczyk image not strictly inside box'
        assert int(record['lower_scaled'])==kl-bl and int(record['upper_scaled'])==bh-kh
        margin=min(kl-bl,bh-kh)
        smallest=margin if smallest is None else min(smallest,margin)
    return Q(smallest,1<<bits)


def enclosing_dyadic(value,bits):
    value=Q(value)*(1<<bits)
    return value.numerator//value.denominator,-((-value.numerator)//value.denominator)


def check_i2_image(r,centers,selected,defect,bits):
    # Matrix defect is at scale S^2; all other stored quantities have scale S.
    s=1<<bits; n=len(selected)
    c=[enclosing_dyadic(centers[k],bits) for k in selected]
    box=[interval(v,bits) for v in r['coordinate_box']]
    displacement=[(b[0]-a[1],b[1]-a[0]) for b,a in zip(box,c)]
    y=[[interval(v,bits) for v in row] for row in r['preconditioner']]
    f=[interval(v,bits) for v in r['f_center']]
    for i in range(n):
        yf=[product(y[i][k],f[k]) for k in range(n)]
        rem=[product(defect[i][k],displacement[k]) for k in range(n)]
        lo=c[i][0]*s*s-s*sum(v[1] for v in yf)+sum(v[0] for v in rem)
        hi=c[i][1]*s*s-s*sum(v[0] for v in yf)+sum(v[1] for v in rem)
        stored=interval(r['krawczyk_image'][i],bits)
        assert stored[0]*s*s <= lo <= hi <= stored[1]*s*s, 'reconstructed image not enclosed'


def inspect(path):
    d=json.loads(path.read_bytes())
    schema=d.get('schema','')
    known=('impact-i2-robust-root-krawczyk-v1','prl-upgrade-b1-augmented-rank-certificate-v1',
           'prl-v18-generic-product-module-strict-root-v1',
           'prl-v20-three-module-open-entangling-family-interval-v1',
           'prl-v21-three-module-quantitative-radius-v1')
    if schema not in known: return None
    assert d['pass'] is True
    expected=('STRICT_INTERVAL_ROOT_PASS','STRICT_AUGMENTED_RANK_PASS',
              'GENERIC_PRODUCT_MODULE_STRICT_ROOT_PASS',
              'THREE_MODULE_MINIMAL_OPEN_ENTANGLING_FAMILY_PASS',
              'THREE_MODULE_QUANTITATIVE_PATH_LIFTING_PASS')
    assert d['classification']==expected[known.index(schema)]
    if 'results' in d: assert sorted(r['bits'] for r in d['results'])==[320,512]
    report={'file':path.as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'schema':schema,'checks':[]}
    for r in d.get('results',[d]):
        bits=r['bits']; assert bits in (320,512)
        out={'bits':bits}
        if schema==known[0]:
            assert r['pass'] is True
            margin=check_inclusion(r,bits)
            upper=Q(int(r['q_infinity']['hi_scaled']),1<<bits)
            q,defect=check_neumann(r['jacobian_box'],r['preconditioner'],bits,upper)
            check_i2_image(r,d['parameter_exact_decimals'],d['selected_variable_indices'],defect,bits)
            assert interval(r['trace_abs_box'],bits)[0]>0
            assert interval(r['entangling_minor_abs_box'],bits)[0]>0
            out.update(exact_neumann_upper=str(q),reported_neumann_upper=str(upper),
                       strict_inclusion_margin=str(margin),krawczyk_reconstruction=True)
        elif schema==known[1]:
            assert r['pass'] is True and r['augmented_dimension']==44
            assert len(r['visible_coordinate_rows'])==7
            assert r['augmented_jacobian_box']==r['constraint_jacobian_box']+r['selected_visible_jacobian_box']
            upper=Q(int(r['q_infinity']['hi_scaled']),1<<bits)
            q,_=check_neumann(r['augmented_jacobian_box'],r['preconditioner'],bits,upper)
            out.update(exact_neumann_upper=str(q),reported_neumann_upper=str(upper))
        elif schema==known[2]:
            assert r['module'] in (0,15,25) and r['free_coordinates_preserved'] is True
            assert len(r['selected_variable_indices'])==37 and len(r['fixed_variable_indices'])==7
            assert set(r['selected_variable_indices']).isdisjoint(r['fixed_variable_indices'])
            assert set(r['selected_variable_indices']+r['fixed_variable_indices'])==set(range(44))
            c=r['certificate']; assert c['pass'] is True and c['bits']==bits
            margin=check_inclusion(c,bits)
            upper=Q(int(c['q_infinity_hi_scaled']),1<<c['q_infinity_scale_bits'])
            assert 0 <= upper < 1 and c['trace_chart'] is True
            assert interval(c['trace_center'],bits)[0]>0
            out.update(module=r['module'],strict_inclusion_margin=str(margin),
                       reported_neumann_upper=str(upper),
                       limitation='Jacobian/preconditioner not serialized by selected-root worker; inclusion checked from its replayed image.')
        else:
            assert r['modules']==[0,15,25] and len(r['pivot_columns'])==15
            assert len(set(r['pivot_columns']))==15
            assert all(0<=v<21 for v in r['pivot_columns'])
            field='minor_neumann_q_upper' if schema==known[3] else 'uniform_neumann_q_upper'
            upper=Q(r[field])
            q,_=check_neumann(r['minor'],r['minor_preconditioner'],bits,upper)
            out.update(exact_neumann_upper=str(q),reported_neumann_upper=str(upper))
            if schema==known[3]:
                assert Q(interval(r['entangling_minor_abs'],bits)[0],1<<bits)>Q(1,2)
                e=r['entangling_minor']; re=interval(e['re'],bits); im=interval(e['im'],bits)
                lower=lambda a: 0 if a[0]<=0<=a[1] else min(abs(a[0]),abs(a[1]))
                assert Q(lower(re)**2+lower(im)**2,1<<(2*bits))>Q(1,4)
                assert r['composite_differential_rank']==15
                for b in r['branch_certificates']:
                    assert 0<=Q(b['dependent_neumann_q_upper'])<1
                out.update(target_rank=15,fixed_target_fibre_dimension=3*7-15)
            else:
                norm=Q(max(sum(max(abs(lo),abs(hi)) for lo,hi in
                                  (interval(x,bits) for x in row)) for row in r['minor_preconditioner']),1<<bits)
                assert norm==Q(r['preconditioner_infinity_norm_upper']) and norm>0
                rho=Q(r['input_control_radius'])
                delta=rho*(1-upper)/norm
                assert delta==Q(r['principal_log_frobenius_radius_lower'])
                assert delta>Q(r['declared_decimal_lower_bound'])==Q('3.4e-17')
                for b in r['module_records']:
                    assert b['pass'] is True and b['bits']==bits
                    assert 0<=Q(b['continuation_q_upper'])<1 and 0<=Q(b['continuation_ratio_max'])<1
                    assert 0<=Q(b['implicit_q_upper'])<1 and interval(b['hidden_chart_abs'],bits)[0]>0
                out.update(radius_lower=str(delta),radius_lower_display=float(delta),
                           preconditioner_norm_display=float(norm))
        report['checks'].append(out)
    report['pass']=True
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise SystemExit('Refusing existing audit output')
    start=time.perf_counter()
    reports=[]
    directory=args.directory.resolve()
    if os.name=='nt' and not str(directory).startswith(chr(92)*2+'?'+chr(92)):
        directory=Path(chr(92)*2+'?'+chr(92)+str(directory))
    for path in sorted(directory.rglob('*.json')):
        item=inspect(path)
        if item:
            item['file']=path.relative_to(directory).as_posix()
            reports.append(item)
    counts={}
    for r in reports: counts[r['schema']]=counts.get(r['schema'],0)+1
    assert sorted(counts.values())==[1,1,2,2,6],counts
    assert sum(len(r['checks']) for r in reports)==14
    module_cases={(v['module'],v['bits']) for r in reports for v in r['checks'] if 'module' in v}
    assert module_cases=={(m,b) for m in (0,15,25) for b in (320,512)}
    for schema in ('prl-v20-three-module-open-entangling-family-interval-v1',
                   'prl-v21-three-module-quantitative-radius-v1'):
        assert sorted(v['bits'] for r in reports if r['schema']==schema for v in r['checks'])==[320,512]
    # Meaningful failure controls: force a singular minor and violate strict inclusion.
    one={'lo_scaled':'1','hi_scaled':'1','scale_bits':0}
    zero={'lo_scaled':'0','hi_scaled':'0','scale_bits':0}
    rejected=False
    try: check_neumann([[zero]],[[one]],0,Q(0))
    except AssertionError: rejected=True
    assert rejected
    bad={'coordinate_box':[one]*37,'krawczyk_image':[one]*37,
         'strict_margins':[{'lower_scaled':'0','upper_scaled':'0'}]*37}
    rejected=False
    try: check_inclusion(bad,0)
    except AssertionError: rejected=True
    assert rejected
    result={'status':'INDEPENDENT_EXACT_ARITHMETIC_AUDIT_PASS','python':sys.version,
            'elapsed_seconds':time.perf_counter()-start,'records':reports,'counts':counts,
            'negative_controls_pass':True,
            'scope':'Exact integer/rational checks of serialized enclosures and acceptance inequalities; no reimplementation of the interval Hamiltonian/jet evaluator.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,indent=2)
    print(result['status'],len(reports),'records, 14 precision cases')


if __name__=='__main__': main()
