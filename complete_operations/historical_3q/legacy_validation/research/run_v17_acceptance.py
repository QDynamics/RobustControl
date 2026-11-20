"""Sequential acceptance after library freeze; never repeats existing run outputs."""
from pathlib import Path
import os,sys,time,json,subprocess,traceback,hashlib
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
HERE=Path(__file__).resolve().parent; BASE=HERE.parent
OUT=HERE/'runs/v17_workflow_v1';OUT.mkdir(exist_ok=False)
started=time.perf_counter(); records=[]
try:
    while not (HERE/'runs/v17_revision_v1/result.json').exists():
        if (HERE/'runs/v17_revision_v1/failure.txt').exists():raise RuntimeError('Search failed; no dependent validation')
        if time.perf_counter()-started>1800:raise TimeoutError('No frozen library within 30 minutes')
        time.sleep(3)
    commands=[['audit_v17_training.py'],['check_v17_tradeoff.py'],['run_v17_validation.py'],['replay_v17_controls.py','--out',str(HERE/'runs/v17_replay_v1'),'--full-ode']]
    for index,args in enumerate(commands):
        source=HERE/args[0];cmd=[sys.executable,str(source),*args[1:]];tick=time.perf_counter()
        print('START',args[0],flush=True)
        with (OUT/(str(index)+'_'+source.stem+'.log')).open('xb') as log:
            completed=subprocess.run(cmd,cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
        records.append({'command':cmd,'exit_code':completed.returncode,'elapsed_s':time.perf_counter()-tick,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
        (OUT/'progress.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        print('END',args[0],completed.returncode,flush=True)
        if completed.returncode:raise RuntimeError('Acceptance stage failed: '+args[0])
    with (OUT/'result.json').open('x',encoding='utf-8') as h:json.dump({'status':'ALL_SCIENTIFIC_ACCEPTANCE_STAGES_PASSED','commands':records,'elapsed_s':time.perf_counter()-started},h,indent=2)
except Exception:
    (OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
