"""Verify public payload identity and Python syntax; no scientific execution."""
from pathlib import Path
import ast
import hashlib
import json
import os
import tokenize

ROOT=Path(__file__).resolve().parent
if os.name=='nt' and not str(ROOT).startswith('\\\\?\\'):
    ROOT=Path('\\\\?\\'+str(ROOT))
manifest=json.loads((ROOT/'PUBLICATION_SHA256.json').read_text(encoding='utf-8'))
failures=[];parsed=0
for name,record in manifest['files'].items():
    path=ROOT/name
    if not path.is_file():failures.append(name+': missing');continue
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=record['sha256']:failures.append(name+': SHA mismatch')
    if len(data)!=record['bytes']:failures.append(name+': size mismatch')
    if path.suffix=='.py':
        try:
            with tokenize.open(path) as stream:ast.parse(stream.read(),filename=name)
            parsed+=1
        except Exception as error:failures.append(name+': '+str(error))
report={'pass':not failures,'files':len(manifest['files']),'python_sources_parsed':parsed,
        'failures':failures,'scope':'File identity and Python syntax only; not scientific replay.'}
print(json.dumps(report,indent=2))
raise SystemExit(bool(failures))
