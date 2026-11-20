"""Bind the accepted research outputs, chronological schedule, and delivery ZIP."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from fractions import Fraction
import hashlib
import json
import re
import subprocess
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/cnot_tolerance_20261012_v1'
RESULT=ROOT/'results/cnot_tolerance_20261012_v1'
PDF=ROOT/'output/pdf/cnot_tolerance_20261012_v1'
BUILD=ROOT/'tmp/pdfs/cnot_tolerance_20261012_v1/build_r4'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p,d):
    with p.open('x',encoding='utf-8',newline='\n') as f:json.dump(d,f,ensure_ascii=False,indent=2)


def main():
    for p in [PDF/'CNOT_TOLERANCE_NOTE.pdf',RESULT/'CNOT_TOLERANCE_REPRODUCIBILITY.zip',RESULT/'DELIVERY_MANIFEST_SHA256.json']:
        assert not p.exists(),f'Refuse to overwrite {p}'
    old=ROOT/'manuscript/robust_operations_20261012_v19'
    manifest=json.loads((old/'V19_DELIVERY_MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    for row in manifest['files']:assert digest(old/row['relative_path'])==row['sha256']
    accepted=json.loads((RUN/'primary_audit/ACCEPTANCE.json').read_text())
    assert accepted['primary_registered_criterion_passed']
    log=(BUILD/'CNOT_TOLERANCE_NOTE.log').read_text(encoding='utf-8',errors='replace')
    assert not re.search(r'Overfull|Underfull|Warning:|undefined|^!',log,re.M|re.I)
    assert (BUILD/'CNOT_TOLERANCE_NOTE.tex').read_bytes()==(RESULT/'CNOT_TOLERANCE_NOTE.tex').read_bytes()
    assert (BUILD/'cnot_tolerance_summary.pdf').read_bytes()==(PDF/'cnot_tolerance_summary.pdf').read_bytes()
    reader=PdfReader(str(BUILD/'CNOT_TOLERANCE_NOTE.pdf'))
    assert len(reader.pages)==5
    extracted=subprocess.run(['pdftotext','-layout',str(BUILD/'CNOT_TOLERANCE_NOTE.pdf'),'-'],capture_output=True,check=True).stdout.decode('utf-8')
    text=re.sub(r'\s+','',extracted)
    for token in ['0.018249','0.009937328198746783','17.547134']:
        assert token in text,token
    assert len(PdfReader(str(PDF/'cnot_tolerance_summary.pdf')).pages)==1
    fonts=subprocess.run(['pdffonts',str(PDF/'cnot_tolerance_summary.pdf')],capture_output=True,text=True,check=True).stdout
    font_lines=fonts.splitlines()[2:]
    assert all('TimesNewRoman' in line for line in font_lines if line.strip())
    # The 90 stored durations are in AREA_INDICES order: each module's 15
    # rotations followed by its 15 twists. Provide an explicit physical order.
    schedule=accepted['continuous_family']['common_actuator_schedules']['0.0037']
    chronological=[];clock=Fraction(0)
    for j in (2,1,0):
        for k in range(15):
            for kind,local in [('twist',15+k),('rotation',k)]:
                index=30*j+local
                duration=schedule['durations_t0'][index]
                chronological.append({'slot':len(chronological)+1,'module_index':j,'pair_index':k,
                    'generator':kind,'parameter_index_132':44*j+(29+k if kind=='twist' else k),
                    'area_order_index_90':index,'duration_t0':duration,
                    'start_t0_display':float(clock),'start_t0_exact_rational':str(clock),
                    'peak_command_upper':schedule['peak_command_upper'][index]})
                clock+=Fraction.from_float(duration)
                if len(chronological)<90:clock+=Fraction(1,1000)
    assert len(set(x['area_order_index_90'] for x in chronological))==90
    assert clock<Fraction.from_float(schedule['gate_time_upper_t0'])
    save(RUN/'primary_audit/SCHEDULE_CHRONOLOGICAL.json',{
        'scope':'One common schedule for the entire registered interpolating path and W<=0.0037; shared calibration square r=0.01 and optional gamma*t0=1e-5.',
        'source_acceptance_sha256':digest(RUN/'primary_audit/ACCEPTANCE.json'),
        'stored_duration_order':'AREA_INDICES: modules 0,1,2; each with 15 rotation areas followed by 15 twisting areas.',
        'physical_execution_order':'Modules 2,1,0; within each, pairs 0..14, twisting then rotation.',
        'durations_are_authoritative_binary64_real_numbers':True,
        'switch_duration_exact_t0':'1/1000','switch_count':89,
        'total_exact_rational_t0':str(clock),'total_upper_t0':schedule['gate_time_upper_t0'],
        'slots':chronological})
    state={
        'date':'2026-10-12','status':'RESEARCH_MILESTONE_ACCEPTED',
        'main_result':'Continuous shared-calibration and control-neighborhood bounds for the actual full8 encoded CNOT; a finite-tolerance continuous family and common actuator schedule.',
        'primary_control_radius_r':0.018249,'channel_tolerance':0.01,
        'continuous_family':{'shared_error_radius':0.01,'weighted_control_budget':0.0037,'local_dephasing_gamma_t0':1e-5,'complete_channel_halfdiamond_upper':accepted['continuous_family']['uniform_halfdiamond_upper_with_dephasing'],'common_time_upper_t0':schedule['gate_time_upper_t0']},
        'exact_manifold_root_certificate':False,'device_validation':False,'global_optimality':False,
        'manuscript_v19_unchanged':True,'v19_manifest_entries_verified':len(manifest['files']),
        'next_priority':'Interval root and nonlinear fixed-target fiber continuation, or model extension to module-varying errors with independently registered resource comparisons.',
        'primary_acceptance':str((RUN/'primary_audit/ACCEPTANCE.json').relative_to(ROOT)),
        'PDF_visual_acceptance':'Primary agent inspected all 5 pages at 110 dpi, then re-inspected final changed pages 3 and 5; other pages had identical rendered hashes. The final standalone figure was inspected at full resolution.',
        'figure_fonts':'Embedded Times New Roman regular and italic only.',
        'toolchain_repairs':'Preserved two failed initial LaTeX builds due to package load order. Final build_r4 has no errors, unresolved references or box warnings.'}
    save(RESULT/'RESEARCH_STATE.json',state)
    with (PDF/'CNOT_TOLERANCE_NOTE.pdf').open('xb') as f:f.write((BUILD/'CNOT_TOLERANCE_NOTE.pdf').read_bytes())
    sources=[ROOT/'docs/CNOT_TOLERANCE_PROTOCOL_20261012_v1.json']
    sources+=[ROOT/'src'/name for name in ['cnot_tolerance_v1.py','cnot_interval_v1.py','cnot_interval_ball_v2.py']]
    sources+=[ROOT/'tools'/name for name in ['run_cnot_tolerance_numerics_20261012_v1.py','cnot_interval_preflight_20261012_v1.py','certify_cnot_tolerance_20261012_v1.py','certify_cnot_tolerance_20261012_v2.py','audit_cnot_tolerance_20261012_v1.py','replay_cnot_tolerance_20261012_v1.py','plot_cnot_tolerance_20261012_v1.py','plot_cnot_tolerance_20261012_v2.py','closeout_cnot_tolerance_20261012_v1.py']]
    sources+=sorted(p for p in RUN.rglob('*') if p.is_file())
    sources+=sorted(p for p in RESULT.iterdir() if p.is_file())
    sources+=sorted(p for p in PDF.iterdir() if p.is_file())
    sources+=[BUILD/name for name in ['COMMANDS.json','CNOT_TOLERANCE_NOTE.log']]
    bindings=[{'workspace_relative_path':p.relative_to(ROOT).as_posix(),'size_bytes':p.stat().st_size,'sha256':digest(p)} for p in sources]
    save(RESULT/'CONTENT_MANIFEST_SHA256.json',{'recorded_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'scope':'Byte bindings of the new research package; scientific scope is set by the primary acceptance record.','files':bindings})
    sources.append(RESULT/'CONTENT_MANIFEST_SHA256.json')
    archive=RESULT/'CNOT_TOLERANCE_REPRODUCIBILITY.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sources:z.write(p,p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in sources:assert z.read(p.relative_to(ROOT).as_posix())==p.read_bytes()
    save(RESULT/'DELIVERY_MANIFEST_SHA256.json',{'status':'DELIVERY_VERIFIED','archive':{'name':archive.name,'sha256':digest(archive),'size_bytes':archive.stat().st_size,'entries':len(sources)},'content_manifest_sha256':digest(RESULT/'CONTENT_MANIFEST_SHA256.json'),'PDFs':[{'workspace_relative_path':p.relative_to(ROOT).as_posix(),'sha256':digest(p)} for p in [PDF/'CNOT_TOLERANCE_NOTE.pdf',PDF/'cnot_tolerance_summary.pdf']],'v19_preserved':True})
    print(json.dumps({'status':'DELIVERY_VERIFIED','note_pages':len(reader.pages),'zip_entries':len(sources),'zip_bytes':archive.stat().st_size,'v19_preserved_files':len(manifest['files'])}))


if __name__=='__main__':main()
