"""Structural checks of the local educational Fanuc program; not machine certification."""
from pathlib import Path
import re
import math
import json

directory=Path(__file__).resolve().parent
program=next(directory.glob('program*.nc'))
text=program.read_text()
assert text.isascii()
assert text.startswith('%\n')
assert '#' not in text and 'G28' not in text
position={'X':0.0,'Y':0.0,'Z':0.0}
metric=False
motion=None
feed=None
numbers=[]
errors=[]
drills=[]
feeds=set()
speeds=set()
tool_changes=[]
cut_levels=set()
clearance_raises=0
allowed_g={0,1,2,3,17,21,40,49,54,80,81,90,94,98}
for line_number,line in enumerate(text.splitlines(),1):
    clean=re.sub(r'\([^)]*\)','',line).strip()
    if not clean or clean=='%':continue
    words=re.findall(r'([A-Z])([+-]?(?:\d+(?:\.\d*)?|\.\d+))',clean)
    remainder=re.sub(r'[A-Z][+-]?(?:\d+(?:\.\d*)?|\.\d+)','',clean).strip()
    assert not remainder,(line_number,remainder)
    g=[int(float(v)) for k,v in words if k=='G']
    assert all(n in allowed_g for n in g),(line_number,g)
    params={k:float(v) for k,v in words if k not in ('G','N')}
    ns=[int(v) for k,v in words if k=='N']
    numbers.extend(ns)
    if 21 in g:metric=True
    if 'F' in params:
        feed=params['F']
        assert feed>0
        feeds.add(feed)
    if 'S' in params:speeds.add(params['S'])
    if params.get('M')==6:tool_changes.append(int(params['T']))
    if 81 in g:
        drills.append({'X':params.get('X',position['X']),'Y':params.get('Y',position['Y']),'Z':params['Z'],'R':params['R'],'F':feed})
        assert params['R']>0 and params['Z']<0
        continue
    if 80 in g:motion=None
    explicit=[n for n in g if n in (0,1,2,3)]
    if explicit:motion=explicit[-1]
    if not any(a in params for a in ('X','Y','Z')):continue
    if motion is None:continue
    assert metric,('motion before G21',line_number)
    target={k:params.get(k,v) for k,v in position.items()}
    if motion in (1,2,3):
        assert feed and feed>0
        cut_levels.add(round(target['Z'],3))
    if motion==0 and target['Z']>position['Z'] and abs(target['Z']-40)<1e-6:
        clearance_raises+=1
    if motion in (2,3):
        center=(position['X']+params.get('I',0),position['Y']+params.get('J',0))
        r0=math.hypot(position['X']-center[0],position['Y']-center[1])
        r1=math.hypot(target['X']-center[0],target['Y']-center[1])
        errors.append(abs(r0-r1))
        assert abs(r0-r1)<0.003,(line_number,r0,r1)
    position=target
assert numbers==sorted(set(numbers))
assert re.search(r'\bM30\s*$',text)
pair=program.stem.removeprefix('program')
expected={'11-12':({500,800},{1200,4000},[1,2]),'13-14':({1600},{8000},[5]),'15-16':({250},{2500},[5])}[pair]
assert expected[0].issubset(feeds),(feeds,expected[0])
assert speeds==expected[1]
assert tool_changes==expected[2]
if pair=='11-12':
    assert cut_levels=={20,30},cut_levels
    assert len(errors)==3,('expected three exact corner arcs',len(errors))
if pair=='13-14':
    assert cut_levels=={20},cut_levels
    assert clearance_raises<=3,('unexpected extra retractions',clearance_raises)
if pair=='15-16':
    assert len(drills)==3
    assert [(d['X'],d['Y']) for d in drills]==[(15,-15),(45,-30),(65,-15)]
    assert drills[0]['Z']==-13 and drills[1]['Z']==-18 and drills[2]['Z']<-23
stats={'line_count':len(text.splitlines()),'ascii':True,'cut_levels_z':sorted(cut_levels),'raises_to_z40':clearance_raises,'metric_before_first_move':True,'numbered_blocks':len(numbers),'arc_count':len(errors),'max_arc_radius_mismatch_mm':max(errors,default=0),'feeds_mm_min':sorted(feeds),'spindles_rpm':sorted(speeds),'tool_changes':tool_changes,'drilling_cycles':drills,'end_position':position,'status':'passed; structural checks only, not a target-controller test'}
path=directory/'validation.json'
data=json.loads(path.read_text())
data['gcode_checks']=stats
path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(program.name,json.dumps(stats,ensure_ascii=False))
