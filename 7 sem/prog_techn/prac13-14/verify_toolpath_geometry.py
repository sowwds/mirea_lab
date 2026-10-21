"""Run with FreeCADCmd, in a separate process. Sampled checks, not exact certification.

Checks the saved CAM path without modifying the project. Tests wall clearance
at 0.5 mm centreline intervals and pocket coverage on a 0.5 mm XY grid.
"""
from pathlib import Path
import math,json
import numpy as np
import FreeCAD as App
import Part

folder=Path(__file__).resolve().parent
doc=App.openDocument(str(folder/'model13-14.FCStd'))
part=doc.FinishedPart.Shape
op=doc.Pocket
radius=float(op.ToolController.Tool.Diameter)/2
pos=App.Vector(0,0,40)
segments=[]
samples=[]
cut_levels=set()
for cmd in op.Path.Commands:
 p=cmd.Parameters
 end=App.Vector(p.get('X',pos.x),p.get('Y',pos.y),p.get('Z',pos.z))
 if cmd.Name in ('G1','G01','G2','G02','G3','G03'):
  cut_levels.add(round(end.z,5))
  if cmd.Name in ('G1','G01'):
   pts=[(pos.x,pos.y),(end.x,end.y)]
  else:
   cx,cy=pos.x+p.get('I',0),pos.y+p.get('J',0)
   r=math.hypot(pos.x-cx,pos.y-cy)
   a0=math.atan2(pos.y-cy,pos.x-cx);a1=math.atan2(end.y-cy,end.x-cx)
   da=(a1-a0)%(2*math.pi) if cmd.Name in ('G3','G03') else -((a0-a1)%(2*math.pi))
   n=max(2,math.ceil(abs(da)*math.sqrt(max(r,0.001)/(8*0.002))))
   pts=[(cx+r*math.cos(a0+da*i/n),cy+r*math.sin(a0+da*i/n)) for i in range(n+1)]
   pts[-1]=(end.x,end.y)
  for a,b in zip(pts,pts[1:]):
   segments.append((a,b))
   n=max(1,math.ceil(math.dist(a,b)/0.5))
   samples.extend([(a[0]+(b[0]-a[0])*i/n,a[1]+(b[1]-a[1])*i/n) for i in range(n)])
  samples.append(pts[-1])
 pos=end
assert cut_levels=={20},cut_levels
# Mid-height section ensures the floor does not mask the wall clearance.
minimum=min(part.distToShape(Part.Vertex(App.Vector(x,y,25)))[0] for x,y in samples)
pitch=0.5
xx,yy=np.meshgrid(np.arange(-85, -15+pitch/2,pitch),np.arange(15,85+pitch/2,pitch))
x,y=xx.ravel(),yy.ravel()
inside=np.array([not part.isInside(App.Vector(float(a),float(b),25),1e-7,True) for a,b in zip(x,y)])
x,y=x[inside],y[inside]
dist2=np.full(x.shape,np.inf)
for (ax,ay),(bx,by) in segments:
 dx,dy=bx-ax,by-ay
 length2=dx*dx+dy*dy
 t=np.zeros(x.shape) if length2<1e-15 else np.clip(((x-ax)*dx+(y-ay)*dy)/length2,0,1)
 dist2=np.minimum(dist2,(x-ax-t*dx)**2+(y-ay-t*dy)**2)
missing=int(np.count_nonzero(dist2>(radius+0.02)**2))
penetration=max(0,radius-minimum)
stats={'method':'sampled wall clearance and raster pocket coverage; no 3D boolean loop',
 'wall_sample_count':len(samples),'wall_sampling_max_step_mm':0.5,'minimum_center_to_part_mm':minimum,
 'maximum_sampled_wall_intrusion_mm':penetration,'wall_tolerance_mm':0.05,
 'coverage_grid_mm':pitch,'pocket_grid_points':len(x),'uncovered_grid_points':missing,
 'arc_chord_sagitta_mm':0.002,'cut_levels_z':sorted(cut_levels)}
print(json.dumps(stats,indent=2),flush=True)
assert penetration<=0.05,stats
assert missing==0,stats
path=folder/'validation.json';data=json.loads(path.read_text());data['sampled_geometry_checks']=stats
path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print('SAMPLED_GEOMETRY_OK',flush=True)
