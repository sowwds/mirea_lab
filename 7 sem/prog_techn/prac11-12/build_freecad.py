CONFIG = {'practice': 11, 'directory': '/home/aleksey/Documents/mirea_lab/7 sem/prog_techn/prac11-12', 'name': 'Practical11_12', 'label': 'Face and contour milling', 'figure': '44', 'change_point': [50, 50, 150], 'program': 'program11-12.nc', 'project': 'model11-12.FCStd'}

"""Run inside FreeCAD's Python console or through its MCP execute_code tool.
All outputs are written beside this script. Geometry follows the local PDF.
"""
import math
import json
import importlib
from pathlib import Path as FilePath
import FreeCAD as App
import FreeCADGui as Gui
import Part
import Sketcher
import Path
import Path.Main.Job as JobAPI
import Path.Main.Gui.Job as JobGUI
import Path.Op.Gui.Base as OpGUI
from Path.Tool.toolbit import ToolBit

V = App.Vector

def sketch(doc, name, geometries, z=0):
    obj = doc.addObject('Sketcher::SketchObject', name)
    obj.Placement.Base = V(0,0,z)
    for geometry in geometries:
        idx = obj.addGeometry(geometry, False)
        obj.addConstraint(Sketcher.Constraint('Block',idx))
    doc.recompute()
    return obj

def polygon(doc, name, points, z=0):
    return sketch(doc,name,[Part.LineSegment(V(*a,0),V(*b,0))
        for a,b in zip(points,points[1:]+points[:1])],z)

def extrusion(doc, name, base, height):
    obj=doc.addObject('Part::Extrusion',name)
    obj.Base=base
    obj.Dir=V(0,0,height)
    obj.Solid=True
    doc.recompute()
    base.Visibility=False
    return obj

def add_box(doc,name,size,origin):
    obj=doc.addObject('Part::Box',name)
    obj.Length,obj.Width,obj.Height=size
    obj.Placement.Base=V(*origin)
    doc.recompute()
    return obj

def final_cut(doc,base,tool):
    obj=doc.addObject('Part::Cut','FinishedPart')
    obj.Base=base
    obj.Tool=tool
    doc.recompute()
    base.Visibility=False
    tool.Visibility=False
    return obj

def arc(center,r,start,end):
    def pt(angle):
        return V(center[0]+r*math.cos(math.radians(angle)),center[1]+r*math.sin(math.radians(angle)),0)
    return Part.Arc(pt(start),pt((start+end)/2),pt(end))

def geometry_11(doc):
    outer=[(-150,0),(0,0),(0,100),(-100,100),(-150,50)]
    inner=[(-140,10),(0,10),(0,90),(-95.86,90),(-140,45.86)]
    lower=extrusion(doc,'LowerPlate',polygon(doc,'BaseOutline',outer),20)
    upper=extrusion(doc,'UpperStep',polygon(doc,'MachinedOutline',inner,20),10)
    part=doc.addObject('Part::Fuse','FinishedPart')
    part.Base=lower
    part.Tool=upper
    part.Refine=True
    doc.recompute()
    lower.Visibility=False
    upper.Visibility=False
    blank=extrusion(doc,'Blank',polygon(doc,'BlankOutline',outer),30)
    return part,blank,{'outer_xy':outer,'upper_xy':inner,'lower_height':20,'total_height':30,'blank_height':30}

def geometry_13(doc):
    base=add_box(doc,'BaseBlock',(100,100,30),(-100,0,0))
    a=math.sqrt(38**2-8**2)
    cx=-15-a
    cy=15+a
    angle=math.degrees(math.atan2(8,a))
    # Exact R8/R30 tangencies reproduce the drawing's rounded 7.82/23.01/21.32 chain.
    curves=[
        Part.LineSegment(V(-65,85,0),V(-35,85,0)),
        arc((-35,65),20,90,0),
        Part.LineSegment(V(-15,65,0),V(-15,cy,0)),
        arc((-23,cy),8,0,-90+angle),
        arc((-15,15),30,90+angle,180-angle),
        arc((cx,23),8,-angle,-90),
        Part.LineSegment(V(cx,15,0),V(-70,15,0)),
        arc((-70,30),15,-90,-180),
        Part.LineSegment(V(-85,30,0),V(-85,65,0)),
        arc((-65,65),20,180,90),
    ]
    profile=sketch(doc,'PocketOutline',curves,30)
    tool=extrusion(doc,'PocketVolume',profile,-10)
    part=final_cut(doc,base,tool)
    blank=add_box(doc,'Blank',(100,100,30),(-100,0,0))
    return part,blank,{'block':[100,100,30],'pocket_floor_z':20,'pocket_depth':10,'pocket_bounds':[-85,-15,15,85],'radii':[20,20,15,8,30,8],'small_fillet_centers':[[cx,23],[-23,cy]]}

def geometry_15(doc):
    base=add_box(doc,'BaseBlock',(80,40,20),(0,-40,-20))
    holes=[(15,-15,13),(45,-30,18),(65,-15,None)]
    cone_h=5/math.tan(math.radians(59))
    cutters=[]
    for i,(x,y,depth) in enumerate(holes,1):
        if depth is None:
            cut=doc.addObject('Part::Cylinder','Hole3Through')
            cut.Radius=5
            cut.Height=21
            cut.Placement.Base=V(x,y,-21)
            cutters.append(cut)
        else:
            cut=doc.addObject('Part::Cylinder','Hole%dCylinder'%i)
            cut.Radius=5
            cut.Height=depth-cone_h
            cut.Placement.Base=V(x,y,-depth+cone_h)
            tip=doc.addObject('Part::Cone','Hole%dTip'%i)
            tip.Radius1=0
            tip.Radius2=5
            tip.Height=cone_h
            tip.Placement.Base=V(x,y,-depth)
            cutters.extend([cut,tip])
    cutter=doc.addObject('Part::MultiFuse','DrillingVolumes')
    cutter.Shapes=cutters
    cutter.Refine=True
    doc.recompute()
    part=final_cut(doc,base,cutter)
    for obj in cutters: obj.Visibility=False
    blank=add_box(doc,'Blank',(80,40,20),(0,-40,-20))
    return part,blank,{'block':[80,40,20],'holes':holes,'diameter':10,'tip_angle':118,'tip_height':cone_h,'depths_measured_to_tip':True,'through_tip_z':-20-cone_h-0.5}

def operation(doc,job,kind,name,tc,base=None):
    api=importlib.import_module('Path.Op.'+kind)
    gui=importlib.import_module('Path.Op.Gui.'+kind)
    obj=api.Create(name,parentJob=job)
    obj.ViewObject.Proxy=OpGUI.ViewProvider(obj.ViewObject,gui.Command.res)
    obj.ToolController=tc
    if base is not None: obj.Base=base
    obj.CoolantMode='Flood'
    doc.recompute()
    return obj

def depths(obj,start,end,step=1):
    for name,value in [('StartDepth',start),('FinalDepth',end),('ClearanceHeight',max(start,0)+10),('SafeHeight',max(start,0)+5),('StepDown',step)]:
        if hasattr(obj,name):
            obj.setExpression(name,None)
            setattr(obj,name,value)

def tc_settings(tc,name,diameter,feed,rpm,number):
    tc.Label=name
    tc.Tool.Diameter=diameter
    tc.Tool.Label=name
    tc.ToolNumber=number
    tc.HorizFeed='%s mm/min'%feed
    tc.VertFeed='%s mm/min'%min(feed,250)
    tc.SpindleSpeed=rpm
    tc.SpindleDir='Forward'

def frame(view, width=1400, height=1000):
    bounds=App.BoundBox()
    for obj in App.ActiveDocument.Objects:
        if not obj.Visibility:
            continue
        if hasattr(obj,'Shape') and not obj.Shape.isNull():
            bounds.add(obj.Shape.BoundBox)
        elif hasattr(obj,'Mesh'):
            bounds.add(obj.Mesh.BoundBox)
        elif hasattr(obj,'Path'):
            bounds.add(obj.Path.BoundBox)
    view.fitAll()
    if bounds.isValid():
        rotation=view.getCameraOrientation()
        right=rotation.multVec(V(1,0,0))
        up=rotation.multVec(V(0,1,0))
        pts=[V(x,y,z) for x in (bounds.XMin,bounds.XMax) for y in (bounds.YMin,bounds.YMax) for z in (bounds.ZMin,bounds.ZMax)]
        xs=[p.dot(right) for p in pts]
        ys=[p.dot(up) for p in pts]
        view.getCameraNode().height.setValue(max(max(ys)-min(ys),(max(xs)-min(xs))*height/width)*1.15)

def screenshots(ctx):
    doc,job,part,blank,out=ctx['doc'],ctx['job'],ctx['part'],ctx['blank'],ctx['out']
    App.setActiveDocument(doc.Name)
    Gui.Selection.clearSelection()
    for obj in doc.Objects:
        if hasattr(obj,'ViewObject'): obj.Visibility=False
    part.Visibility=True
    part.ViewObject.ShapeColor=(0.76,0.79,0.83)
    part.ViewObject.LineColor=(0.12,0.14,0.18)
    part.ViewObject.DisplayMode='Flat Lines'
    part.ViewObject.Transparency=0
    view=Gui.activeDocument().activeView()
    view.setAnimationEnabled(False)
    view.stopAnimating()
    view.setCameraType('Orthographic')
    view.viewAxonometric()
    frame(view)
    Gui.updateGui()
    view.redraw()
    view.saveImage(str(out/'model.png'),1400,1000,'White')
    view.viewTop()
    frame(view)
    Gui.updateGui()
    view.redraw()
    view.saveImage(str(out/'model_top.png'),1400,1000,'White')
    for op in ctx['ops']: op.Visibility=True
    part.ViewObject.Transparency=35
    view.viewAxonometric()
    frame(view)
    Gui.updateGui()
    view.redraw()
    view.saveImage(str(out/'toolpaths.png'),1400,1000,'White')
    part.ViewObject.Transparency=0
    for op in ctx['ops']: op.Visibility=False
    part.Visibility=True

def export_nc(ctx):
    post=importlib.reload(importlib.import_module('Path.Post.scripts.fanuc_post'))
    # A working-coordinate change point is explicitly supplied; no machine-home G28
    # or proprietary tool-table macro is assumed for this educational setup.
    post.TOOL_CHANGE=''
    ni=CONFIG['change_point']
    pre='G21\nG17 G54 G40 G49 G80 G90 G94\nG0 Z%.3f\nG0 X%.3f Y%.3f'%(ni[2],ni[0],ni[1])
    end='G0 Z%.3f\nG0 X%.3f Y%.3f\nM9\nM5\nM30'%(ni[2],ni[0],ni[1])
    import shlex
    args='--no-show-editor --line-numbers --no-tlo --no-header --preamble '+shlex.quote(pre)+' --postamble '+shlex.quote(end)
    ctx['job'].PostProcessor='fanuc'
    ctx['job'].PostProcessorArgs=args
    objects=[]
    previous=None
    for op in ctx['ops']:
        if op.ToolController is not previous:
            if previous is not None:
                retreat=ctx['doc'].getObject('ToolChangeRetract') or ctx['doc'].addObject('Path::Feature','ToolChangeRetract')
                retreat.Path=Path.Path([Path.Command('G0',{'Z':ni[2]}),Path.Command('G0',{'X':ni[0],'Y':ni[1]})])
                retreat.Visibility=False
                objects.append(retreat)
            objects.append(op.ToolController)
            previous=op.ToolController
        objects.append(op)
    path=ctx['out']/CONFIG['program']
    nc=post.export(objects,str(path),args)
    assert nc and nc.startswith('%\n') and 'M30' in nc and '#' not in nc
    ctx['nc']=nc
    ctx['program_file']=str(path)

def open_contour(doc, job, tc):
    """Four drawing edges offset by the cutter radius, joined by exact arcs.

    Custom CAM stores the resulting centreline; rerun this build after changing
    the tool diameter or drawing, rather than applying G41/G42 a second time.
    """
    old=doc.getObject('OpenContour') or doc.getObject('Contour')
    if old is not None:
        doc.removeObject(old.Name)
        doc.recompute()
    op=operation(doc,job,'Custom','OpenContour',tc)
    op.Label='Open contour - one pass Z20'
    r=float(tc.Tool.Diameter)/2
    outline=[V(0,90,20),V(-95.86,90,20),V(-140,45.86,20),V(-140,10,20),V(0,10,20)]
    normals=[]
    for a,b in zip(outline,outline[1:]):
        delta=b-a
        normals.append(V(delta.y,-delta.x,0)/delta.Length)
    start=V(25,90+r,20)
    commands=[Path.Command('G0',{'Z':40}),Path.Command('G0',{'X':start.x,'Y':start.y}),
              Path.Command('G0',{'Z':35}),Path.Command('G1',{'Z':20,'F':float(tc.VertFeed)})]
    for i in range(1,len(outline)-1):
        a=outline[i]+normals[i-1]*r
        commands.append(Path.Command('G1',{'X':a.x,'Y':a.y,'F':float(tc.HorizFeed)}))
        b=outline[i]+normals[i]*r
        commands.append(Path.Command('G3',{'X':b.x,'Y':b.y,'Z':20,'I':outline[i].x-a.x,'J':outline[i].y-a.y,'K':0,'F':float(tc.HorizFeed)}))
    commands.append(Path.Command('G1',{'X':25,'Y':10-r,'F':float(tc.HorizFeed)}))
    commands.append(Path.Command('G0',{'Z':40}))
    op.Source='Text'
    op.Gcode=[c.toGCode() for c in commands]
    op.addProperty('App::PropertyString','RouteDescription','Source')
    op.RouteDescription='Drawing 44 / practical 4: top, bevel, left, bottom; one pass Z20; centreline radius compensated'
    op.addProperty('App::PropertyLength','ProgrammedDiameter','Source')
    op.ProgrammedDiameter=2*r
    doc.recompute()
    return op

def build():
    out=FilePath(CONFIG['directory'])
    out.mkdir(parents=True,exist_ok=True)
    doc=App.newDocument(CONFIG['name'])
    App.setActiveDocument(doc.Name)
    Gui.activateWorkbench('CAMWorkbench')
    part,blank,params=globals()['geometry_'+str(CONFIG['practice'])](doc)
    part.Label=CONFIG['label']
    assert part.Shape.isValid() and len(part.Shape.Solids)==1
    part.addProperty('App::PropertyString','SourceFigure','Source')
    part.SourceFigure=CONFIG['figure']
    part.addProperty('App::PropertyString','SourceParameters','Source')
    part.SourceParameters=json.dumps(params,ensure_ascii=False)
    job=JobAPI.Create('Job',[part],None)
    job.ViewObject.Proxy=JobGUI.ViewProvider(job.ViewObject)
    oldstock=job.Stock
    job.Stock=blank
    doc.removeObject(oldstock.Name)
    clone=job.Model.Group[0]
    tc=job.Tools.Group[0]
    ops=[]
    if CONFIG['practice']==11:
        tc_settings(tc,'T2 Endmill D30',30,800,4000,2)
        op=open_contour(doc,job,tc)
        ops.append(op)
    elif CONFIG['practice']==13:
        tc_settings(tc,'T5 Endmill D10',10,1600,8000,5)
        floor=['Face'+str(i+1) for i,f in enumerate(clone.Shape.Faces) if abs(f.BoundBox.ZMin-20)<1e-6 and abs(f.BoundBox.ZMax-20)<1e-6]
        op=operation(doc,job,'PocketShape','Pocket',tc,[(clone,floor)])
        depths(op,30,20,10)
        op.StepOver=40
        op.KeepToolDown=True
        op.MinTravel=True
        ops.append(op)
    else:
        drill=ToolBit.from_shape_id('drill.fcstd').attach_to_doc(doc=doc)
        tc.Tool=drill
        tc_settings(tc,'T5 Drill D10',10,250,2500,5)
        for i,(x,y,depth) in enumerate(params['holes'],1):
            op=operation(doc,job,'Drilling','DrillHole%d'%i,tc)
            op.Base=[]
            op.Locations=[V(x,y,0)]
            op.AddTipLength=False
            op.PeckEnabled=False
            op.DwellEnabled=False
            op.setExpression('RetractHeight',None)
            op.RetractHeight=3
            depths(op,0,-depth if depth else params['through_tip_z'])
            ops.append(op)
    doc.recompute()
    for op in ops:
        op.touch()
    doc.recompute()
    counts={op.Name:len(op.Path.Commands) for op in ops}
    assert all(n>3 for n in counts.values()),counts
    ctx={'doc':doc,'part':part,'blank':blank,'job':job,'ops':ops,'out':out,'params':params}
    export_nc(ctx)
    screenshots(ctx)
    doc.recompute()
    doc.saveAs(str(out/CONFIG['project']))
    ctx['validation']={'solid_valid':part.Shape.isValid(),'solids':len(part.Shape.Solids),'part_volume_mm3':part.Shape.Volume,'blank_volume_mm3':blank.Shape.Volume,'commands':counts,'parameters':params,'postprocessor':'fanuc','change_point':CONFIG['change_point'],'tool_length_compensation':'pre-set tool tip, G49; no G43/H table assumed'}
    (out/'validation.json').write_text(json.dumps(ctx['validation'],ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(ctx['validation'],ensure_ascii=False,indent=2))
    return ctx

context=build()


def setup_fixture(ctx):
    doc,stock=ctx['doc'],ctx['blank']
    bb=stock.Shape.BoundBox
    group=doc.addObject('App::DocumentObjectGroup','Fixture')
    group.Label='Vise (schematic, assumed dimensions)'
    h=15 if CONFIG['practice'] in (11,13) else 12
    pieces=[
        ('ViseBase',(bb.XLength+20,bb.YLength+40,8),(bb.XMin-10,bb.YMin-20,bb.ZMin-15)),
        ('ViseJawFixed',(bb.XLength+10,10,h),(bb.XMin-5,bb.YMin-10,bb.ZMin)),
        ('ViseJawMoving',(bb.XLength+10,10,h),(bb.XMin-5,bb.YMax,bb.ZMin)),
    ]
    for name,size,origin in pieces:
        obj=add_box(doc,name,size,origin)
        obj.ViewObject.ShapeColor=(0.30,0.36,0.44)
        group.addObject(obj)
    ctx['fixture']=group
    for obj in doc.Objects: obj.Visibility=False
    for obj in group.Group: obj.Visibility=True
    ctx['part'].Visibility=True
    stock.Visibility=True
    stock.ViewObject.Transparency=75
    stock.ViewObject.ShapeColor=(0.95,0.72,0.28)
    view=Gui.activeDocument().activeView()
    view.setAnimationEnabled(False)
    view.stopAnimating()
    view.viewAxonometric()
    frame(view)
    Gui.updateGui()
    view.redraw()
    view.saveImage(str(ctx['out']/'setup.png'),1400,1000,'White')
    for obj in group.Group: obj.Visibility=False
    stock.Visibility=False

def simulate(ctx,resolution=None):
    return simulate_exact(ctx)


def simulate_exact(ctx):
    macro=ctx['out']/'animate11-12.FCMacro'
    namespace={'__name__':'practical11_simulation', '__file__':str(macro), '_ACTUAL_STOCK_DEFINE_ONLY':True}
    exec(compile(macro.read_text(),str(macro),'exec'),namespace)
    shape,stats=namespace['compute_actual_stock_result'](ctx['doc'],ctx['job'])
    doc=ctx['doc']
    old=doc.getObject('SimulationResult')
    if old is not None and not hasattr(old,'Shape'):
        doc.removeObject(old.Name)
        old=None
    result=old or doc.addObject('Part::Feature','SimulationResult')
    result.Label='Actual stock after boolean CAM simulation'
    result.Shape=shape
    result.ViewObject.ShapeColor=(0.76,0.79,0.83)
    result.ViewObject.LineColor=(0.12,0.14,0.18)
    result.ViewObject.DisplayMode='Flat Lines'
    for obj in doc.Objects:obj.Visibility=False
    result.Visibility=True
    view=Gui.activeDocument().activeView()
    view.setAnimationEnabled(False)
    view.viewAxonometric()
    frame(view)
    Gui.updateGui()
    view.redraw()
    view.saveImage(str(ctx['out']/'simulation.png'),1400,1000,'White')
    import MeshPart
    mesh=MeshPart.meshFromShape(Shape=shape,LinearDeflection=0.05,AngularDeflection=0.1,Relative=False)
    mesh.write(str(ctx['out']/'simulation.stl'))
    ctx['simulation']=result
    ctx['validation']['simulation']=stats
    (ctx['out']/'validation.json').write_text(json.dumps(ctx['validation'],ensure_ascii=False,indent=2)+'\n')
    result.Visibility=False
    ctx['part'].Visibility=True
    for op in ctx['ops']:op.Visibility=True
    doc.recompute()
    doc.save()
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    return stats


setup_fixture(context)
simulate(context)
