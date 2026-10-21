/* Local WebGL viewer: no external libraries, server or network required. */
'use strict';
(() => {
const cfg=window.PRACTICE_MODEL;
document.head.insertAdjacentHTML('beforeend',`<style>
*{box-sizing:border-box}body{margin:0;background:#101827;color:#e2e8f0;font:15px/1.5 system-ui,sans-serif}main{display:grid;grid-template-columns:1fr 340px;height:100vh}#view{position:relative;min-height:0;overflow:hidden}canvas{position:absolute;width:100%;height:100%;touch-action:none;cursor:grab}canvas:active{cursor:grabbing}aside{background:#192335;overflow-y:auto;padding:28px;border-left:1px solid #304158}h1{font-size:24px;line-height:1.25;margin:6px 0 18px}h2{font-size:16px;margin:22px 0 8px}p{color:#b5c4d8}a{color:#7dd3fc}.eyebrow{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:#93aecb}button{font:inherit;color:#eef4ff;background:#26364d;border:1px solid #435775;border-radius:8px;padding:8px 10px;cursor:pointer}button:hover{background:#344b69}button[aria-pressed=true]{border-color:#5ac8fa;background:#254a60}.buttons{display:flex;gap:6px;flex-wrap:wrap;margin:14px 0}label{display:block;margin:10px 0}input{accent-color:#5ac8fa}.metric{margin-top:14px;padding:12px;background:#111c2c;border:1px solid #33435a;border-radius:10px}.metric strong{color:#6ee7dc}.note{font-size:13px;color:#9cadc5}.tip{position:absolute;left:22px;bottom:18px;right:20px;color:#a9bbcf;font-size:13px;pointer-events:none}.badge{position:absolute;color:#101827;background:#ffc768;border:2px solid #fff9;font-size:12px;font-weight:800;min-width:25px;text-align:center;border-radius:20px;padding:2px 5px;transform:translate(-50%,-50%);pointer-events:none;box-shadow:0 2px 7px #0008}.caption{position:absolute;top:20px;left:22px;color:#c3d5e9;font-size:14px;pointer-events:none}.legend{color:#ffc768}.links{margin-top:24px;display:flex;gap:14px;flex-wrap:wrap;font-size:13px}.error{padding:30px;color:#ffb8b8}@media(max-width:800px){main{display:flex;flex-direction:column;height:auto}#view{height:65vh;min-height:400px;flex:none}aside{overflow:visible;border-left:0;border-top:1px solid #304158;padding:22px}}
</style>`);
document.body.innerHTML=`<main><section id="view"><canvas></canvas><div id="labels"></div><div class="caption" id="caption"></div><div class="tip">Вращение: перетаскивание мышью или пальцем. Масштаб: колесо или жест двумя пальцами.</div></section><aside>
<div class="eyebrow">Практическая ${cfg.number} · Вариант 9</div><h1>${cfg.title}</h1><p>${cfg.description}</p>
<div class="buttons"><button id="finished" aria-pressed="true">Готовая деталь</button><button id="stock" aria-pressed="false">Заготовка</button></div>
<div class="buttons"><button data-view="iso">Общий вид</button><button data-view="top">Сверху</button><button data-view="front">Со стороны Y+</button><button data-view="bottom">Снизу</button></div>
<label><input id="xray" type="checkbox"> Прозрачная заготовка</label>
<label><input id="path" type="checkbox"> Траектория и номера точек</label>
<div class="metric">Заготовка: <b>${cfg.bounds[1]-cfg.bounds[0]} × ${cfg.bounds[3]-cfg.bounds[2]} × ${cfg.height} мм</b><br>Глубина выемки: <strong>${fmt(cfg.height-cfg.floor)} мм</strong><br>Материал под дном: <strong>${fmt(cfg.floor)} мм</strong></div>
<h2>Как читать модель</h2><p>Светлая поверхность — верх заготовки. Бирюзовые поверхности — стенки и дно выемки. Жёлтая линия при включении траектории показывает программируемый контур на уровне дна.</p>
<p class="note">${cfg.note}</p><p class="note">Показана форма выемки, ограниченной контуром, а не симуляция снятия материала отдельными проходами. Коррекция G41/G42 учитывает радиус фрезы в УП; здесь изображён номинальный контур детали. Скругления отрисованы по дугам с небольшой дискретизацией.</p>
<div class="links"><a href="../prac8/model8.html">Модель 8</a><a href="../prac9/model9.html">Модель 9</a><a href="../prac10/model10.html">Модель 10</a><a href="otchetpr${cfg.number}.md">Отчёт</a></div></aside></main>`;
function fmt(n){return String(Math.round(n*1000)/1000).replace('.',',');}
const canvas=document.querySelector('canvas'),gl=canvas.getContext('webgl',{antialias:true,alpha:false});
if(!gl){document.querySelector('#view').innerHTML='<div class="error">Для просмотра модели нужен браузер с поддержкой WebGL.</div>';return;}
const TAU=Math.PI*2,mod=a=>(a%TAU+TAU)%TAU;
function sample(start,segments){
 const points=[[start[0],start[1]]],labels=start[2]==null?[]:[[...start]];let p=points[0];
 for(const[type,x,y,r,id]of segments){const end=[x,y];
  if(type==='line')points.push(end);
  else{
   const dx=x-p[0],dy=y-p[1],d=Math.hypot(dx,dy);if(d>2*r+1e-8)throw Error('Невозможный радиус дуги');
   const h=Math.sqrt(Math.max(0,r*r-d*d/4)),mx=(x+p[0])/2,my=(y+p[1])/2;let chosen;
   for(const sign of[1,-1]){const c=[mx-sign*dy*h/d,my+sign*dx*h/d],a=Math.atan2(p[1]-c[1],p[0]-c[0]),b=Math.atan2(y-c[1],x-c[0]),sweep=type==='ccw'?mod(b-a):-mod(a-b);if(Math.abs(sweep)<=Math.PI+1e-8){chosen={c,a,sweep};break;}}
   if(!chosen)throw Error('Не удалось построить дугу');const{c,a,sweep}=chosen,n=Math.max(8,Math.ceil(Math.abs(sweep)*r/.5));
   for(let j=1;j<n;j++){const t=a+sweep*j/n;points.push([c[0]+r*Math.cos(t),c[1]+r*Math.sin(t)]);}points.push(end);
  }
  if(id!=null)labels.push([x,y,id]);p=end;
 }
 return{points,labels};
}
const boundary=sample(cfg.start,[...cfg.segments,...cfg.closing]).points;
if(Math.hypot(boundary[0][0]-boundary.at(-1)[0],boundary[0][1]-boundary.at(-1)[1])<1e-8)boundary.pop();
const tool=sample(cfg.leadStart,[...cfg.lead,...cfg.segments,...cfg.exit]);
const uniqueLabels=new Map(tool.labels.map(p=>[p[2],p]));
document.querySelector('#labels').innerHTML=[...uniqueLabels.values()].map(p=>`<span class="badge" data-id="${p[2]}">${p[2]}</span>`).join('');
const sub=(a,b)=>a.map((v,i)=>v-b[i]),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],unit=a=>{const n=Math.hypot(...a);return a.map(v=>v/(n||1));};
function tri(out,a,b,c,col){if(Math.hypot(...cross(sub(b,a),sub(c,a)))<1e-10)return;const n=unit(cross(sub(b,a),sub(c,a)));for(const p of[a,b,c])out.push(...p,...n,...col);}
function quad(out,a,b,c,d,col){tri(out,a,b,c,col);tri(out,a,c,d,col);}
const topColor=[.61,.72,.82],wallColor=[.20,.68,.71],floorColor=[.25,.66,.65],sideColor=[.45,.56,.68];
const polygonEdges=boundary.map((p,i)=>[p,boundary[(i+1)%boundary.length]]);
function inside(x,y){let c=false;for(const[a,b]of polygonEdges)if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])c=!c;return c;}
function clip(a,b){const[x0,x1,y0,y1]=cfg.bounds;let lo=0,hi=1,dx=b[0]-a[0],dy=b[1]-a[1];for(const[p,q]of[[-dx,a[0]-x0],[dx,x1-a[0]],[-dy,a[1]-y0],[dy,y1-a[1]]]){if(Math.abs(p)<1e-12){if(q<0)return null;}else{const t=q/p;if(p<0)lo=Math.max(lo,t);else hi=Math.min(hi,t);if(lo>hi)return null;}}return[[a[0]+lo*dx,a[1]+lo*dy],[a[0]+hi*dx,a[1]+hi*dy]];}
function upload(data){const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.STATIC_DRAW);return{buffer,count:data.length/9};}
function addLine(out,a,b,col){for(const p of[a,b])out.push(...p,0,0,1,...col);}
let outer,pocket,outline,pathMesh,stockMesh;let stockMode=false;
function build(){
 const[x0,x1,y0,y1]=cfg.bounds,t=cfg.height,z=cfg.floor,o=[],p=[],l=[],path=[],stock=[];
 const rows=[...new Set([y0,y1,...boundary.map(p=>p[1]).filter(y=>y>y0&&y<y1)])].sort((a,b)=>a-b);
 for(let i=0;i<rows.length-1;i++){
  const ya=rows[i],yb=rows[i+1],ym=(ya+yb)/2;
  const cuts=polygonEdges.filter(([a,b])=>(a[1]>ym)!==(b[1]>ym)).map(([a,b])=>{const at=y=>a[0]+(b[0]-a[0])*(y-a[1])/(b[1]-a[1]);return{mid:at(ym),a:Math.max(x0,Math.min(x1,at(ya))),b:Math.max(x0,Math.min(x1,at(yb))) };}).sort((a,b)=>a.mid-b.mid);
  const all=[{mid:x0,a:x0,b:x0},...cuts,{mid:x1,a:x1,b:x1}];
  for(let j=0;j<all.length-1;j++){
   const A=all[j],B=all[j+1];if(B.mid<=x0||A.mid>=x1)continue;
   const isPocket=inside(Math.max(x0,Math.min(x1,(A.mid+B.mid)/2)),ym),level=isPocket?z:t;
   quad(isPocket?p:o,[A.a,ya,level],[B.a,ya,level],[B.b,yb,level],[A.b,yb,level],isPocket?floorColor:topColor);
  }
 }
 for(const[a,b]of polygonEdges){const segment=clip(a,b);if(!segment)continue;const[u,v]=segment;
  if(Math.hypot(u[0]-v[0],u[1]-v[1])<1e-9)continue;
  quad(p,[...u,z],[...v,z],[...v,t],[...u,t],wallColor);
  addLine(l,[...u,t+.03],[...v,t+.03],[.30,.85,.84]);
 }
 // Outside walls: a wall is lower where the open pocket meets a stock edge.
 for(const[a,b]of[[[x0,y0],[x1,y0]],[[x1,y0],[x1,y1]],[[x1,y1],[x0,y1]],[[x0,y1],[x0,y0]]]){
  const dx=b[0]-a[0],dy=b[1]-a[1],axis=Math.abs(dx)>Math.abs(dy)?0:1,delta=axis===0?dx:dy,ts=[0,1];
  for(const[u,v]of polygonEdges){const other=1-axis;if((u[other]>a[other])!==(v[other]>a[other])){const f=(a[other]-u[other])/(v[other]-u[other]),pos=u[axis]+f*(v[axis]-u[axis]),k=(pos-a[axis])/delta;if(k>0&&k<1)ts.push(k);}}
  ts.sort((a,b)=>a-b);
  for(let i=0;i<ts.length-1;i++){
   const u=[a[0]+dx*ts[i],a[1]+dy*ts[i]],v=[a[0]+dx*ts[i+1],a[1]+dy*ts[i+1]],mx=(u[0]+v[0])/2,my=(u[1]+v[1])/2;
   const xx=Math.max(x0+1e-5,Math.min(x1-1e-5,mx)),yy=Math.max(y0+1e-5,Math.min(y1-1e-5,my)),level=inside(xx,yy)?z:t;
   quad(o,[...u,0],[...v,0],[...v,level],[...u,level],sideColor);addLine(l,[...u,level+.03],[...v,level+.03],[.67,.78,.89]);
  }
 }
 quad(o,[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0],sideColor);
 const corners=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]];
 for(let i=0;i<4;i++){const a=corners[i],b=corners[(i+1)%4];addLine(l,[...a,0],[...b,0],[.67,.78,.89]);addLine(l,[...a,0],[...a,t],[.67,.78,.89]);quad(stock,[...a,0],[...b,0],[...b,t],[...a,t],sideColor);}
 quad(stock,[x0,y0,t],[x1,y0,t],[x1,y1,t],[x0,y1,t],topColor);quad(stock,[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0],sideColor);
 for(let i=0;i<tool.points.length-1;i++)addLine(path,[...tool.points[i],z+.10],[...tool.points[i+1],z+.10],[1,.76,.30]);
 outer=upload(o);pocket=upload(p);outline=upload(l);pathMesh=upload(path);stockMesh=upload(stock);
}
function shader(type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,`attribute vec3 position;attribute vec3 normal;attribute vec3 color;uniform mat4 matrix;varying vec3 n;varying vec3 c;void main(){gl_Position=matrix*vec4(position,1.);n=normal;c=color;}`));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,`precision mediump float;varying vec3 n;varying vec3 c;uniform float alpha;uniform float unlit;void main(){float light=.50+.50*abs(dot(normalize(n),normalize(vec3(-.4,.6,1.))));gl_FragColor=vec4(c*mix(light,1.,unlit),alpha);}`));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
const loc={position:gl.getAttribLocation(program,'position'),normal:gl.getAttribLocation(program,'normal'),color:gl.getAttribLocation(program,'color'),matrix:gl.getUniformLocation(program,'matrix'),alpha:gl.getUniformLocation(program,'alpha'),unlit:gl.getUniformLocation(program,'unlit')};
let yaw=1.0,pitch=.88,zoom=1,vp;
function multiply(a,b){const r=new Float32Array(16);for(let c=0;c<4;c++)for(let row=0;row<4;row++)for(let k=0;k<4;k++)r[c*4+row]+=a[k*4+row]*b[c*4+k];return r;}
function camera(){const[x0,x1,y0,y1]=cfg.bounds,target=[(x0+x1)/2,(y0+y1)/2,cfg.height/2],dist=Math.hypot(x1-x0,y1-y0,cfg.height)*1.65/zoom,eye=[target[0]+dist*Math.sin(pitch)*Math.cos(yaw),target[1]+dist*Math.sin(pitch)*Math.sin(yaw),target[2]+dist*Math.cos(pitch)],z=unit(sub(eye,target)),x=unit(cross([0,0,1],z)),y=cross(z,x),v=new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]),f=1/Math.tan(Math.PI/8),aspect=canvas.width/canvas.height,near=.1,far=3000,p=new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0]);vp=multiply(p,v);}
function draw(mesh,alpha,mode,unlit=0){gl.bindBuffer(gl.ARRAY_BUFFER,mesh.buffer);for(const[k,offset]of[['position',0],['normal',12],['color',24]]){gl.enableVertexAttribArray(loc[k]);gl.vertexAttribPointer(loc[k],3,gl.FLOAT,false,36,offset);}gl.uniform1f(loc.alpha,alpha);gl.uniform1f(loc.unlit,unlit);gl.drawArrays(mode,0,mesh.count);}
function render(){
 if(!outer)return;const box=canvas.getBoundingClientRect(),ratio=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(box.width*ratio);canvas.height=Math.round(box.height*ratio);gl.viewport(0,0,canvas.width,canvas.height);camera();gl.uniformMatrix4fv(loc.matrix,false,vp);gl.clearColor(.063,.094,.153,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);gl.disable(gl.BLEND);gl.depthMask(true);
 const transparent=document.querySelector('#xray').checked,showPath=document.querySelector('#path').checked;
 if(stockMode)draw(stockMesh,1,gl.TRIANGLES);
 else{draw(pocket,1,gl.TRIANGLES);if(transparent){gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false);draw(outer,.12,gl.TRIANGLES);draw(outline,.7,gl.LINES,1);gl.depthMask(true);}else{draw(outer,1,gl.TRIANGLES);draw(outline,1,gl.LINES,1);}}
 if(showPath){gl.disable(gl.DEPTH_TEST);gl.disable(gl.BLEND);draw(pathMesh,1,gl.LINES,1);gl.enable(gl.DEPTH_TEST);}
 document.querySelector('#caption').textContent=stockMode?'Исходная заготовка без обработки':`Готовая деталь · дно Z${fmt(cfg.floor)}`;
 document.querySelectorAll('.badge').forEach(el=>{el.hidden=!showPath;if(!showPath)return;const p=uniqueLabels.get(+el.dataset.id),v=[p[0],p[1],cfg.floor+1,1],q=[0,0,0,0];for(let row=0;row<4;row++)for(let k=0;k<4;k++)q[row]+=vp[k*4+row]*v[k];el.style.left=(q[0]/q[3]*.5+.5)*box.width+'px';el.style.top=(-q[1]/q[3]*.5+.5)*box.height+'px';});
}
for(const id of['xray','path'])document.querySelector('#'+id).addEventListener('change',render);
for(const[id,mode]of[['stock',true],['finished',false]])document.querySelector('#'+id).addEventListener('click',()=>{stockMode=mode;document.querySelector('#stock').setAttribute('aria-pressed',String(mode));document.querySelector('#finished').setAttribute('aria-pressed',String(!mode));render();});
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{zoom=1;({iso:()=>{yaw=1;pitch=.88;},top:()=>{yaw=-Math.PI/2;pitch=.015;},bottom:()=>{yaw=-Math.PI/2;pitch=Math.PI-.015;},front:()=>{yaw=Math.PI/2;pitch=Math.PI/2;}}[b.dataset.view])();render();}));
const pointers=new Map();let prevDist=0;
canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY]);prevDist=0;});
canvas.addEventListener('pointermove',e=>{if(!pointers.has(e.pointerId))return;const old=pointers.get(e.pointerId);pointers.set(e.pointerId,[e.clientX,e.clientY]);if(pointers.size===1){yaw-=(e.clientX-old[0])*.009;pitch=Math.max(.015,Math.min(Math.PI-.015,pitch+(e.clientY-old[1])*.009));}else{const p=[...pointers.values()],d=Math.hypot(p[0][0]-p[1][0],p[0][1]-p[1][1]);if(prevDist)zoom=Math.max(.45,Math.min(3,zoom*d/prevDist));prevDist=d;}render();});
for(const event of['pointerup','pointercancel'])canvas.addEventListener(event,e=>{pointers.delete(e.pointerId);prevDist=0;});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.45,Math.min(3,zoom*Math.exp(-e.deltaY*.001)));render();},{passive:false});
build();new ResizeObserver(render).observe(document.querySelector('#view'));render();
})();
