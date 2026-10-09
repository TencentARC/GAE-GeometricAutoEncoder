// CPU-only DOM/WebGL stub: tests editor events and numerical JS/Python parity.
// The stub does not establish visual/WebGL rendering correctness.
// Run: node tests/probes/check_camera_editor.cjs
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const {spawnSync} = require('child_process');
const rootPath = process.argv[2] || require('path').resolve(__dirname, '../..');
let draws = 0;
const gl = new Proxy({getShaderParameter:()=>true, getProgramParameter:()=>true,
  getAttribLocation:()=>0, getParameter:()=>[1,64], isContextLost:()=>false,
  drawArrays:()=>{draws++;}}, {get:(x,k)=>k in x?x[k]:(k.toUpperCase()===k?1:()=>({}))});
const drawing = new Proxy({}, {get:()=>()=>{}});
class Element {
  constructor(tag='div') {this.tag=tag;this.dataset={};this.style={cssText:'',setProperty(k,v){this[k]=v;}};this.listeners={};this.children=[];this.width=672;this.height=378;this.parentElement=null;}
  querySelector(selector) {
    const match=e=>selector===e.tag || selector===`[data-role="${e.dataset.role}"]`;
    return this.children.find(match)||this.children.map(e=>e.querySelector(selector)).find(Boolean)||null;
  }
  appendChild(e){e.parentElement=this;this.children.push(e);return e;}
  replaceWith(e){const p=this.parentElement;p.children[p.children.indexOf(this)]=e;e.parentElement=p;this.parentElement=null;}
  replaceChildren(){for(const e of this.children)e.parentElement=null;this.children=[];}
  remove(){if(this.parentElement){const p=this.parentElement;p.children=p.children.filter(e=>e!==this);this.parentElement=null;}}
  setAttribute(name,value){this[name]=value;}
  addEventListener(name,fn,options={}){
    const list=this.listeners[name] ||= [];list.push(fn);
    options.signal?.addEventListener('abort',()=>{this.listeners[name]=list.filter(f=>f!==fn);},{once:true});
  }
  emit(name,event={}){for(const fn of [...(this.listeners[name]||[])])fn({preventDefault(){},stopPropagation(){},...event});}
  getBoundingClientRect(){return{x:0,y:0,width:672,height:378};}
  setPointerCapture(){}
  getContext(kind){return kind==='2d'?drawing:gl;}
  toDataURL(){return 'data:image/jpeg;base64,'+Buffer.from(String(draws)).toString('base64');}
}
const root=new Element(), canvas=new Element('canvas'), reset=new Element('button');
canvas.dataset.role='camera-view';reset.dataset.role='reset';root.appendChild(canvas);root.appendChild(reset);
const data=key=>({positions:[[0,0,2],[.1,0,2]],colors:[[255,0,0],[0,255,0]],width:672,height:378,
  K:[[537.6,0,336],[0,537.6,189],[0,0,1]],pivotDepth:2.22,imageKey:key});
const props={value:null};let watched, rafId=0, timerId=0;
const rafs=new Map(),timers=new Map();
const browser=new Element();browser.devicePixelRatio=1;
const context={window:browser,element:root,props,watch:(key,fn)=>watched=fn,
  document:{createElement:tag=>new Element(tag)},AbortController,
  ResizeObserver:class{observe(){}disconnect(){}},
  requestAnimationFrame:fn=>{rafs.set(++rafId,fn);return rafId;},cancelAnimationFrame:id=>rafs.delete(id),
  setTimeout:fn=>{timers.set(++timerId,fn);return timerId;},clearTimeout:id=>timers.delete(id),
  performance:{now:()=>0},console,Float32Array};
vm.createContext(context);
vm.runInContext(fs.readFileSync(rootPath+'/scripts/demo/camera_editor.js','utf8'),context);
const api=browser.gaeCameraEditor, plain=x=>JSON.parse(JSON.stringify(x));
const role=name=>root.querySelector(`[data-role="${name}"]`);
const flushRAF=time=>{const callbacks=[...rafs.values()];rafs.clear();callbacks.forEach(fn=>fn(time));};
assert.equal(api.ready,false);assert.throws(()=>api.serialize());
props.value=data('first');watched();flushRAF(0);
api.moveDraft({x:.2,z:.1},true);assert.ok(api.getState().yaw<0);api.reset();
assert.equal(api.ready,true);assert.equal(api.getKeyframes().length,1);assert.equal(api.finalized,false);
assert.throws(()=>api.serialize(),/final camera/);
api.moveDraft({x:.2,z:.12,yaw:8});assert.equal(api.addKeyframe(),true);
const frozen=plain(api.getKeyframes()[1]);
assert.ok(api.getState().x>frozen.x && api.getState().z>frozen.z,'next draft should extend the captured motion');
api.moveDraft({x:-.25,z:.3,yaw:-10});assert.deepEqual(plain(api.getKeyframes()[1]),frozen);
assert.throws(()=>api.serialize(),/final camera/);
assert.equal(api.addKeyframe(),true);api.moveDraft({x:.1,y:.04,z:.4,yaw:5,pitch:2});assert.equal(api.setFinal(),true);
assert.equal(api.getKeyframes().length,4);assert.equal(api.finalized,true);
const source=plain(api.getKeyframes());
const scene=role('scene-view');scene.emit('pointerdown',{button:0,pointerId:1,clientX:20,clientY:20});
scene.emit('pointermove',{pointerId:1,clientX:160,clientY:70});scene.emit('pointerup');
assert.deepEqual(plain(api.getKeyframes()),source);
api.selectCamera(1);assert.equal(api.moveDraft({x:.5}),true);assert.equal(api.getKeyframes()[1].x,.5);api.cancelEdit();assert.deepEqual(plain(api.getKeyframes()),source);
api.selectCamera(1);api.moveDraft({x:.3});const committed=JSON.parse(api.serialize());assert.equal(committed.keyframes[1].x,.3);assert.equal(api.getKeyframes()[1].x,.3);
for(const views of [17,33,81,161]) {
 api.setViews(views);const payload=JSON.parse(api.serialize());
 const expectedRun=spawnSync('python3',['-c','import sys,json; from scripts.demo.direct_camera import end_view_poses; d=json.load(sys.stdin); print(json.dumps(end_view_poses(d["payload"],d["scene"],d["views"]).tolist()))'],{input:JSON.stringify({payload,scene:data('first'),views}),cwd:rootPath,encoding:'utf8'});
 assert.equal(expectedRun.status,0,expectedRun.stderr);const expected=JSON.parse(expectedRun.stdout);let err=0;
 for(let i=0;i<views;i++){const pose=api.getPoseAt(i/(views-1));for(let r=0;r<3;r++){for(let c=0;c<3;c++)err=Math.max(err,Math.abs(pose.rotation[r*3+c]-expected[i][r][c]));err=Math.max(err,Math.abs(pose.position[r]-expected[i][r][3]));}}
 assert.ok(err<2e-6,`pose parity at ${views} views: ${err}`);
}
api.setViews(81);const saved=plain(api.getKeyframes());props.value=data('first');watched();assert.deepEqual(plain(api.getKeyframes()),saved);assert.equal(api.finalized,true);
api.togglePlayback();flushRAF(2500);assert.equal(api.playing,true);api.togglePlayback();
api.selectCamera(api.getKeyframes().length-1);api.deleteSelected();assert.equal(api.finalized,false);assert.throws(()=>api.serialize(),/final camera/);
assert.equal(api.setFinal(),true);api.continuePath();assert.equal(api.finalized,false);
props.value=data('keyboard');watched();
const keyboardStage=role('scene-view').parentElement;
keyboardStage.emit('keydown',{code:'KeyW',target:role('scene-view')});
for(let i=1;i<=40;i++)flushRAF(i*16);
assert.ok(api.getState().z>.1);
keyboardStage.emit('keydown',{code:'Enter',target:role('scene-view'),repeat:false});
const keySaved=plain(api.getKeyframes()[1]);
for(let i=41;i<=60;i++)flushRAF(i*16);
assert.deepEqual(plain(api.getKeyframes()[1]),keySaved);
keyboardStage.emit('focusout',{relatedTarget:null});const stopped=plain(api.getState());flushRAF(3000);assert.deepEqual(plain(api.getState()),stopped);
keyboardStage.emit('keydown',{code:'KeyW',target:{closest:()=>({})}});flushRAF(4000);assert.deepEqual(plain(api.getState()),stopped);
props.value=data('long-dolly');watched();api.dolly(2.22*5);assert.equal(api.getState().yaw,0);assert.equal(api.getState().pitch,0);assert.ok(api.getState().z>11);api.moveDraft({yaw:240,pitch:70});assert.equal(api.getState().yaw,240);assert.equal(api.getState().pitch,70);
props.value=data('delete-draft');watched();api.addKeyframe();api.moveDraft({x:.5,z:.3,yaw:12});const pending=plain(api.getState());api.selectCamera(1);api.deleteSelected();assert.deepEqual(plain(api.getState()),pending);
props.value=data('new-image');watched();assert.equal(api.getKeyframes().length,1);assert.equal(api.finalized,false);
for(let i=0;i<4;i++){api.moveDraft({x:i*.1});assert.equal(api.addKeyframe(),true);}
assert.equal(api.addKeyframe(),false);assert.equal(api.setFinal(),true);assert.equal(api.getKeyframes().length,6);
api.selectCamera(0);assert.equal(api.editSelected(),false);assert.equal(api.deleteSelected(),false);
const presetScene=data('preset');
presetScene.presetPath={keyframes:[
 {id:'start',role:'start',time:0,x:0,y:0,z:0,yaw:0,pitch:0},
 {id:'preset-1',role:'end',time:1,x:.4,y:0,z:.8,yaw:12,pitch:0},
]};
props.value=presetScene;watched();assert.equal(api.loadPreset(),true);
assert.equal(api.finalized,true);assert.equal(JSON.parse(api.serialize()).keyframes[1].x,.4);
const presetCanvas=role('scene-view');watched();assert.equal(role('scene-view'),presetCanvas);
assert.equal(api.getKeyframes()[1].x,.4);
props.value=data('no-preset');watched();assert.equal(api.loadPreset(),false);
props.value=null;watched();assert.equal(api.ready,false);assert.throws(()=>api.serialize());api.dispose();
console.log('PASS: fixed snapshots; explicit Final gate; navigation isolation; edit/cancel/save; End deletion/reopening; image reset; capacity; 17/33/81/161-frame JS/Python parity');
