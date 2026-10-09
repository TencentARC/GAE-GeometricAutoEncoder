// gr.HTML camera studio. The overview never changes the authored cameras.
(() => {
    window.gaeCameraEditor?.dispose();
    const MAX_CAMERAS = 6;
    const fields = ['x', 'y', 'z', 'yaw', 'pitch'];
    const zero = () => ({x:0, y:0, z:0, yaw:0, pitch:0});
    const copy = camera => Object.fromEntries(fields.map(key => [key, camera[key]]));
    const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
    const add = (a,b) => a.map((x,i) => x+b[i]);
    const sub = (a,b) => a.map((x,i) => x-b[i]);
    const mul = (a,s) => a.map(x => x*s);
    const dot = (a,b) => a.reduce((n,x,i) => n+x*b[i],0);
    const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
    const unit = a => mul(a,1/Math.max(1e-9,Math.sqrt(dot(a,a))));
    let cloud=null, ready=false, disposed=false, generating=false, error=null, ui=null, controller=null;
    let sceneSignature=null;
    let sceneRenderer=null, finderRenderer=null, observer=null, renderRequest=0;
    let cameras=[], draft=null, draftBeforeEdit=null, focusBeforeEdit=null, cameraBeforeEdit=null, finalized=false, selected=-1, editing=null, nextId=1;
    let views=81, playhead=0, playing=false, playbackRequest=0, playbackCamera=null;
    let mode='move', drag=null, focusPoint=null;
    const navigationKeys=new Set();let keyboardRequest=0,lastKeyboardTime=0;
    let orbit={yaw:.28,pitch:.24,distance:1.35,panX:0,panY:0};
    const cameraName = index => index===0 ? 'Start' : cameras[index]?.role==='end' ? 'Final' : `Camera ${index}`;

    function pose(camera) {
        const yaw=camera.yaw*Math.PI/180, pitch=camera.pitch*Math.PI/180;
        const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
        return {position:[camera.x,camera.y,camera.z],rotation:[cy,sy*sp,sy*cp,0,cp,-sp,-sy,cy*sp,cy*cp]};
    }
    function frame(index) { return Math.floor(cameras[index].time*(views-1)+.5); }
    function retime() {
        cameras.forEach((camera,index) => camera.time=cameras.length>1 ? index/(cameras.length-1) : 0);
    }
    function sampleAt(time) {
        if(cameras.length===1) return copy(cameras[0]);
        const t=clamp(Number(time),0,1), times=cameras.map((_,i)=>frame(i)/(views-1));
        let index=0;
        while(index<cameras.length-2 && t>times[index+1]) index++;
        const a=cameras[index],b=cameras[index+1];
        const u=clamp((t-times[index])/(times[index+1]-times[index]),0,1),h=times[index+1]-times[index];
        const slope=(i,key)=>{
            if(i===0||i===cameras.length-1) return 0;
            const h0=times[i]-times[i-1],h1=times[i+1]-times[i];
            const d0=(cameras[i][key]-cameras[i-1][key])/h0,d1=(cameras[i+1][key]-cameras[i][key])/h1;
            if(d0*d1<=0) return 0;
            const w1=2*h1+h0,w2=h1+2*h0;
            return (w1+w2)/(w1/d0+w2/d1);
        };
        return Object.fromEntries(fields.map(key=>[key,(2*u**3-3*u**2+1)*a[key]+(u**3-2*u**2+u)*h*slope(index,key)+(-2*u**3+3*u**2)*b[key]+(u**3-u**2)*h*slope(index+1,key)]));
    }

    function activeCamera() { return playbackCamera || (selected<0 ? draft : cameras[selected]) || cameras[0]; }
    function canGenerate() {
        return ready && finalized && cameras.length>=2 &&
            cameras.every((_,i)=>i===0 || frame(i)>frame(i-1));
    }
    function generationButton() {
        const parent=document.getElementById?.('gae-generate');
        const button=parent?.querySelector?.('button') || parent;
        const enabled=canGenerate()&&!generating;
        if(button && button.disabled===enabled) button.disabled=!enabled;
        const presetParent=document.getElementById?.('gae-load-preset');
        const presetButton=presetParent?.querySelector?.('button') || presetParent;
        if(presetButton) presetButton.disabled=!ready || !Array.isArray(cloud?.presetPath?.keyframes);
    }
    function loadPreset() {
        const preset=cloud?.presetPath?.keyframes;
        if(!ready || !Array.isArray(preset) || preset.length<2 || preset.length>MAX_CAMERAS) return false;
        stopPlayback();
        cameras=preset.map((camera,index)=>({...camera,id:camera.id||`preset-${index}`}));
        draft=copy(cameras.at(-1));finalized=true;selected=-1;editing=null;nextId=cameras.length;playhead=0;
        focusPoint=[0,0,cloud.pivotDepth];rebuildCards();updateUI();scheduleRender();return true;
    }
    function stopPlayback() {
        playing=false;playbackCamera=null;cancelAnimationFrame(playbackRequest);playbackRequest=0;
    }
    function selectCamera(index) {
        if(!ready) return false;
        if(editing===index) return true;
        if(editing!==null) saveEdit();
        stopPlayback();selected=index;
        if(index>=0) playhead=finalized ? frame(index)/(views-1) : 0;
        if(index>0) return editSelected();
        updateUI();scheduleRender();return true;
    }
    function constrain(camera) {
        const d=cloud.pivotDepth;
        if(!fields.every(key=>Number.isFinite(camera[key]))) throw new Error('Invalid camera pose.');
        return {x:clamp(camera.x,-10*d,10*d),y:clamp(camera.y,-10*d,10*d),
            z:clamp(camera.z,-10*d,10*d),yaw:clamp(camera.yaw,-3600,3600),pitch:clamp(camera.pitch,-85,85)};
    }
    function aimed(camera,target=focusPoint) {
        if(!target) return camera;
        const v=sub(target,[camera.x,camera.y,camera.z]);
        const angle=Math.atan2(v[0],v[2])*180/Math.PI;
        const yaw=camera.yaw+((angle-camera.yaw+180)%360+360)%360-180;
        return constrain({...camera,yaw,
            pitch:-Math.atan2(v[1],Math.hypot(v[0],v[2]))*180/Math.PI});
    }
    function moveDraft(changes,autoAim=false) {
        if(!ready || !draft || selected>=0 || playing) return false;
        draft=constrain({...draft,...changes});
        if(autoAim) draft=aimed(draft);
        else if('yaw' in changes || 'pitch' in changes) {
            const p=pose(draft),distance=Math.max(cloud.pivotDepth*.25,Math.sqrt(dot(sub(focusPoint||[0,0,cloud.pivotDepth],p.position),sub(focusPoint||[0,0,cloud.pivotDepth],p.position))));
            focusPoint=add(p.position,mul([p.rotation[2],p.rotation[5],p.rotation[8]],distance));
        }
        if(editing!==null) Object.assign(cameras[editing],copy(draft),{focus:[...focusPoint]});
        updateUI();scheduleRender();return true;
    }
    function frameCamera() {
        if(!ready) return;
        const p=pose(activeCamera());orbit.target=[...p.position];
        orbit.panX=0;orbit.panY=0;orbit.distance=1.1;scheduleRender();
    }
    function dolly(amount) {
        if(!draft || playing) return false;
        if(selected>=0) selectCamera(-1);
        const before=pose(draft),v=[before.rotation[2],before.rotation[5],before.rotation[8]];
        moveDraft({x:draft.x+v[0]*amount,y:draft.y+v[1]*amount,z:draft.z+v[2]*amount});
        const delta=sub(pose(draft).position,before.position);focusPoint=add(focusPoint,delta);
        orbit.target=add(orbit.target||[0,0,.65*cloud.pivotDepth],delta);
        const view=sceneView(),p=project(pose(draft).position,view);
        if(p.z<=.05||p.x<view.width*.12||p.x>view.width*.88||p.y<view.height*.12||p.y>view.height*.88) frameCamera();
        return true;
    }
    function addKeyframe() {
        if(!ready || finalized || editing!==null || !draft || selected>=0 || cameras.length>=MAX_CAMERAS-1) return false;
        cameras.push({id:`camera-${nextId++}`,role:'keyframe',focus:[...focusPoint],...copy(draft)});
        const last=cameras.at(-1),previous=cameras.at(-2),delta=sub(pose(last).position,pose(previous).position);
        const length=Math.sqrt(dot(delta,delta)),r=pose(last).rotation;
        const direction=length>cloud.pivotDepth*.005?unit(delta):[r[2],r[5],r[8]];
        const step=clamp(length*.65,cloud.pivotDepth*.04,cloud.pivotDepth*.12);
        const position=add(pose(last).position,mul(direction,step));
        draft=navigationKeys.size?copy(last):aimed(constrain({...copy(last),x:position[0],y:position[1],z:position[2]}));retime();selected=-1;mode='move';
        rebuildCards();updateUI(`Camera ${cameras.length-1} saved. Continue moving or adjust the next camera.`);ui.sceneCanvas.focus?.({preventScroll:true});scheduleRender();return true;
    }
    function setFinal() {
        if(!ready || finalized || editing!==null) return false;
        if(selected===cameras.length-1 && selected>0) cameras[selected].role='end';
        else if(selected<0 && draft && cameras.length<MAX_CAMERAS) cameras.push({id:`camera-${nextId++}`,role:'end',focus:[...focusPoint],...copy(draft)});
        else return false;
        finalized=true;draft=null;selected=cameras.length-1;retime();playhead=1;
        rebuildCards();updateUI('Path ready. Preview it or generate your video.');scheduleRender();return true;
    }
    function editSelected() {
        if(!ready || selected<=0 || editing!==null) return false;
        stopPlayback();draftBeforeEdit=draft?copy(draft):null;focusBeforeEdit=focusPoint?[...focusPoint]:null;editing=selected;cameraBeforeEdit={...cameras[selected],focus:cameras[selected].focus?[...cameras[selected].focus]:undefined};draft=copy(cameras[selected]);focusPoint=cameras[selected].focus?[...cameras[selected].focus]:[0,0,cloud.pivotDepth];selected=-1;
        updateUI('Drag or use the keyboard to adjust this camera. Changes save automatically; Undo reverts this edit.');scheduleRender();return true;
    }
    function saveEdit() {
        if(editing===null || !draft) return false;
        Object.assign(cameras[editing],copy(draft),{focus:[...focusPoint]});selected=editing;editing=null;
        draft=finalized ? null : copy(draftBeforeEdit || cameras.at(-1));draftBeforeEdit=null;focusPoint=focusBeforeEdit||focusPoint;focusBeforeEdit=null;
        rebuildCards();updateUI();scheduleRender();return true;
    }
    function cancelEdit() {
        if(editing===null) return false;
        cameras[editing]={...cameraBeforeEdit};
        selected=editing;editing=null;draft=finalized ? null : copy(draftBeforeEdit || cameras.at(-1));draftBeforeEdit=null;focusPoint=focusBeforeEdit||focusPoint;focusBeforeEdit=null;
        updateUI();scheduleRender();return true;
    }
    function deleteSelected() {
        const index=editing===null?selected:editing;
        if(!ready || index<=0) return false;
        if(editing!==null) saveEdit();
        selected=index;stopPlayback();const removed=cameras.splice(selected,1)[0];
        if(removed.role==='end') {finalized=false;draft=copy(removed);focusPoint=removed.focus?[...removed.focus]:[0,0,cloud.pivotDepth];selected=-1;}
        else {selected=Math.min(selected,cameras.length-1);}
        retime();rebuildCards();updateUI();scheduleRender();return true;
    }
    function continuePath() {
        if(editing!==null) saveEdit();
        if(!finalized) return false;
        stopPlayback();const end=cameras.pop();draft=copy(end);focusPoint=end.focus?[...end.focus]:[0,0,cloud.pivotDepth];finalized=false;selected=-1;
        retime();rebuildCards();updateUI('Final camera reopened. Add a keyframe or choose the new final view.');scheduleRender();return true;
    }
    function togglePlayback() {
        if(editing!==null) saveEdit();
        if(!canGenerate()) return false;
        if(playing) {stopPlayback();updateUI();scheduleRender();return true;}
        playing=true;const began=performance.now();
        const tick=now=>{
            if(!playing || !ready || disposed) return;
            playhead=clamp((now-began)/(views/12*1000),0,1);playbackCamera=sampleAt(playhead);
            ui.playhead.value=String(Math.round(playhead*(views-1)));ui.finderTitle.textContent=`Path preview · Frame ${Math.round(playhead*(views-1))}`;scheduleRender();
            if(playhead>=1) {stopPlayback();selected=cameras.length-1;updateUI();scheduleRender();}
            else playbackRequest=requestAnimationFrame(tick);
        };
        playbackRequest=requestAnimationFrame(tick);updateUI();return true;
    }
    function selectTime(time) {
        if(editing!==null) saveEdit();
        if(!canGenerate()) return false;
        stopPlayback();playhead=clamp(Number(time),0,1);playbackCamera=sampleAt(playhead);
        selected=cameras.findIndex((_,i)=>Math.abs(frame(i)/(views-1)-playhead)<1e-7);
        updateUI();scheduleRender();return true;
    }
    function setViews(value) {
        const n=Number(value);if(!Number.isInteger(n)||n<2||n>161) throw new Error('Invalid frame count.');
        views=n;if(ui) {ui.playhead.max=String(n-1);updateUI();scheduleRender();}
    }
    function setKeyTime(value) {
        const selectedKey=editing===null?selected:editing;
        if(!finalized || selectedKey<=0 || selectedKey>=cameras.length-1) return;
        const index=clamp(Math.round(Number(value)),frame(selectedKey-1)+1,frame(selectedKey+1)-1);
        cameras[selectedKey].time=index/(views-1);playhead=index/(views-1);playbackCamera=null;
        updateUI();scheduleRender();
    }

    function updateUI(message) {
        if(!ui) {generationButton();return;}
        const editable=ready && !!draft && selected<0 && !playing;
        const intermediate=cameras.filter(camera=>camera.role==='keyframe').length;
        ui.add.hidden=finalized || editing!==null;ui.final.hidden=finalized || editing!==null;
        ui.add.disabled=!editable || finalized || editing!==null || intermediate>=4;
        ui.final.disabled=!ready || finalized || editing!==null || !(selected<0 || selected===cameras.length-1 && selected>0);
        ui.final.textContent=selected>0 ? 'Use selected as final' : 'Set final frame';
        ui.play.disabled=!canGenerate();ui.play.textContent=playing ? 'Stop preview' : 'Preview path';
        ui.edit.hidden=true;ui.remove.hidden=selected<=0 && editing===null;
        ui.edit.disabled=!ready || selected<=0 || editing!==null || playing;
        ui.remove.disabled=!ready || (selected<=0 && editing===null) || playing;
        ui.save.hidden=editing===null;ui.cancel.hidden=editing===null;
        ui.continue.hidden=!finalized;ui.continue.disabled=playing;
        ui.draft.disabled=!ready || finalized || playing;ui.draft.hidden=finalized || (selected<0 && editing===null);
        ui.modeMove.hidden=true;ui.modeAim.hidden=true;ui.reset.hidden=!editable;
        ui.modeMove.disabled=!editable;ui.modeAim.disabled=!editable;
        ui.modeMove.setAttribute('aria-pressed',String(mode==='move'));
        ui.modeAim.setAttribute('aria-pressed',String(mode==='aim'));
        ui.reset.disabled=!editable;
        ui.playhead.disabled=!canGenerate();ui.playhead.value=String(Math.round(playhead*(views-1)));
        const timingKey=editing===null?selected:editing;
        ui.timing.hidden=!finalized || timingKey<=0 || timingKey>=cameras.length-1;
        if(!ui.timing.hidden) {ui.timing.min=String(frame(timingKey-1)+1);ui.timing.max=String(frame(timingKey+1)-1);ui.timing.value=String(frame(timingKey));}
        ui.count.textContent=`${intermediate} / 4 keyframes · ${finalized ? 'Final chosen' : 'Choose final when ready'}`;
        ui.finderTitle.textContent=playing || playbackCamera ? `Path preview · Frame ${Math.round(playhead*(views-1))}` : selected<0 ? editing!==null ? `Editing Camera ${editing}` : 'Draft camera view' : `${cameraName(selected)} view`;
        ui.frameLabel.textContent=`${views} frames · ${(views/12).toFixed(2)} s`;
        ui.message.textContent=message || (editing!==null ? 'Edit directly with mouse or keyboard. Switch cameras or generate to save; Undo reverts this edit.' :
            selected>=0 ? selected===0 ? 'Start is fixed. Select the draft to place a camera.' : finalized ?
                'Click a camera to adjust it, preview the path, or generate.' : 'Click a camera to adjust it, or select Draft to add the next view.' :
            'Drag the orange camera, then add a keyframe. Choose Final when your path is complete.');
        ui.status.textContent=finalized ? 'Path ready' : editing!==null ? 'Editing camera' : 'Placing cameras';
        for(const [index,card] of ui.cards.entries()) {
            card.setAttribute('aria-pressed',String(index===selected||index===editing));
            card.dataset.selected=String(index===selected||index===editing);
            card.querySelector('[data-role="card-time"]').textContent=finalized ? `Frame ${frame(index)}` : index===0 ? 'Original image' : 'Fixed camera';
        }
        ui.draft.setAttribute('aria-pressed',String(selected<0));generationButton();
    }
    function button(label,role,action,parent) {
        const node=document.createElement('button');node.type='button';node.textContent=label;node.dataset.role=role;
        node.addEventListener('click',action,{signal:controller.signal});parent.appendChild(node);return node;
    }
    function createUI() {
        element.replaceChildren();
        const shell=document.createElement('div');shell.className='camera-studio';element.appendChild(shell);
        const heading=document.createElement('div');heading.className='studio-bar';shell.appendChild(heading);
        const label=document.createElement('div');label.innerHTML='<strong>Camera path</strong><span>Place cameras in the scene</span>';heading.appendChild(label);
        const status=document.createElement('span');status.className='studio-status';heading.appendChild(status);
        const stage=document.createElement('div');stage.className='scene-stage';shell.appendChild(stage);
        const sceneCanvas=document.createElement('canvas');sceneCanvas.dataset.role='scene-view';sceneCanvas.tabIndex=0;
        sceneCanvas.setAttribute('aria-label','3D scene: drag background to orbit; drag the orange camera to move');stage.appendChild(sceneCanvas);
        const overlay=document.createElement('canvas');overlay.dataset.role='scene-overlay';overlay.className='scene-overlay';stage.appendChild(overlay);
        const hits=document.createElement('div');hits.className='scene-hit-targets';stage.appendChild(hits);
        const notice=document.createElement('div');notice.className='scene-notice';stage.appendChild(notice);
        const hint=document.createElement('div');hint.className='scene-hint';hint.textContent='Drag camera: move + auto-aim · Drag the small dot: frame your shot · Scroll: move in/out';stage.appendChild(hint);
        const finder=document.createElement('div');finder.className='camera-finder';stage.appendChild(finder);
        const finderTitle=document.createElement('div');finder.appendChild(finderTitle);
        const finderCanvas=document.createElement('canvas');finderCanvas.dataset.role='camera-view';finder.appendChild(finderCanvas);
        const tools=document.createElement('div');tools.className='studio-tools';shell.appendChild(tools);
        const modeMove=button('Move camera','move-mode',()=>{mode='move';updateUI();scheduleRender();},tools);
        const modeAim=button('Aim camera','aim-mode',()=>{mode='aim';updateUI();scheduleRender();},tools);
        const reset=button('Reset draft','reset',()=>moveDraft(zero()),tools);
        button('Frame camera','frame-camera',frameCamera,tools);
        button('Reset scene view','reset-scene',()=>{orbit={yaw:.28,pitch:.24,distance:1.35,panX:0,panY:0};scheduleRender();},tools);
        const strip=document.createElement('div');strip.className='camera-cards';strip.dataset.role='keyframe-timeline';shell.appendChild(strip);
        const actions=document.createElement('div');actions.className='studio-actions';shell.appendChild(actions);
        const add=button('+ Add keyframe','add-keyframe',addKeyframe,actions);add.className='primary-camera-action';
        const final=button('Set final frame','set-final',setFinal,actions);final.className='final-camera-action';
        const play=button('Preview path','play-camera-preview',togglePlayback,actions);
        const save=button('Done','save-camera',saveEdit,actions);
        const cancel=button('Undo changes','cancel-edit',cancelEdit,actions);
        const cont=button('Reopen final','continue-path',continuePath,actions);
        const keyboardHelp=document.createElement('span');keyboardHelp.className='keyboard-help';keyboardHelp.textContent='WASD move · Q/E height · arrows aim · Enter save · F finish';keyboardHelp.title='Click the scene first. Hold Shift to move faster. Shortcuts never run while typing.';tools.appendChild(keyboardHelp);
        const count=document.createElement('span');count.className='camera-count';actions.appendChild(count);
        const editActions=document.createElement('div');editActions.className='camera-edit-actions';shell.appendChild(editActions);
        const draftButton=button('Continue placing','select-draft',()=>selectCamera(-1),editActions);
        const edit=button('Edit selected','edit-camera',editSelected,editActions);
        const remove=button('Delete selected','delete-keyframe',deleteSelected,editActions);
        const timing=document.createElement('input');timing.type='range';timing.dataset.role='keyframe-time';
        timing.setAttribute('aria-label','Selected keyframe timing');timing.addEventListener('input',()=>setKeyTime(timing.value),{signal:controller.signal});editActions.appendChild(timing);
        const scrub=document.createElement('div');scrub.className='camera-scrub';shell.appendChild(scrub);
        const playhead=document.createElement('input');playhead.type='range';playhead.min='0';playhead.max=String(views-1);playhead.step='1';playhead.dataset.role='camera-playhead';
        playhead.setAttribute('aria-label','Camera path playback frame');playhead.addEventListener('input',()=>selectTime(Number(playhead.value)/(views-1)),{signal:controller.signal});scrub.appendChild(playhead);
        const frameLabel=document.createElement('span');scrub.appendChild(frameLabel);
        const message=document.createElement('div');message.className='camera-message';message.setAttribute('aria-live','polite');shell.appendChild(message);
        ui={shell,stage,sceneCanvas,overlay,hits,notice,finderCanvas,finderTitle,modeMove,modeAim,reset,status,strip,add,final,play,save,cancel,continue:cont,count,edit,remove,draft:draftButton,timing,playhead,frameLabel,message,cards:[],markers:new Map()};
        bindScene();
    }
    function rebuildCards() {
        if(!ui) return;ui.strip.replaceChildren();ui.cards=[];
        cameras.forEach((camera,index)=>{
            const card=button('', 'keyframe-card',()=>selectCamera(index),ui.strip);
            card.className='camera-card';card.dataset.index=String(index);
            card.setAttribute('aria-label',`${cameraName(index)} camera`);
            const image=document.createElement('img');image.alt=`${cameraName(index)} view`;image.draggable=false;
            if(finderRenderer) {
                finderRenderer.draw(pose(camera));
                image.src=ui.finderCanvas.toDataURL('image/jpeg',.75);
            }
            card.appendChild(image);
            const name=document.createElement('strong');name.textContent=cameraName(index);card.appendChild(name);
            const time=document.createElement('span');time.dataset.role='card-time';card.appendChild(time);ui.cards.push(card);
        });
        updateUI();scheduleRender();
    }

    function sceneView() {
        const rect=ui.sceneCanvas.getBoundingClientRect(),width=Math.max(1,rect.width),height=Math.max(1,rect.height);
        const d=cloud.pivotDepth,target=add(orbit.target||[0,0,.65*d],[orbit.panX*d,orbit.panY*d,0]);
        const eye=add(target,[Math.sin(orbit.yaw)*Math.cos(orbit.pitch)*orbit.distance*d,-Math.sin(orbit.pitch)*orbit.distance*d,-Math.cos(orbit.yaw)*Math.cos(orbit.pitch)*orbit.distance*d]);
        const forward=unit(sub(target,eye)),right=unit(cross([0,1,0],forward)),down=unit(cross(forward,right));
        const camera={position:eye,rotation:[right[0],down[0],forward[0],right[1],down[1],forward[1],right[2],down[2],forward[2]]};
        const focal=.9*width,K=[[focal,0,width/2],[0,focal,height/2],[0,0,1]];
        return {camera,width,height,K,right,down,forward};
    }
    function project(point,view) {
        const p=sub(point,view.camera.position),x=dot(p,view.right),y=dot(p,view.down),z=dot(p,view.forward);
        return {x:view.K[0][0]*x/z+view.width/2,y:view.K[1][1]*y/z+view.height/2,z};
    }
    function screenRay(x,y,view) {
        return unit(add(add(mul(view.right,(x-view.width/2)/view.K[0][0]),mul(view.down,(y-view.height/2)/view.K[1][1])),view.forward));
    }
    function world(camera,local) {
        const r=pose(camera).rotation;
        return [camera.x+r[0]*local[0]+r[1]*local[1]+r[2]*local[2],camera.y+r[3]*local[0]+r[4]*local[1]+r[5]*local[2],camera.z+r[6]*local[0]+r[7]*local[1]+r[8]*local[2]];
    }
    function drawOverlay(view) {
        const canvas=ui.overlay,dpr=Math.min(window.devicePixelRatio||1,2);
        canvas.width=Math.round(view.width*dpr);canvas.height=Math.round(view.height*dpr);
        const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,view.width,view.height);
        const line=(a,b,color,width=2,dashed=false)=>{
            const x=project(a,view),y=project(b,view);if(x.z<=.01||y.z<=.01) return;
            ctx.beginPath();ctx.moveTo(x.x,x.y);ctx.lineTo(y.x,y.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dashed?[5,5]:[]);ctx.stroke();ctx.setLineDash([]);
        };
        const d=cloud.pivotDepth;
        for(let n=-5;n<=5;n++) {
            const v=n*d*.16;
            line([v,0,-.45*d],[v,0,1.3*d],'rgba(118,149,174,.14)',1);
            line([-.8*d,0,v],[.8*d,0,v],'rgba(118,149,174,.14)',1);
        }
        if(cameras.length>1) {const path=Array.from({length:Math.max(81,views)},(_,i)=>pose(sampleAt(i/(Math.max(81,views)-1))).position);for(let i=1;i<path.length;i++) line(path[i-1],path[i],'#60d7ff',2);}
        if(draft && !finalized && editing===null) line(pose(cameras.at(-1)).position,pose(draft).position,'#f6b75c',2,true);
        for(const marker of ui.markers.values()) marker.hidden=true;
        const indexForId=id=>cameras.findIndex(camera=>camera.id===id);
        function hit(id,label,point,color,action,kind,pixel=null) {
            const p=pixel||project(point,view);if(p.z<=.01 || p.x<-24||p.x>view.width+24||p.y<-24||p.y>view.height+24) return;
            let node=ui.markers.get(id);
            if(!node) {
                node=document.createElement('button');node.type='button';node.dataset.role=kind||'scene-camera';node.dataset.cameraId=id;
                node.addEventListener('click',event=>{event.stopPropagation();if(!drag) node._studioAction?.();ui.sceneCanvas.focus?.({preventScroll:true});},{signal:controller.signal});
                if(kind==='camera-handle') node.addEventListener('pointerdown',event=>beginDrag(event,id),{signal:controller.signal});
                else if(id==='draft' || id==='draft-body') node.addEventListener('pointerdown',event=>beginDrag(event,'body'),{signal:controller.signal});
                else if(id!=='start' && id!=='playback') node.addEventListener('pointerdown',event=>{selectCamera(cameras.findIndex(camera=>camera.id===id));beginDrag(event,'body');},{signal:controller.signal});
                if(id==='draft') {
                    node.addEventListener('wheel',event=>{event.preventDefault();event.stopPropagation();dolly(-clamp(event.deltaY,-300,300)*cloud.pivotDepth*(event.shiftKey ? .0002 : .0008));},{signal:controller.signal,passive:false});
                    node.addEventListener('contextmenu',event=>event.preventDefault(),{signal:controller.signal});
                }
                ui.hits.appendChild(node);ui.markers.set(id,node);
            }
            node._studioAction=action;node.hidden=false;node.disabled=id==='playback';node.setAttribute('aria-label',label);node.style.left=`${p.x}px`;node.style.top=`${p.y}px`;node.style.setProperty('--camera-color',color);
            node.title=id==='draft'?'Drag to move · Right-drag to aim · Shift-drag for height · Scroll to move in/out':label;
            node.dataset.selected=String(id==='draft'||indexForId(id)===selected);
            const badge=id==='start'?'S':id==='draft'?'':id==='playback'?'':cameras.find(c=>c.id===id)?.role==='end'?'F':String(indexForId(id));
            if(id==='aim') {node.innerHTML='<span class=aim-dot></span>';return;}
            if(node.dataset.badge!==badge) {
                node.innerHTML='<svg width="17" height="14" viewBox="0 0 24 18" aria-hidden="true"><rect x="2" y="4" width="13" height="11" rx="2" fill="currentColor"/><path d="M16 7L22 4V15L16 12Z" fill="currentColor"/><rect x="5" y="1" width="6" height="3" rx="1" fill="currentColor"/></svg>';
                if(badge) {const tag=document.createElement('span');tag.className='camera-number';tag.textContent=badge;node.appendChild(tag);}
                node.dataset.badge=badge;
            }
        }
        function glyph(camera,color,label,id,index,large=false) {
            const centerPixel=project(pose(camera).position,view);
            const depth=Math.max(.001,centerPixel.z/view.K[0][0]*(large?13:9)),K=cloud.K;
            const corners=[[0,0],[cloud.width,0],[cloud.width,cloud.height],[0,cloud.height]].map(([px,py])=>{
                const y=(py-K[1][2])*depth/K[1][1];
                return [((px-K[0][2])*depth-K[0][1]*y)/K[0][0],y,depth];
            });
            const center=pose(camera).position,points=corners.map(p=>world(camera,p));
            for(let i=0;i<4;i++) {line(center,points[i],color,large?1.5:1);line(points[i],points[(i+1)%4],color,large?1.5:1);}
            const w=depth*.28,h=depth*.2;
            const body=[[-w,-h,-depth*.35],[w,-h,-depth*.35],[w,h,-depth*.35],[-w,h,-depth*.35]].map(p=>world(camera,p));
            for(let i=0;i<4;i++) line(body[i],body[(i+1)%4],color,large?1.7:1.2);
            hit(id,label,center,color,()=>selectCamera(index));
        }
        cameras.forEach((camera,index)=>{if(index!==editing)glyph(camera,index===0?'#6ce2b0':camera.role==='end'?'#ba9aff':'#a5cfff',cameraName(index),camera.id,index,index===selected);});
        if(draft) {glyph(draft,'#ffc166',editing!==null?'Editing':'Draft','draft',-1,true);}
        if(playing||playbackCamera) glyph(activeCamera(),'#ffed96','Playback','playback',-2,true);
        if(draft && selected<0 && !playing) {
            const center=pose(draft).position,a=project(center,view),b=project(world(draft,[0,0,d*.2]),view);
            let dx=b.x-a.x,dy=b.y-a.y,length=Math.hypot(dx,dy);if(length<1){dx=0;dy=-1;length=1;}
            const tip={x:a.x+dx/length*38,y:a.y+dy/length*38,z:a.z};
            ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(tip.x,tip.y);ctx.strokeStyle='rgba(255,193,102,.55)';ctx.lineWidth=1;ctx.stroke();
            hit('aim','Drag to aim the camera',center,'#ffc166',null,'camera-handle',tip);
        }
    }

    function beginDrag(event,kind) {
        if(ready && draft && selected>=0 && editing===null && kind==='body') selectCamera(-1);
        if(!ready || !draft || selected>=0 || playing) return;
        event.preventDefault();event.stopPropagation();ui.sceneCanvas.focus?.({preventScroll:true});
        const rect=ui.sceneCanvas.getBoundingClientRect(),view=sceneView();
        drag={kind,id:event.pointerId,x:event.clientX,y:event.clientY,start:copy(draft),view,rect};
        if(kind==='body' && (event.button===2 || event.altKey)) drag.kind='aim';
        else if(kind==='body' && event.shiftKey) drag.kind='height';
        if(drag.kind==='body') {
            const ray=screenRay(event.clientX-rect.x,event.clientY-rect.y,view);
            const denominator=dot(ray,view.forward);
            if(Math.abs(denominator)<.05) {drag=null;return;}
            const distance=dot(sub(pose(draft).position,view.camera.position),view.forward)/denominator;
            drag.plane=add(view.camera.position,mul(ray,distance));
        }
        ui.sceneCanvas.setPointerCapture(event.pointerId);
    }
    function stopKeyboard() {
        navigationKeys.clear();cancelAnimationFrame(keyboardRequest);keyboardRequest=0;
    }
    function keyboardTick(now) {
        keyboardRequest=0;const dt=Math.min(.05,Math.max(0,(now-lastKeyboardTime)/1000));lastKeyboardTime=now;
        if(ready && draft && selected<0 && !playing && (!drag || drag.kind==='aim')) {
            const on=key=>navigationKeys.has(key)?1:0;
            const turn=40*dt;
            if(!drag && (on('ArrowLeft')||on('ArrowRight')||on('ArrowUp')||on('ArrowDown')))
                moveDraft({yaw:draft.yaw+(on('ArrowRight')-on('ArrowLeft'))*turn,pitch:draft.pitch+(on('ArrowUp')-on('ArrowDown'))*turn});
            const p=pose(draft),r=p.rotation;
            let direction=add(mul([r[2],r[5],r[8]],on('KeyW')-on('KeyS')),mul([r[0],r[3],r[6]],on('KeyD')-on('KeyA')));
            direction=add(direction,[0,on('KeyQ')-on('KeyE'),0]);
            if(dot(direction,direction)>0) {
                const speed=cloud.pivotDepth*.18*(on('ShiftLeft')||on('ShiftRight')?3:1);
                const delta=mul(unit(direction),speed*dt);
                moveDraft({x:draft.x+delta[0],y:draft.y+delta[1],z:draft.z+delta[2]});
                const actual=sub(pose(draft).position,p.position);focusPoint=add(focusPoint,actual);
                orbit.target=add(orbit.target||[0,0,.65*cloud.pivotDepth],actual);
            }
        }
        if([...navigationKeys].some(key=>!key.startsWith('Shift'))) keyboardRequest=requestAnimationFrame(keyboardTick);
    }
    function bindScene() {
        const canvas=ui.sceneCanvas,signal=controller.signal;
        const movement=new Set(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','ShiftLeft','ShiftRight']);
        ui.stage.addEventListener('keydown',event=>{
            if(event.ctrlKey||event.metaKey||event.altKey||event.target?.closest?.('input,textarea,[contenteditable=true]')) return;
            if(['Enter','KeyF','Escape'].includes(event.code)) {
                event.preventDefault();event.stopPropagation();if(event.repeat) return;
                if(event.code==='Enter') {if(editing!==null)saveEdit();else addKeyframe();}
                else if(event.code==='KeyF') {stopKeyboard();if(editing!==null)saveEdit();setFinal();}
                else {stopKeyboard();stopPlayback();if(editing!==null)cancelEdit();updateUI();scheduleRender();}
                return;
            }
            if(!movement.has(event.code)) return;
            event.preventDefault();event.stopPropagation();
            if(!ready||!draft||selected>=0||playing) {updateUI('Click a camera to edit it, or select Draft to continue placing cameras.');return;}
            navigationKeys.add(event.code);
            if(event.shiftKey && !event.code.startsWith('Shift')) navigationKeys.add('ShiftLeft');
            if(!keyboardRequest && !event.code.startsWith('Shift')) {const now=performance.now();lastKeyboardTime=now-16;keyboardTick(now);}
        },{signal});
        ui.stage.addEventListener('keyup',event=>{navigationKeys.delete(event.code);if(!navigationKeys.size)stopKeyboard();},{signal});
        ui.stage.addEventListener('focusout',event=>{if(!ui.stage.contains?.(event.relatedTarget))stopKeyboard();},{signal});
        window.addEventListener('blur',stopKeyboard,{signal});
        document.addEventListener?.('visibilitychange',()=>{if(document.hidden)stopKeyboard();},{signal});
        canvas.addEventListener('pointerdown',event=>{
            if(event.button!==0 && event.button!==2) return;
            event.preventDefault();canvas.focus?.();
            drag={kind:event.shiftKey||event.button===2?'pan':'orbit',id:event.pointerId,x:event.clientX,y:event.clientY,startOrbit:{...orbit}};
            canvas.setPointerCapture(event.pointerId);
        },{signal});
        canvas.addEventListener('pointermove',event=>{
            if(!drag || event.pointerId!==drag.id) return;
            const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
            if(drag.kind==='orbit') {orbit.yaw=drag.startOrbit.yaw-dx*.006;orbit.pitch=clamp(drag.startOrbit.pitch+dy*.005,.08,1.05);scheduleRender();return;}
            if(drag.kind==='pan') {orbit.panX=drag.startOrbit.panX-dx/500;orbit.panY=drag.startOrbit.panY-dy/500;scheduleRender();return;}
            if(drag.kind==='height') {moveDraft({...drag.start,y:drag.start.y+dy*cloud.pivotDepth/400},true);return;}
            if(drag.kind==='aim') {moveDraft({yaw:drag.start.yaw+dx*.15,pitch:drag.start.pitch-dy*.12});return;}
            if(drag.kind==='body') {
                const ray=screenRay(event.clientX-drag.rect.x,event.clientY-drag.rect.y,drag.view);
                const denominator=dot(ray,drag.view.forward);
                if(Math.abs(denominator)<.05) return;
                const distance=dot(sub(pose(drag.start).position,drag.view.camera.position),drag.view.forward)/denominator;
                if(distance<=0) return;
                const point=add(drag.view.camera.position,mul(ray,distance));
                moveDraft({...drag.start,x:drag.start.x+point[0]-drag.plane[0],y:drag.start.y+point[1]-drag.plane[1],z:drag.start.z+point[2]-drag.plane[2]},true);return;
            }
            const axis={x:[1,0,0],y:[0,1,0],z:[0,0,1]}[drag.kind];
            if(axis) {
                const position=pose(drag.start).position,a=project(position,drag.view),b=project(add(position,mul(axis,cloud.pivotDepth*.18)),drag.view);
                const sx=b.x-a.x,sy=b.y-a.y,delta=(dx*sx+dy*sy)/Math.max(1,sx*sx+sy*sy)*cloud.pivotDepth*.18;
                moveDraft({...drag.start,[drag.kind]:drag.start[drag.kind]+delta});
            }
        },{signal});
        const stop=()=>{drag=null;if(editing!==null)rebuildCards();};
        for(const event of ['pointerup','pointercancel','lostpointercapture']) canvas.addEventListener(event,stop,{signal});
        window.addEventListener('blur',stop,{signal});
        canvas.addEventListener('wheel',event=>{event.preventDefault();orbit.distance=clamp(orbit.distance*Math.exp(clamp(event.deltaY,-300,300)*.0015),.4,30);scheduleRender();},{signal,passive:false});
        canvas.addEventListener('contextmenu',event=>event.preventDefault(),{signal});
        ui.hits.addEventListener('keydown',event=>{
            if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) return;
            const axis=event.target?.dataset?.cameraId;
            if(!['x','y','z','aim'].includes(axis) || !['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) return;
            event.preventDefault();const sign=['ArrowRight','ArrowUp'].includes(event.key)?1:-1;
            if(axis==='aim') {const field=['ArrowUp','ArrowDown'].includes(event.key)?'pitch':'yaw';moveDraft({[field]:draft[field]+sign*(event.shiftKey ? .25 : 1)});}
            else moveDraft({[axis]:draft[axis]+sign*cloud.pivotDepth*(event.shiftKey ? .005 : .02)});
        },{signal});
        canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();ready=false;stopPlayback();updateUI();ui.notice.textContent='Restoring the scene preview…';ui.notice.hidden=false;},{signal});
        canvas.addEventListener('webglcontextrestored',()=>initialize(true),{signal});
    }
    function scheduleRender() {
        if(disposed || renderRequest) return;
        renderRequest=requestAnimationFrame(()=>{renderRequest=0;render();});
    }
    function render() {
        generationButton();
        if(!ready || !ui || !sceneRenderer || !finderRenderer) return;
        const view=sceneView();sceneRenderer.draw(view.camera,{width:view.width,height:view.height,K:view.K});
        finderRenderer.draw(pose(activeCamera()));drawOverlay(view);
    }
    function createRenderer(canvas, data) {
        const gl = canvas.getContext("webgl", {
            alpha: false, antialias: false, depth: true, preserveDrawingBuffer: true,
        });
        if (!gl) throw new Error("This browser cannot show the 3D preview. Enable WebGL and reload.");
        const vertexSource = `
            precision highp float;
            attribute vec3 aPosition;
            attribute vec3 aColor;
            uniform mat3 uWorldToCamera;
            uniform vec3 uCameraPosition;
            uniform vec4 uIntrinsics;
            uniform float uSkew;
            uniform vec2 uImageSize;
            uniform float uFar;
            uniform float uPointSize;
            varying vec3 vColor;
            void main() {
                vec3 c = uWorldToCamera * (aPosition - uCameraPosition);
                float nearPlane = 0.01;
                if (c.z <= nearPlane || c.z >= uFar) {
                    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                } else {
                    vec2 pixel = vec2(
                        (uIntrinsics.x * c.x + uSkew * c.y) / c.z + uIntrinsics.z,
                        uIntrinsics.y * c.y / c.z + uIntrinsics.w);
                    vec2 ndc = vec2(2.0 * pixel.x / uImageSize.x - 1.0,
                                   1.0 - 2.0 * pixel.y / uImageSize.y);
                    float clipZ = (uFar + nearPlane) / (uFar - nearPlane) * c.z
                        - 2.0 * uFar * nearPlane / (uFar - nearPlane);
                    gl_Position = vec4(ndc * c.z, clipZ, c.z);
                }
                gl_PointSize = uPointSize;
                vColor = aColor;
            }`;
        const fragmentSource = `
            precision mediump float;
            varying vec3 vColor;
            void main() {
                vec2 d = gl_PointCoord - vec2(0.5);
                if (dot(d, d) > 0.25) discard;
                gl_FragColor = vec4(vColor, 1.0);
            }`;
        const shaders = [];
        function compile(type, source) {
            const shader = gl.createShader(type);
            shaders.push(shader);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                throw new Error(`Camera preview shader failed: ${gl.getShaderInfoLog(shader)}`);
            }
            return shader;
        }
        const program = gl.createProgram();
        let buffer = null;
        try {
            gl.attachShader(program, compile(gl.VERTEX_SHADER, vertexSource));
            gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragmentSource));
            gl.linkProgram(program);
            if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
                throw new Error(`Camera preview link failed: ${gl.getProgramInfoLog(program)}`);
            }
            const packed = new Float32Array(data.positions.length * 6);
            let count = 0, maxMagnitude = data.pivotDepth, maxColor = 0;
            for (const color of data.colors) {
                if (Array.isArray(color)) for (const channel of color) maxColor = Math.max(maxColor, channel);
            }
            const colorScale = maxColor > 1 ? 255 : 1;
            for (let i = 0; i < data.positions.length; i++) {
                const xyz = data.positions[i], rgb = data.colors[i];
                if (!Array.isArray(xyz) || xyz.length !== 3 || !xyz.every(Number.isFinite) ||
                    !Array.isArray(rgb) || rgb.length !== 3 || !rgb.every(Number.isFinite)) continue;
                packed.set(xyz, count * 6);
                packed.set(rgb.map(c => clamp(c / colorScale, 0, 1)), count * 6 + 3);
                maxMagnitude = Math.max(maxMagnitude, Math.abs(xyz[0]), Math.abs(xyz[1]), Math.abs(xyz[2]));
                count++;
            }
            if (!count) throw new Error("No valid geometry was found for this image.");
            buffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
            gl.bufferData(gl.ARRAY_BUFFER, packed.subarray(0, count * 6), gl.STATIC_DRAW);
            const location = name => gl.getUniformLocation(program, name);
            const uniforms = Object.fromEntries([
                "uWorldToCamera", "uCameraPosition", "uIntrinsics", "uSkew", "uImageSize", "uFar", "uPointSize",
            ].map(name => [name, location(name)]));
            const positionAttribute = gl.getAttribLocation(program, "aPosition");
            const colorAttribute = gl.getAttribLocation(program, "aColor");
            const pointRange = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
            return {
                draw(camera, projection = data) {
                    if (gl.isContextLost()) return;
                    const rect = canvas.getBoundingClientRect();
                    const dpr = Math.min(window.devicePixelRatio || 1, 2);
                    const width = Math.max(1, Math.round((rect.width || data.width) * dpr));
                    const height = Math.max(1, Math.round((rect.height || data.height) * dpr));
                    if (canvas.width !== width || canvas.height !== height) {
                        canvas.width = width;
                        canvas.height = height;
                    }
                    gl.viewport(0, 0, width, height);
                    gl.clearColor(0.025, 0.035, 0.05, 1);
                    gl.clearDepth(1);
                    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
                    gl.enable(gl.DEPTH_TEST);
                    gl.depthFunc(gl.LESS);
                    gl.disable(gl.BLEND);
                    gl.useProgram(program);
                    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
                    gl.enableVertexAttribArray(positionAttribute);
                    gl.vertexAttribPointer(positionAttribute, 3, gl.FLOAT, false, 24, 0);
                    gl.enableVertexAttribArray(colorAttribute);
                    gl.vertexAttribPointer(colorAttribute, 3, gl.FLOAT, false, 24, 12);
                    // WebGL reads column-major. Row-major camera-to-world R
                    // becomes R^T here, exactly the world-to-camera rotation.
                    gl.uniformMatrix3fv(uniforms.uWorldToCamera, false, new Float32Array(camera.rotation));
                    gl.uniform3fv(uniforms.uCameraPosition, camera.position);
                    gl.uniform4f(uniforms.uIntrinsics, projection.K[0][0], projection.K[1][1], projection.K[0][2], projection.K[1][2]);
                    gl.uniform1f(uniforms.uSkew, projection.K[0][1]);
                    gl.uniform2f(uniforms.uImageSize, projection.width, projection.height);
                    gl.uniform1f(uniforms.uFar, Math.max(1, maxMagnitude * 8, Math.hypot(...camera.position) + maxMagnitude * 3));
                    // Small splats cover sampling gaps without filling genuinely
                    // unseen regions; depth testing keeps nearer surfaces in front.
                    const spacing = Math.sqrt(data.width * data.height / count);
                    const size = clamp(spacing * width / data.width * 1.2, 1, 5 * dpr);
                    gl.uniform1f(uniforms.uPointSize, clamp(size, pointRange[0], pointRange[1]));
                    gl.drawArrays(gl.POINTS, 0, count);
                },
                dispose() {
                    gl.deleteBuffer(buffer);
                    gl.deleteProgram(program);
                    for (const shader of shaders) gl.deleteShader(shader);
                },
            };
        } catch (e) {
            if (buffer) gl.deleteBuffer(buffer);
            gl.deleteProgram(program);
            for (const shader of shaders) gl.deleteShader(shader);
            throw e;
        }
    }

    function cleanup() {
        stopKeyboard();stopPlayback();cancelAnimationFrame(renderRequest);renderRequest=0;drag=null;
        controller?.abort();controller=null;observer?.disconnect();observer=null;
        sceneRenderer?.dispose();finderRenderer?.dispose();sceneRenderer=null;finderRenderer=null;
        ready=false;
    }
    function initialize(force=false) {
        if(disposed) return;
        const data=props.value,previousKey=cloud?.imageKey,previousDepth=cloud?.pivotDepth;
        const nextSignature=data?.imageKey ? [data.imageKey,data.pivotDepth,data.positions?.length,data.colors?.length,
            data.width,data.height,data.K?.[0]?.[0],data.K?.[1]?.[1]].join('|') : 'empty';
        // Gradio may notify a custom HTML component more than once for the same
        // output. Rebuilding here destroys both WebGL contexts and the authored
        // camera state, which presents as a repeated page refresh.
        if(!force && ui && nextSignature===sceneSignature) return;
        sceneSignature=nextSignature;
        cleanup();controller=new AbortController();createUI();error=null;
        if(!data?.imageKey) {
            cloud=null;cameras=[];draft=null;finalized=false;selected=-1;editing=null;
            ui.notice.textContent='Choose an image to start placing cameras.';ui.notice.hidden=false;
            updateUI();scheduleRender();return;
        }
        try {
            if(!Number.isFinite(data.pivotDepth)||data.pivotDepth<=.01||!Array.isArray(data.positions)||!Array.isArray(data.colors)||data.positions.length!==data.colors.length||
               !Array.isArray(data.K)||data.K.length!==3||!data.K.every(row=>Array.isArray(row)&&row.length===3&&row.every(Number.isFinite))||data.K[0][0]<=0||data.K[1][1]<=0)
                throw new Error('The scene preview is incomplete. Choose the image again.');
            cloud=data;
            if(previousKey!==data.imageKey || previousDepth!==data.pivotDepth || !cameras.length) {
                cameras=[{id:'start',role:'start',time:0,...zero()}];draft=zero();finalized=false;selected=-1;editing=null;nextId=1;playhead=0;
                focusPoint=[0,0,data.pivotDepth];
                orbit={yaw:.28,pitch:.24,distance:1.35,panX:0,panY:0};
            }
            sceneRenderer=createRenderer(ui.sceneCanvas,data);finderRenderer=createRenderer(ui.finderCanvas,data);
            ready=true;ui.notice.hidden=true;rebuildCards();updateUI();
            observer=new ResizeObserver(scheduleRender);observer.observe(ui.sceneCanvas);observer.observe(ui.finderCanvas);
            scheduleRender();
        } catch(e) {
            error=String(e.message||e);ready=false;ui.notice.textContent=error;ui.notice.hidden=false;
            updateUI();console.error('GAE camera studio:',e);
        }
    }
    const api={
        root:element,
        get ready(){return ready&&!disposed;}, get error(){return error;}, get finalized(){return finalized;},
        get selectedIndex(){return selected;},get playing(){return playing;},get playhead(){return playhead;},get views(){return views;},
        getState(){return draft?{...copy(draft),imageKey:cloud?.imageKey,pivotDepth:cloud?.pivotDepth}:null;},
        getKeyframes(){return cameras.map(camera=>({...camera}));},
        getPose(){return pose(activeCamera());}, getPoseAt(time){return pose(sampleAt(time));}, sample(time){return {...sampleAt(time)};},
        setGenerating(value){generating=!!value;generationButton();},
        canGenerate,loadPreset,moveDraft,dolly,frameCamera,selectCamera,selectTime,addKeyframe,setFinal,editSelected,saveEdit,cancelEdit,deleteSelected,continuePath,togglePlayback,setViews,
        reset(){return moveDraft(zero());},render:scheduleRender,
        serialize(){
            if(!this.ready) throw new Error('Choose an image and wait for its scene preview.');
            if(!finalized) throw new Error('Choose the final camera with Set final frame before generating.');
            if(editing!==null) saveEdit();
            if(!canGenerate()) throw new Error('Keyframes are too close for this frame count. Adjust their timing or use more frames.');
            return JSON.stringify({schema:'camera-path-v2',interpolation:'pchip',finalized:true,imageKey:cloud.imageKey,pivotDepth:cloud.pivotDepth,
                keyframes:cameras.map(camera=>({...camera}))});
        },
        dispose(){disposed=true;cleanup();cloud=null;cameras=[];draft=null;ui=null;}
    };
    window.gaeCameraEditor=api;watch('value',initialize);initialize();
})();
