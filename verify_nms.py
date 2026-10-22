import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import ctypes
import gc
import json
import math
from pathlib import Path
import runpy
import statistics
import sys
import time
import numpy as np
import torch
import thin_backend
from kernel_timer import time_nms_kernel

torch.set_num_threads(1)
root=Path('evidence-nms')
root.mkdir(exist_ok=True)
results=[]
stages=[]

def snapshot(stage):
    assert not any(k=='torch_npu' or k.startswith('torch_npu.') for k in sys.modules)
    parents={}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try: parents[int(p.name)]=int((p/'stat').read_text().rsplit(')',1)[1].split()[1])
            except (OSError,ValueError,IndexError): pass
    found={os.getpid()}
    while True:
        new={p for p,parent in parents.items() if parent in found}-found
        if not new: break
        found |= new
    assert found=={os.getpid()}, found
    maps=Path('/proc/self/maps').read_text()
    for token in ('torch_npu','aclnn','libopcompiler','libtbe'):
        assert token not in maps,token
    rollup=Path('/proc/self/smaps_rollup').read_text()
    pss=int(next(x.split()[1] for x in rollup.splitlines() if x.startswith('Pss:')))/1024
    row=dict(stage=stage,pid=os.getpid(),pss_mib=pss,torch_npu=False,descendants=[])
    stages.append(row)
    (root/(stage+'-smaps_rollup.txt')).write_text(rollup)
    (root/'maps.txt').write_text(maps)
    print(json.dumps(row),flush=True)

def reference(b,s,thr,score_thr,limit):
    order=np.argsort(-s,kind='stable')
    order=order[s[order].astype(np.float64)>score_thr]
    areas=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    keep=[]
    while len(order) and len(keep)<limit:
        i=order[0]; keep.append(int(i)); rem=order[1:]
        lo=np.maximum(b[i,:2],b[rem,:2]); hi=np.minimum(b[i,2:],b[rem,2:])
        wh=np.maximum(hi-lo,0)
        inter=wh[:,0]*wh[:,1]
        with np.errstate(divide='ignore',invalid='ignore'):
            iou=inter/(areas[i]+areas[rem]-inter)
        order=rem[~(iou.astype(np.float64)>thr)]
    return keep

def check(b,s,thr,st,limit,label):
    bn=torch.from_numpy(b.copy()).to('npu'); sn=torch.from_numpy(s.copy()).to('npu')
    t=time.perf_counter_ns()
    inds,count=torch.ops.thin_backend.nms(bn,sn,float(thr),float(st),limit)
    elapsed=(time.perf_counter_ns()-t)/1e6
    assert inds.device.type=='npu' and count.device.type=='npu'
    n=count.cpu().item(); host=inds.cpu().tolist(); actual=host[:n]
    expected=reference(b,s,thr,st,limit)
    assert actual==expected,(label,len(s),thr,st,limit,actual[:50],expected[:50])
    assert host[n:]==[-1]*(limit-n),(label,'tail')
    assert np.array_equal(bn.cpu().numpy(),b) and np.array_equal(sn.cpu().numpy(),s)
    row=dict(label=label,m=len(s),iou=thr,score_threshold=st,max_out=limit,kept=n,op_ms=elapsed,exact_order=True)
    results.append(row)
    print(json.dumps(row),flush=True)
    return bn,sn

snapshot('loaded')
# Existing operators and NMS share this extension and this Python process.
runpy.run_path('verify_all.py',run_name='__main__')
runpy.run_path('verify_ops.py',run_name='__main__')
snapshot('existing_operators')
for seed in (7,19,43):
    rng=np.random.default_rng(seed)
    for m in (0,1,3,7,31,65,127,1023,1024,1025,2000,4096):
        xy=rng.uniform(-500,500,(m,2)).astype(np.float32)
        wh=rng.uniform(1,350,(m,2)).astype(np.float32)
        b=np.concatenate((xy,xy+wh),axis=1)
        s=rng.permutation(np.arange(m,dtype=np.float32))/max(m,1)
        for thr in (0.0,0.3,0.5,0.7,1.0):
            check(b,s,thr,-float('inf'),m,'random-'+str(seed))
        check(b,s,0.5,0.2,min(300,m),'threshold-cap-'+str(seed))
        check(b,s,0.5,2.0,5,'none-'+str(seed))
        check(b,s,0.5,-float('inf'),0,'zero-cap-'+str(seed))
    snapshot('random-'+str(seed))

for m in (3,33,129,1025,4097):
    b=np.tile(np.array([[10,20,30,40]],np.float32),(m,1))
    s=np.arange(m,dtype=np.float32)
    check(b,s,0.5,-float('inf'),m,'identical')
    check(b,s,1.0,-float('inf'),min(300,m),'iou-equality')
b=np.array([[0,0,0,0],[0,0,0,0],[0,0,10,10],[0,0,10,10],[10,0,20,10]],np.float32)
s=np.arange(5,dtype=np.float32)
check(b,s,0.0,-float('inf'),10,'degenerate-touching')
b=np.array([[0,0,10,10],[0,0,10,10],[20,0,30,10],[20,0,30,10]],np.float32)
s=np.ones(4,np.float32)
check(b,s,0.5,-float('inf'),10,'ties')
b=np.array([[0,0,3,1],[1,0,4,1]],np.float32)
s=np.array([0.75,0.5],np.float32)
for thr in (math.nextafter(0.5,0.0),0.5,math.nextafter(0.5,1.0)):
    check(b,s,thr,-float('inf'),5,'iou-double-threshold')
for st in (math.nextafter(0.5,0.0),0.5,math.nextafter(0.5,1.0)):
    check(b,s,1.0,st,5,'score-double-threshold')

bench=[]
for m in (64,256,1024,4096,10000):
    rng=np.random.default_rng(m)
    xy=rng.uniform(0,640,(m,2)).astype(np.float32)
    wh=rng.uniform(1,200,(m,2)).astype(np.float32)
    b=np.concatenate((xy,xy+wh),axis=1)
    s=rng.permutation(np.arange(m,dtype=np.float32))/m
    bn,sn=check(b,s,0.5,0.25,300,'benchmark')
    wall=[]; cpu=[]
    for _ in range(7):
        t=time.perf_counter_ns()
        torch.ops.thin_backend.nms(bn,sn,0.5,0.25,300)
        wall.append((time.perf_counter_ns()-t)/1e6)
        t=time.perf_counter_ns(); reference(b,s,0.5,0.25,300)
        cpu.append((time.perf_counter_ns()-t)/1e6)
    row=dict(m=m,max_out=300,npu_resident_op_ms_median=statistics.median(wall),
             numpy_cpu_ms_median=statistics.median(cpu),npu_samples_ms=wall,cpu_samples_ms=cpu)
    event_timing,event_kept=time_nms_kernel(bn,sn,0.5,0.25,300)
    assert event_kept==reference(b,s,0.5,0.25,300)
    row.update(event_timing)
    bench.append(row)
    print('BENCH '+json.dumps(row),flush=True)
snapshot('all_checks_untrimmed')
gc.collect(); ctypes.CDLL(None).malloc_trim(0)
snapshot('trimmed')
(root/'modules.txt').write_text('\n'.join(sorted(sys.modules)))
(root/'results.json').write_text(json.dumps(dict(pid=os.getpid(),autoload=os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'],
    torch_version=torch.__version__,cases=results,stages=stages,benchmark=bench),indent=2))
print('NMS_ALL_PASS '+str(len(results)),flush=True)
