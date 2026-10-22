import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import math
from pathlib import Path
import torch
import torch.nn.functional as F
import thin_backend

torch.ops.load_library(str(Path(__file__).parent / 'thin_backend/libthin_blocked.so'))
torch.ops.load_library(str(Path(__file__).parent / 'thin_backend/libconv_flow.so'))
torch.set_num_threads(1)
cases = [(32,320,64,3,2),(32,160,32,3,1),(64,160,64,1,1),(96,160,64,1,1),
         (64,80,64,3,1),(128,80,128,3,1),(128,80,256,3,2),(128,40,128,3,1),
         (256,40,256,3,1),(256,40,512,3,2),(768,40,256,1,1),(512,20,128,3,1),
         (64,20,64,3,1),(128,80,17,1,1),(128,40,17,1,1),(128,20,17,1,1)]
checks = 0
with torch.inference_mode():
    torch.manual_seed(42)
    x = torch.randn(1,3,640,640).half()
    w = (torch.randn(32,3,3,3) / math.sqrt(27)).half()
    b = torch.randn(32).half()
    y = torch.ops.thin_native_flow.stem(x.to('npu'),w.to('npu'),b.to('npu'))
    expected = F.silu(F.conv2d(x.float(),w.float(),b.float(),2,1))
    torch.testing.assert_close(torch.ops.thin_blocked.unpack(y).cpu().float(), expected, atol=4e-3,rtol=0)
    print('STEM_PASS',flush=True)
    for ci,size,co,k,s in cases:
        x = torch.randn(1,ci,size,size).half()
        bound = 1 / math.sqrt(ci*k*k)
        w = torch.empty(co,ci,k,k).uniform_(-bound,bound).half()
        b = torch.empty(co).uniform_(-bound,bound).half()
        bp = F.pad(b,(0, (co+31)//32*32-co)).to('npu')
        wd = w.to('npu')
        pw = torch.ops.thin_backend.conv2d_prepare(wd)
        xd = torch.ops.thin_blocked.pack(x.to('npu'))
        ref = F.conv2d(x.float(),w.float(),b.float(),s,k//2)
        for fused in (False,True):
            for nchw in ((True,) if co%16 else (False,True)):
                yd = torch.ops.thin_native_flow.conv(xd,pw,bp,co,k,s,fused,nchw)
                actual = yd.cpu() if nchw else torch.ops.thin_blocked.unpack(yd).cpu()
                expected = F.silu(ref) if fused else ref
                torch.testing.assert_close(actual.float(),expected,atol=2e-3,rtol=0)
                checks += 1
        print('FLOW_PASS',ci,size,co,k,s,flush=True)
print('FLOW_CONV_ALL_PASS',checks,flush=True)
