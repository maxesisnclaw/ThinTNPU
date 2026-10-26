import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import json
from pathlib import Path
import sys

import torch
import thin_backend

torch.ops.load_library(str(Path(__file__).parent / 'thin_backend/libbmm_flow.so'))
torch.set_num_threads(1)
torch.manual_seed(2026)
with torch.inference_mode():
    for batch, m, k, n in ((4, 400, 32, 400), (4, 64, 400, 400),
                            (2, 17, 9, 23), (3, 5, 37, 1031), (1, 1, 1, 1)):
        for transposed in (False, True):
            a = (torch.randn(batch, k, m).transpose(1, 2) if transposed else torch.randn(batch, m, k)).half()
            b = (torch.randn(batch, n, k).transpose(1, 2) if transposed else torch.randn(batch, k, n)).half()
            left, right = a.to('npu'), b.to('npu')
            expected = torch.bmm(a.float(), b.float()).half()
            actual_device = torch.bmm(left, right)
            actual = actual_device.cpu()
            error = (actual.float() - expected.float()).abs()
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
            graph = torch.classes.thin_backend.RuntimeGraph()
            graph.begin()
            captured = torch.bmm(left, right)
            graph.end()
            graph.replay()
            replay = captured.cpu()
            assert torch.equal(actual.view(torch.int16), replay.view(torch.int16))
            left.copy_((a * 0.5).to('npu'))
            right.copy_((b * -0.25).to('npu'))
            changed_eager = torch.bmm(left, right).cpu()
            graph.replay()
            assert torch.equal(changed_eager.view(torch.int16), captured.cpu().view(torch.int16))
            graph.close()
            print(json.dumps(dict(shape=[batch,m,k,n], transposed=transposed,
                                  max_abs=error.max().item(), mean_abs=error.mean().item(),
                                  eager_capture_bit_equal=True, changed_input_replay_bit_equal=True)), flush=True)
assert not any(name.startswith('torch_npu') for name in sys.modules)
print('BMM_FLOW_ALL_PASS', flush=True)
