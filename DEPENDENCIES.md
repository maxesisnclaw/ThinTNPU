# Dependencies and asset licenses

The ThinTNPU source in this repository is licensed under Apache-2.0. That license does not relicense its dependencies, trained weights, or input assets. No upstream source trees, SDK libraries, trained checkpoints, or sample images are bundled.

| Dependency | Tested version / terms | Role |
|---|---|---|
| PyTorch | 2.8.0; BSD-3-Clause | CPU-built tensor runtime and PrivateUse1 interfaces. |
| Ultralytics v8/v11 | 8.3.233 in the unified-family environment; AGPL-3.0 or separately obtained commercial terms | Model classes and forwards, including C2f, C3k2/C3k, C2PSA and Detect. Install separately. |
| Ultralytics YOLOv5 | Separately supplied compatible upstream revision; check its AGPL-3.0 or separately obtained commercial terms | Classic C3/stem/Detect model definitions. Supply an upstream-constructed fused `nn.Module`; ThinTNPU does not bundle the upstream tree, checkpoints or an application-specific loader. |
| Ultralytics v26 | 8.4.189 in a separate frontend environment; AGPL-3.0 or separately obtained commercial terms | Direct-distance dictionary Detect, one-to-one fusion, C3k/PSABlock and residual SPPF model definitions. Install separately; no weights are bundled. |
| pybind11 | Headers provided by PyTorch; BSD-style license | Native frozen-state guard binding. |
| Ascend CANN / AscendC | CANN 9.0.0; vendor terms | Compiler, kernel APIs and runtime. Obtain the SDK separately and comply with its use and redistribution terms. |
| torch_npu | 2.8.0.post1 in the comparison environment; upstream terms | Optional native benchmark baseline only; not required or imported by the Thin process. |

The Apache-2.0 grant applies to ThinTNPU's own code only. It does not establish that an application combining this code with Ultralytics can be redistributed or offered as a service solely under Apache-2.0. Review the applicable Ultralytics terms, including AGPL obligations or a commercial license, for your usage. Weights and datasets have their own licenses; supply assets you have permission to use. Only load trusted PyTorch checkpoints and reference fixtures.

This dependency boundary covers model classes and forwards across v5, v8, v11 and v26, not just the checkpoint-loading API. Architecture adapters do not remove upstream obligations. Check the license of the exact upstream revision and assets you use; historical YOLOv5 revisions may have different terms. Weights remain independent inputs, and no custom checkpoint loader or model asset is redistributed here.
