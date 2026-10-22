# Dependencies and asset licenses

The ThinTNPU source in this repository is licensed under Apache-2.0. That license does not relicense its dependencies, trained weights, or input assets. No upstream source trees, SDK libraries, trained checkpoints, or sample images are bundled.

| Dependency | Tested version / terms | Role |
|---|---|---|
| PyTorch | 2.8.0; BSD-3-Clause | CPU-built tensor runtime and PrivateUse1 interfaces. |
| Ultralytics | 8.3.233; AGPL-3.0 or separately obtained commercial terms | YOLO model loading, model classes and forwards used by the optional model integration. |
| pybind11 | Headers provided by PyTorch; BSD-style license | Native frozen-state guard binding. |
| Ascend CANN / AscendC | CANN 9.0.0; vendor terms | Compiler, kernel APIs and runtime. Obtain the SDK separately and comply with its use and redistribution terms. |
| torch_npu | 2.8.0.post1 in the comparison environment; upstream terms | Optional native benchmark baseline only; not required or imported by the Thin process. |

The Apache-2.0 grant applies to ThinTNPU's own code only. It does not establish that an application combining this code with Ultralytics can be redistributed or offered as a service solely under Apache-2.0. Review the applicable Ultralytics terms, including AGPL obligations or a commercial license, for your usage. Weights and datasets have their own licenses; supply assets you have permission to use. Only load trusted PyTorch checkpoints and reference fixtures.
