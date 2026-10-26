# YOLO11 blocked inference

The blocked inference adapter supports the C3k2/C3k, C2PSA and depthwise-separable Detect structure used by YOLO11. It reuses the anchor-free DFL and box decoder. Checkpoints and reference fixtures are external inputs, not repository assets.

## Build

Build as root in the CANN environment, from the repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build_all.sh
source env.sh
```

Use the existing `load_native_flow()`, `prepare_model(model, fuse_silu=True)` and `WholeBlocked(model)` entry points. Prepare an eval-mode model with fused batch normalization, frozen FP16 parameters and NPU placement under `torch.no_grad()`. Enable the adapter before inference or graph capture. The external input and graph lifetime contracts are described in [Whole blocked inference](WHOLE_BLOCKED.md).

## Computation and layout

- Dense convolution wrappers accept SiLU or Identity. Identity selects the existing convolution epilogue without activation; SiLU keeps FP32 accumulation, bias and activation followed by one FP16 cast.
- C3k2 and C3k reuse their module forwards over blocked channel chunks, convolution, concatenation and residual addition. They do not require new numerical kernels.
- C2PSA splits logical channels in units of C0=16. Q, K and V are logical channel ranges within the blocked QKV output. The matrix consumer interprets channel blocks, group origins and tokens directly; it does not unpack QKV into NCHW or materialize transposed Q or probability matrices. Individual head/key widths need not be C0-aligned: group addressing may cross physical channel-block boundaries, while convolution inputs and total blocked outputs retain the C0 contract. `channel_select` compacts the selected V channels once for the existing depthwise position-convolution consumer.
- Attention uses dynamic tensor–tensor `aten.bmm`: Q-transpose times K, scaling and last-axis softmax, then V times probability-transpose. The existing generic softmax handles the token axis. The position encoding remains depthwise convolution of V, not an absolute-position table.
- `bmm_flow.cpp` accepts same-device rank-three FP16 tensors with arbitrary nonnegative strides. `layout_bmm` also accepts channel-blocked `[B,C1,T,C0]` operands and logical transposition flags. Operand descriptors derive addresses from strides, including channel-block boundaries and storage offsets, without temporary contiguous operands. Bounded local-memory panels and aligned strided DMA feed a row-tiled vector outer product. Products accumulate in sequential K order in FP32 before one FP16 cast, preserving the original numerical contract. An optional blocked result performs one device-side output transpose. Work and attention storage grow with token count; there is no shape lookup or model-specific dispatch.
- `depthwise_flow.cpp` implements channelwise convolution on blocked input. The multiplier-one weight packing is `[C/16,K,K,16]`, derived from ordinary `[C,1,K,K]` parameters. Stride-one vector multiplication broadcasts weights across four pixels per repeat; arbitrary row tails and stride two retain the same FP32 product/add order. Depthwise activation is selected independently, covering both Identity position encoding and SiLU classification layers.

## Scope

The existing N=1, FP16, C0-aligned intermediate-channel contract remains. The model adapter accepts depthwise k3/s1/p1 with multiplier one; general grouped convolution, depthwise multiplier greater than one and non-aligned logical intermediate channels are not supported. **YOLO11n is unsupported** because its eight-channel intermediate layouts do not satisfy this contract. The stem, Detect and DFL contracts are still explicit model-family adapters, not an arbitrary PyTorch graph compiler.

The bmm and depthwise kernels are general numerical primitives. The C2PSA/Attention forward replacements encode Ultralytics module semantics; they are architecture adapters, not model- or image-size special cases. Each captured graph fixes its own input shape. Fresh execution and capture derive shapes from tensors and module attributes.

## Verification

```bash
python3 verify_bmm_flow.py
python3 verify_bmm_layout.py
python3 verify_depthwise_flow.py
python3 verify_yolo11.py \
  --model /path/to/fused_model.pt \
  --fixture /path/to/cpu_reference.pt \
  --output /path/to/validation.json \
  --shape-edges
```

The whole-model harness checks eager/capture equality, changed-input replay and restoration of the original input. It reports coordinate, class, raw-head, DFL and C2PSA intermediate errors against both CPU FP32 execution and the FP16-activation reference. Reference parameters are rounded to FP16 first; the FP16-activation reference performs each convolution, bias and selected activation in FP32 before a single cast, matching the backend's convolution contract. Reported CPU differences are measurements, not a claim of CPU bit equality. The harness also checks that torch_npu, aclnn, the operator compiler and TBE are absent from the process.

`--model` is a trusted PyTorch file containing `{'model': fused_module}` with separately installed public Ultralytics model classes. `--fixture` is a separately prepared trusted CPU reference file, not the simple v8 reference generated by `examples/make_reference.py`. Its `cases` mapping contains named records with `image_fp16`, `decoded`, `raw_heads`, `dfl` (a list whose first record has `output`), `c2psa` (named hook records with `out`), and `fp16_params_ref` (the matching `decoded`, `raw_heads`, `dfl` and `c2psa` reference fields). Hook keys are `c2psa.cv1`, `c2psa.cv2`, `psa.attn`, `psa.ffn`, `attn.qkv`, `attn.proj` and `attn.pe`.

Supply both artifacts independently; no checkpoint, fixture or application-specific loader is bundled. This harness's assertions and error reports are not evidence that every model or shape has already passed. Bitwise whole-model claims concern Thin eager/capture or the corresponding existing Thin fixture, not CPU FP32 output. The historical v8 benchmark JSON is not YOLO11 evidence.
