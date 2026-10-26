# Detection selection generalization

The input contract remains inference-only FP16 `[B,A,4+nc]` with nonnegative strides. Output is FP32 `[B,min(max_det,A),6]`. Selection is exact class maximum, stable anchor top-k, gathered-pair top-k and final packing; no threshold, IoU suppression, clipping or approximation is introduced.

## Dispatch properties

| Property | Implementation | Preserved invariant |
|---|---|---|
| Every initial run | Vector proposal sort of 128 records | Exact score key plus unique local rank; original payload moved as uint32 words |
| Anchor-contiguous, aligned class planes | Sixteen-class blocked vector reduction | First NaN or first equal class wins; raw score bits survive |
| Anchor-contiguous, unaligned class planes | Per-class vector comparison and selection | Same class order and original payload |
| Other layouts | Direct stride-aware reads | No materialization or host selection |
| K from 1 through 512 | Merge retained prefixes rounded to 16 records | Discarded records already have at least K predecessors in their own run |
| K greater than 512 | Complete merge-path sort | Same score/index total order without restricting K |
| Gather working set fits UB | Vector class-plane gather and vector record assembly | Flattened `[K,nc]` order is unchanged |
| Other gather working sets | Direct stride-aware reads and record assembly | Same input contract and ordering |

The capacities are local-memory tiling choices. They do not reject inputs beyond those capacities and do not select a kernel by model name, image shape, class identity, score distribution or private fixture content.

## Executable coverage

`verify_selection_flow.py` provides 47 fixtures and compares all eight trace tensors to a stable raw-bit reference. The same fixtures can be saved for two independently loaded implementations and compared as raw bytes.

- Contiguous, transposed, gapped/offset and expanded/broadcast layouts.
- Multiple batches, zero batch, zero anchors, zero K, `K>A`, minimum K and duplicate anchors.
- Initial sort boundaries A=1/127/128/129/257 and full detection A=3360/8400.
- Retained-prefix boundaries K=511/512/513 with tied scores and A=1025.
- Class reduction boundaries nc=1/15/16/17/31/80/257, plus nc=1031.
- All 65,536 half encodings as class candidates, and all encodings in a complete K=65536 ordered output.
- Infinities, subnormals, signed zeros, NaN signs and payloads, including signaling NaNs.
- Production/trace equality, eager/capture equality and changed-input replay.

The stable reference uses int16 gathers for half payload movement because CPU half gather quiets signaling NaNs. The official unstable `topk` diagnostic remains distinct; score equality does not imply tied anchor or label equality.

Full-network acceptance uses three input shapes and complete bus/zidane outputs. Its raw-byte comparison includes all output tensor leaves, intermediate/layer snapshots, decoded inputs and device selection traces. These private model/input artifacts belong in external evidence, not the source branch.
