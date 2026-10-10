# HWX coefficient restoration and graph construction

These offline Python 3.12 programs restore the compiled FP16 face-branch coefficients of one archived H18 resource and insert them into a plain host-FP32 ONNX graph. They do not execute Apple binaries, access a device or implement Photos model selection, preprocessing, normalization or ANE arithmetic. No Apple model/compiler file is distributed here.

## Public parser versus this reconstruction

The external BSD-3-Clause project [coreml_to_ane_hwx](https://github.com/freedomtan/coreml_to_ane_hwx/tree/0da81de0551ec8b0c01f4e346d9e13129257da2d), commit `0da81de0551ec8b0c01f4e346d9e13129257da2d`, supplied register syntax/names and descriptions of coefficient DMA, sparse and palettized modes. Its parser locates byte ranges and describes their format; it did not restore the numerical arrays reported here. It is cited, not vendored or executed by these programs.
The supplied code parses sections/symbols and task writes, associates compiler coefficient ranges with graph layers, decodes lane/bitmap layouts, checks complete effective FP16 control tensors, and constructs the ONNX initializers. The exact original script hashes and path/import/text-only edits are listed in SOURCE_MANIFEST.json.

## Mapping, order, padding and precision

- `inspect_hwx.py`: parse the BEEFFACE container, file-backed sections, symbols, v11+ burst/scatter task register writes and coefficient DMA bounds. An end-of-section symbol is not a buffer. Symbols sharing an address alias a lane rather than adding channels.
- `map_graph.py`: explicit graph/channel/kernel geometry plus graph precedence, one coefficient task per weighted layer, and distinct task assignments. Missing registers stay unknown. The face output is `flatten_output`; the captured face mapping is unique under this conditional one-task model, not a proof of every compiler fusion or native model-selection path.
- Dense tensor order is `(C, K/groups, Ny, Nx)` = output/input-per-group/height/width. A lane contains an optional 128-byte activation LUT, then tiles of optional FP16 biases followed by input-element-major, output-channel-within-tile FP16 weights. Output tile first index is `(block*lanes+lane)*tile`.
- Dense tile is **`1 << (NECfg & 7)`**, with NECfg register 0x40. Omitted NECfg uses the observed compiler default table value **3**, giving tile 8. The original static H18 subtype-selection chain remains a conditional application; it is not promoted to runtime proof. Seven target layers Conv_337/349/361/373/385/397/409 explicitly use tile 32. The obsolete tile 16 cap is not used.
- Lane boundaries/tails and output/spatial/terminal bitmap padding are checked, not inferred as learned coefficients. Dense lane tails must be zero. The sigmoid prefix has 86 serialized bytes plus 42 alignment zeros; its compiler table/constants and FP32 evaluation sequence are preserved.
- Sparse streams have a little-endian 24-bit payload length and little-endian 16-bit occupancy masks. Present entries retain their FP16 bits; absent entries represent effective zero. Spatial parity2 and stored/output tile padding are verified by inverse encoding. Omitted source signed-zero/subnormal information cannot be recovered.
- NPZ arrays store little-endian FP16. The graph widens each restored array to FP32 without changing tensor order. All 282 graph initializers have the expected widened bytes. Original FP32 bias bits are not recovered. Controls with FP32 source weights/biases are explicitly rounded to FP16 before comparison.

## Required Apple resources

Obtain the public iOS 26.6.1 build 23G83 restore images yourself. Target archive: `iPhone18,5_26.6.1_23G83_Restore.ipsw` (iPhone17e; archived IPSW SHA1 `9decc7f50e88597893449a38a8f73939c9ad35df`). Compiler/static-chain archive: `iPhone12,1_26.6.1_23G83_Restore.ipsw` (iPhone11; archived IPSW SHA1 `94b8cfcd85de64dc6bb18287c340d5abe92109c3`). These are firmware identities, not personal-device acquisitions.
The archived control-copy receipts bind file hashes, not a definitive extraction-device model. The reproducible resource recipe uses the same 23G83 Vision resources from the public firmware images above, retaining the H18 variants and requiring every listed file SHA. Do not infer the extraction device from an H18 filename or silently replace it with H12/H13. The exact acquisition-side device identity of those control copies remains unestablished by their receipts.

| Resource | Firmware / internal path | Required SHA-256 |
|---|---|---|
| facerec_fp4.0_fa1.4_md2_2_fp16.espresso.net | iPhone18,5 /23G83 root filesystem; `System/Library/PrivateFrameworks/VisionCore.framework/facerec_fp4.0_fa1.4_md2_2_fp16.espresso.net` | `efe558270f01bde6fbbb92803af078f7e6bbe6d09851af950ab29cd2f6ffcbaf` |
| facerec_fp4.0_fa1.4_md2_2_fp16.espresso.shape | iPhone18,5 /23G83 root filesystem; `System/Library/PrivateFrameworks/VisionCore.framework/facerec_fp4.0_fa1.4_md2_2_fp16.espresso.shape` | `f9829c40a28d59f19693a55bee86d0e290537469825182ad75c5de1d1af9cd01` |
| facerec_fp4.0_fa1.4_md2_2_fp16_dev_op.H18.espresso.hwx | iPhone18,5 /23G83 root filesystem; `System/Library/PrivateFrameworks/VisionCore.framework/facerec_fp4.0_fa1.4_md2_2_fp16_dev_op.H18.espresso.hwx` | `84d98755394a3c2167f68293541e60dccee0bc38ae47b90720563c8a6bb505da` |
| ANECompiler | iPhone12,1 /23G83 dyld cache; `System/Library/PrivateFrameworks/ANECompiler.framework/ANECompiler` | `5b4dc5f332d16b454171e82a989f169585a170b190fd178c9956ed31f22c20d4` |
| 6ziz6uinva_opt.H18.espresso.hwx | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/6ziz6uinva_opt.H18.espresso.hwx` | `9a133a38d37926608c31c9edf0587b555f0fec5c7998c6513f00198b85e471c9` |
| 6ziz6uinva_opt.espresso.net | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/6ziz6uinva_opt.espresso.net` | `f2e2e617da9f085cdfa2f7be4fff2881c24669bcba1c7ce572ae78580d51377b` |
| 6ziz6uinva_opt.espresso.weights | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/6ziz6uinva_opt.espresso.weights` | `65d3ea140a034be8c6fd129eb2b5fca974f3b410ac0507de8c1b0a093858b828` |
| anodv4_drop6_fp16.H18.espresso.hwx | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/anodv4_drop6_fp16.H18.espresso.hwx` | `8a4ea56eee6c21a829e67848456110e1ffc4249452d45787f87ef28f1cd6d838` |
| anodv4_drop6_fp16.espresso.net | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/anodv4_drop6_fp16.espresso.net` | `51abd1bc7b3c3a3beb7b5a81477fb88381712950797b2c4dd193c6152c847e96` |
| anodv4_drop6_fp16.espresso.weights | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/anodv4_drop6_fp16.espresso.weights` | `0f5b0ba6cf4f7b2d4f5f511f16eb7e4dab76f8d62b0f9a647717b88ddf9cfbf2` |
| learned-matting-1512x2016.H18.espresso.hwx | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/learned-matting-1512x2016.H18.espresso.hwx` | `641624cfce2019e04ad5875f56231974d74425f15f906c2488566c542605a117` |
| learned-matting-1512x2016.espresso.net | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/learned-matting-1512x2016.espresso.net` | `583facbe73b55f8e003a19374ffc636d51e0576db0e0485686effda6e60faee6` |
| learned-matting-1512x2016.espresso.weights | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/learned-matting-1512x2016.espresso.weights` | `7ba174e871b25c901a84c1b79abcca246a8cd4f118cc33d8d3e323471f5d642a` |
| saliency_attention_box_head_i4fgq3rswb_fp16.H18.espresso.hwx | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/saliency_attention_box_head_i4fgq3rswb_fp16.H18.espresso.hwx` | `06dc6fe839a4765ba52be83dd3b5e3f13a4052c79ee3cdeb7c237ff3074aeb71` |
| saliency_attention_box_head_i4fgq3rswb_fp16.espresso.net | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/saliency_attention_box_head_i4fgq3rswb_fp16.espresso.net` | `d2c9b7f61ae7cbd3e58c57a26888c1d21d94533d8b1d6d5d74590ee5f8b40f6c` |
| saliency_attention_box_head_i4fgq3rswb_fp16.espresso.weights | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/saliency_attention_box_head_i4fgq3rswb_fp16.espresso.weights` | `071232e0505c2e28a2ad2450b2fab72ae63c21c6a0da0e808119d4aea8f2c8af` |
| saliency_objectness_boxes_head_ecvqeduzc7_46800_fp16.H18.espresso.hwx | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/saliency_objectness_boxes_head_ecvqeduzc7_46800_fp16.H18.espresso.hwx` | `84ae9eacae20210da2d6977a0c10ebed70d17f38bf0f99fa23735132785c8e72` |
| saliency_objectness_boxes_head_ecvqeduzc7_46800_fp16.espresso.net | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/saliency_objectness_boxes_head_ecvqeduzc7_46800_fp16.espresso.net` | `9b3a0e16329f9fc1b82e6ed4fcf9637fae7673ddca8df48955bc43e52fdcc4ea` |
| saliency_objectness_boxes_head_ecvqeduzc7_46800_fp16.espresso.weights | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/saliency_objectness_boxes_head_ecvqeduzc7_46800_fp16.espresso.weights` | `e3c31df7e1c71aabd08a96d5826bd55a7585a621e76adad14eb1ca597373ae75` |
| solo_landmarks_s9min6ugm8_opt.H18.espresso.hwx | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/solo_landmarks_s9min6ugm8_opt.H18.espresso.hwx` | `0ec1e1e00799ea682187f654164f505cf2f5ae75ace456df64ad43a3f6ed373b` |
| solo_landmarks_s9min6ugm8_opt.espresso.net | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/solo_landmarks_s9min6ugm8_opt.espresso.net` | `28eec7ba33864fe49644c30d60b96498e8fb5cc183eb94bdbd4d4c2ffdbc4050` |
| solo_landmarks_s9min6ugm8_opt.espresso.weights | 23G83 Vision resource recipe above; `System/Library/Frameworks/Vision.framework/solo_landmarks_s9min6ugm8_opt.espresso.weights` | `c3d20b2449603da3c924935eb018ec6d623c6d86e4089cc7ae1417d3ff0bb58e` |

Extract and decrypt the firmware filesystem with the external [ipsw CLI](https://github.com/blacktop/ipsw/releases/tag/v3.1.672). The archived extraction used ipsw 3.1.672 and 7-Zip 24.09. The target FileSystem image is `094-95970-089.dmg.aea` (iPhone18,5); the iPhone12,1 image is `094-95954-089.dmg.aea`. SystemOS/shared image names must be obtained from each BuildManifest/image listing; do not assume a filesystem image contains all framework resources. The archived shared DMG candidates are 094-96441-093.dmg, 094-95379-093.dmg and 094-96961-092.dmg; their identity/type must be checked before choosing one.
Example extraction commands (use your own paths and inspect the tool help for the selected release):

```text
ipsw extract --dmg fs -o extracted iPhone18,5_26.6.1_23G83_Restore.ipsw
ipsw fw aea -o decrypted extracted/23G83__iPhone18,5/094-95970-089.dmg.aea
7zz l decrypted/094-95970-089.dmg
7zz e decrypted/094-95970-089.dmg -omodel-inputs "System/Library/PrivateFrameworks/VisionCore.framework/facerec_fp4.0_fa1.4_md2_2_fp16*"
ipsw extract --dyld -o dyld-inputs iPhone12,1_26.6.1_23G83_Restore.ipsw
ipsw dyld extract dyld-inputs/dyld_shared_cache_arm64e ANECompiler -o compiler-inputs
```

For each of the six control stems below, extract `.H18.espresso.hwx`, `.espresso.net` and `.espresso.weights` from its exact Vision.framework path in the matching decoded firmware/shared image. An APFS listing must parse cleanly and its declared physical size must equal the decoded file size. Preserve original IPSW/resource bytes; use `freeze_inputs.py SOURCE NEW_COPY NEW_RECEIPT` to hash/copy before parsing. Never execute ANECompiler; it is static input. INPUTS.json checks the reviewed file hashes and fails closed.

## Six models and 148 selected convolution layers

| Model stem | Selected layers |
|---|---:|
| 6ziz6uinva_opt | 25 |
| anodv4_drop6_fp16 | 42 |
| learned-matting-1512x2016 | 21 |
| saliency_attention_box_head_i4fgq3rswb_fp16 | 29 |
| saliency_objectness_boxes_head_ecvqeduzc7_46800_fp16 | 29 |
| solo_landmarks_s9min6ugm8_opt | 2 |

CONTROL_LAYERS.json lists every model/layer identifier, value count and verified lane/tile size. The checks cover 148 selected full convolution tensors, not all layers of six complete networks. Expected totals: 3,052,688 weight values + 11,915 bias values = 6,129,206 bytes of effective FP16. Four selected layers have no bias. The independent archive audit also used a separate output-channel-to-byte-address equation; the public main checker preserves the original decode_dense full-tensor comparisons. Sparse 6-layer controls are separate and are not added to the 148 dense count.

## Minimal execution

From this folder, using Python 3.12:

```text
python -m pip install -r requirements.txt
python -B self_check.py
python -B export_face_tensors.py --net model-inputs/facerec_fp4.0_fa1.4_md2_2_fp16.espresso.net --shape model-inputs/facerec_fp4.0_fa1.4_md2_2_fp16.espresso.shape --hwx model-inputs/facerec_fp4.0_fa1.4_md2_2_fp16_dev_op.H18.espresso.hwx --compiler compiler-inputs/ANECompiler --mapping face_mapping.json --lane-layout lane_layout.json --sparse-results sparse_results.json --sigmoid-prefix verify_sigmoid_prefix.json --ipsw ipsw --output-dir restored-new
python -B build_onnx.py restored-new/face_tensors.npz --net model-inputs/facerec_fp4.0_fa1.4_md2_2_fp16.espresso.net --shape model-inputs/facerec_fp4.0_fa1.4_md2_2_fp16.espresso.shape --onnx face_candidate-new.onnx
python -B check_controls.py --controls-dir control-inputs --output controls-new.json
```

Outputs must not exist. Unknown/mismatched input SHA is rejected before restoration output creation. NPZ writing fixes ZIP create_system=3 so the output bytes are identical across operating systems; array bytes, names, order, dates and other archive fields retain the original NumPy format. The current public graph script checks 308 declared intermediate shapes and runs the original zeros/ones/seeded-random host-FP32 smoke tests. It does not simulate ANE arithmetic.

## Analytical sidecars and regeneration

- face_mapping.json: conditional geometry/dependency assignment. Regenerate with `python -B map_graph.py --net NET --shape SHAPE --hwx HWX`; use its constraint_solver face mapping, retaining SAT and alternate_face_assignment=UNSAT and all 175 face assignments. Do not select an arbitrary alternative.
- lane_layout.json: complete known-control comparisons and layout annotations. Regenerate with `python -B verify_lane_layout.py control-inputs`; retain the complete JSON stdout. check_controls.py asserts the reviewed six-model 148-layer totals.
- sparse_results.json: compiler-derived sparse layout/coverage and known-control/target checks. Regenerate with `python -B verify_sparse.py --controls-dir control-inputs --hwx HWX --net NET --mapping face_mapping.json --output-dir sparse-new`. Its six sparse target records include four mapped face layers; the two auxiliary task records are not silently mapped to face layers. This uses the original compare/decoder logic. It is not a generic sparse/palettized-format solver.
- verify_sigmoid_prefix.json: static compiler LUT reconstruction and task/lane prefix evidence. Regenerate with `python -B verify_sigmoid_prefix.py --compiler COMPILER --hwx HWX --ipsw ipsw --output sigmoid-new.json`. The 128-byte prefix contains 86 serialized LUT bytes and 42 zero alignment bytes.
These sidecars contain derived annotations/hashes, not Apple coefficient payloads. Published path-only provenance is normalized to basenames; layer names, shapes, offsets, arithmetic and ordering are retained. Lane metadata was updated to the verified 148-layer record; it affects the restoration report hash, not reconstructed coefficient values. Regenerated data must be reviewed against the existing annotations before updating INPUTS.json pins.

## Verified results and current archive-format boundary

Windows Python 3.12.10 / NumPy 2.4.6: both original and cleaned restorers produce 282 arrays (175 weights + 107 biases), 4,263,680 weight values and 26,368 bias values. Their names/order/dtypes/shapes/raw bytes match the archived reference exactly. Nine synthetic boundary checks pass, including exact archive-field preservation apart from creator-system metadata.
**NPZ SHA-256: `ced7baa922975aadc8ae26595327d47c749a6b7973790daeb55b2e0b9e63d513`.** A fresh restoration with the approved metadata-only NPZ writer reproduces every reference file byte. The earlier unmodified NumPy Windows output, preserved separately as `be2f07448caf356c536a440af22508e010a7ba7b01af305f39d78a7985299b5c`, differed in exactly 282 creator-system metadata bytes (Windows 0 versus Unix 3). No original output file was patched. This is deterministic serialization, not weight reconstruction, arithmetic tuning or Apple runtime behavior.
Both original and cleaned ONNX builders reproduce **`ea5f959113b172174c5ef001300f9a92a1c56bbd450191d92f342eda191ca49d`** byte-for-byte: 483 nodes, 282 initializers, IR10/opset13, 308 face layers and 175 weighted layers. All initializer shapes and widened FP32 bytes match the restored arrays. This artifact equality is not native Photos/ANE execution parity.
Original/cleaned control outputs agree exactly and reproduce 148 layers, 3,052,688 / 11,915 values and 6,129,206 bytes. Compiler source selection, source FP32 loss, omitted sparse signed-zero/subnormal bits and full Photos preprocessing/normalization remain separate limits.

## Files and provenance

The flat folder retains the requested restore/control helpers, their graph-build dependencies and the sparse-sidecar regeneration helper. SOURCE_MANIFEST.json records original file names/hashes and modification kinds. INPUTS.json binds every reviewed Apple/analytical input; CONTROL_LAYERS.json names every selected control. No external parser, Apple executable/model/resource, device acquisition, database adapter or application runtime integration is included.
