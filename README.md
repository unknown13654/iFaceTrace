# iFaceTrace

## Repository layout

```text
iFaceTrace/                          repository root
├── README.md
├── iFaceTrace/                      the application (no installation)
│   ├── iFaceTrace.exe               start this file
│   └── _internal/                   runtime, models, third-party licenses
├── paper_manifest/                  photo lists of the experiments in the paper
│   ├── README.md
│   ├── vector_correspondence_9450.csv
│   ├── deletion_library_1350.csv
│   ├── external_search_N_FD_180.csv
│   ├── external_search_U_FD_179.csv
│   └── nonlibrary_people_8100.csv
└── hwx_parser/                      FPV15 weight restoration from the compiled
    ├── README.md                    model (.hwx), Python source; start here
    ├── export_face_tensors.py       .hwx -> face_tensors.npz
    ├── build_onnx.py                face_tensors.npz -> face_candidate.onnx
    ├── check_controls.py            148-layer check on six other networks
    └── ...                          decoders, mapping data, self_check.py
```

## Requirements

- Windows 10 or 11, x64

## Run

1. Download the repository (Code > Download ZIP, then extract it) or clone it.
2. Start `iFaceTrace\iFaceTrace.exe`.

The executable is not signed. If Windows SmartScreen shows a warning, select "More info" and then "Run anyway".
iFaceTrace writes its working files to `%USERPROFILE%\iFaceTrace`.

## Input

In the Backup tab, select one of these folders with Browse, and then select Open:

```text
case1\                     an acquired folder: select case1, Media or PhotoData
└── Media\                 (some AFC tools write PhotoData\ directly into the folder you choose)
    └── PhotoData\
        └── Photos.sqlite

<backup folder>\           an iTunes backup folder (contains Manifest.db): select it,
                           then select Extract Photos
```

You can also select the `Photos.sqlite` file itself.

The Match tab shows a ranked list of candidates. The threshold slider marks the candidates above the threshold. The default threshold is 0.60 for stored FPV11 data and 0.83 for stored FPV15 data. These defaults are near the values determined on the calibration set of the paper.

## Experiment photo lists

`paper_manifest/` lists the DemogPairs photos used in the experiments of the paper. The photos are not included. See `paper_manifest/README.md`.

## Weight restoration code

`hwx_parser/` restores the FPV15 recognition weights from the Neural Engine compiled file (`.hwx`) and builds the ONNX graph that the application uses. The Apple input files are not included; `hwx_parser/README.md` lists how to obtain them, the commands, and the expected SHA-256 values.

## Third-party software

Third-party license texts and notices are in `iFaceTrace\_internal\licenses\`.
