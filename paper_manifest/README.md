# DemogPairs photo lists of the experiments

These CSV files list the input photos of the paper. The files are UTF-8 and the first row has the column names.
The photos are not included. Each path is relative to the extracted `DemogPairs/` folder:
for example, `person/001.jpg` means `DemogPairs/person/001.jpg`.

## Files

Row counts do not include the header. The photo count is the number of input paths, which can differ from the number of distinct image contents.

| File | Photos | Persons | Use |
|---|---:|---:|---|
| `vector_correspondence_9450.csv` | 9,450 | 600 | All inputs of the vector correspondence evaluation |
| `deletion_library_1350.csv` | 1,350 | 150 | Library inputs of the deletion experiment, with the deletion assignment of each photo |
| `external_search_N_FD_180.csv` | 180 | 30 | External photos of the named, fully deleted persons |
| `external_search_U_FD_179.csv` | 179 | 30 | External photos of the unnamed, fully deleted persons |
| `nonlibrary_people_8100.csv` | 8,100 | 450 | Threshold calibration and evaluation photos of persons who are not in the deletion library |

The 9,450-photo list is the complete input. It is not filtered by face detection or by eligibility for the vector comparison on a device.
The external photo lists keep the original evaluation denominators, including inputs for which no face vector was generated.

## Columns

| Column | Meaning |
|---|---|
| `relative_path` | Path inside DemogPairs. The separator is `/`. |
| `person` | Person folder name of the dataset. It is not a person record of a device or a name entered by a user. |
| `demographic_group` | Group name from the dataset metadata. It is not an attribute assigned by the authors. |
| `experiment_group` | Deletion experiment group (table below). |
| `deleted` | `yes`: photo assigned for deletion. `no`: photo assigned to be kept. Only in the deletion library list. |
| `split` | `calibration`: used to set the search threshold. `evaluation`: used to evaluate that threshold. Only in the 8,100-photo list. |

The group values are `Asian_Females`, `Asian_Males`, `Black_Females`, `Black_Males`, `White_Females`, and `White_Males` (original labels).

## Deletion experiment groups

| Group | Persons | Input photos | Deleted | Kept | Condition |
|---|---:|---:|---:|---:|---|
| U_PD | 30 | 300 | 150 | 150 | Unnamed, 5 of 10 photos per person deleted |
| U_FD | 30 | 300 | 300 | 0 | Unnamed, all photos deleted |
| U_ND | 30 | 300 | 0 | 300 | Unnamed, kept |
| N_FD | 30 | 300 | 300 | 0 | Named, all photos deleted |
| filler | 30 | 150 | 0 | 150 | Kept |
| Total | 150 | 1,350 | 750 | 600 | |

Each group has 5 persons from each of the six demographic groups. Across the deletion experiment, each demographic group has 25 persons and 225 photos.
The paper shows U_ND and filler together in one row ("persons not deleted", 60 persons, 450 photos).

## One photo excluded from the external photos

Of the 360 original search candidates, `cameron_crowe/031.jpg` was excluded because its file content is the same as the deletion experiment input `cameron_crowe/026.jpg`. The SHA-256 of both files is:

```text
d2ae7fae287d0d29599661fc557e6c20e85a23cb6b67c6295685a95399914f53
```

So U_FD has 179 photos and N_FD has 180 photos. In U_FD, `cameron_crowe` has 5 photos and each of the other 59 persons has 6. U_FD `White_Males` has 29 photos; every other group and demographic group combination has 30.
A later assignment table contains the replacement photo `cameron_crowe/047.jpg`, but the original 359-photo evaluation did not use it. This list follows the original evaluation.

## Calibration and evaluation photos (8,100)

These are the 450 persons who are not among the 150 persons of the deletion library, with all 18 photos of each person.
Persons are split into 225 calibration persons (4,050 photos) and 225 evaluation persons (4,050 photos). No person and no file content is in both splits.
The split sorts `(demographic_group, person)` in ascending order and assigns persons at even positions (counted from 0) to calibration and persons at odd positions to evaluation. These lists keep that fixed split. Calibration here means setting the search threshold; it is not a training split of the face vector model.

| Group | Calibration persons | Evaluation persons | Calibration photos | Evaluation photos |
|---|---:|---:|---:|---:|
| Asian_Females | 38 | 37 | 684 | 666 |
| Asian_Males | 37 | 38 | 666 | 684 |
| Black_Females | 38 | 37 | 684 | 666 |
| Black_Males | 37 | 38 | 666 | 684 |
| White_Females | 38 | 37 | 684 | 666 |
| White_Males | 37 | 38 | 666 | 684 |
| Total | 225 | 225 | 4,050 | 4,050 |

## Relations and duplicates between the lists

- The 9,450 and 1,350 lists share no relative path. Together the 10,800 paths are the whole dataset, 1,575 + 225 = 1,800 photos for each demographic group.
- The contents are not all different: `cameron_crowe/031.jpg` (9,450 list) and `cameron_crowe/026.jpg` (1,350 list) are the same-content pair given above.
- Distinct file SHA-256 values: 9,440 in the 9,450 list, 1,348 in the 1,350 list, and 10,787 over all 10,800 paths. Content duplicates were kept to preserve the original input paths and denominators.
- The 359 external photos and the 1,350 deletion library inputs share no path and no file content. The external photos are other photos of the same persons.
- The 359 external photos and the 8,100 non-library photos are subsets of the 9,450 inputs. Do not add the row counts of the five files to get the number of distinct photos.
- The 8,100 photos contain 9 same-content pairs within the same person (8,091 distinct SHA-256 values). They share no content with the deletion library or the external photos.
- "Not in the library" refers to the deletion library only. The 9,450 inputs also contain the 150 persons of the deletion experiment, with other photos.

## Dataset source

Download the dataset from the Dataset download link on the [DemogPairs authors' page](https://ihupont.github.io/publications/2019-05-16-demogpairs). The page asks for non-commercial research use and citation of the paper; follow the terms of the original distributor. These lists do not redistribute photos or metadata files.

`DemogPairs.zip` recorded when the experiments were prepared:

```text
Size: 180935766 bytes
SHA-256: a8760ab7682cc00de537917bb45593d1691ba62fe6987a0f6ed7d077d0643cbc
```

If a downloaded zip has a different hash, do not assume it is the same version; check the contents first.

Citation: Hupont, I. and Fernández, C. (2019). *DemogPairs: quantifying the impact of demographic imbalance in deep face recognition*. IEEE FG 2019. DOI: `10.1109/FG.2019.8756625`.

Rows are sorted by `relative_path` in ascending order; the order is not the input order or the storage order on a device.
