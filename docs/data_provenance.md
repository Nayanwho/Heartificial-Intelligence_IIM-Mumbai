# Dataset provenance verification

Checked on **24 October 2026** against the [official figshare article](https://doi.org/10.6084/m9.figshare.21600933), whose metadata identifies the licence as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) and the authors as Qinpei Zhao, Jinhao Zhu, Congrong Wang and Weixiong Rao.

The downloaded archive, [figshare file 42966622](https://ndownloader.figshare.com/files/42966622), is 3,738,354 bytes. Its MD5, `4bfb61cfa506b48155fd9e841ee48e21`, matches the official metadata. All 110 included ShanghaiT2DM workbooks were compared with their corresponding archive members using SHA-256. Workbooks with different bytes were also compared by worksheet names, dimensions and parsed cell values. [Per-file hashes and differences](data_source_comparison.json) are included for inspection.

| Comparison | Files | Finding |
| --- | ---: | --- |
| Byte-identical | 98 | Exact match with the official archive |
| Different bytes, identical parsed cells | 9 | No worksheet-value differences found |
| Summary workbook | 1 | Cell J1 says `Duration of Diabetes (years)` in the mirror and `Duration of diabetes (years)` in the official archive |
| Period `2003_0_20210615.xlsx` | 1 | Mirror has 585 data rows versus 669 officially. Its rows match the official prefix; the last 84 rows are absent. The mirror ends at 19:20 on 21 June 2021; the official file ends at 16:20 on 22 June 2021. This period is excluded from the supplied scored evaluation for insufficient post-calibration follow-up. |
| Period `2029_0_20210526.xls` | 1 | Mirror has 833 data rows versus 845 officially. The first 741 match; the mirror has no valid dates after 09:14 on 3 June 2021, whereas the official file continues to 11:16 on 4 June. There are 741 dated CGM readings in the mirror versus 832 officially. This period is included in the supplied evaluation. |

## Interpretation for the submitted results

The supplied source files match **all 110 data hashes** in `results/manifest.json`. All 19 code hashes and the evaluation protocol hash also match that manifest at audit time. No dataset files, model files, evaluation outputs or modelling code were replaced during submission preparation.

The reported results therefore describe the **included mirror-derived dataset**, with the limitations above. A rerun against the official archive could change scored follow-up, training observations and headline metrics; that comparison has not been performed. The byte and cell comparison does not establish clinical validity or independently reproduce the full training run.

## Redistribution and attribution

The repository includes only the T2DM source subset, reorganised under `data/shanghai_t2dm/`; the official archive also includes other material. Source workbooks retain CC BY 4.0. Results and synthetic personas are separate derived outputs. See [data documentation](../data/README.md) for the mirror source, article citation and licence details. Dataset-author endorsement is not implied.
