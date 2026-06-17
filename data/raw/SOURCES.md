# Raw data sources (not tracked in git)

These shapefiles are vendored locally to regenerate `data/thailand_admin_centroids.json`
via `scripts/build_thailand_admin_centroids.py`. They are excluded from git (see
`.gitignore`) because they are large, third-party-derived binary data.

All sourced from https://github.com/prasertcbs/thailand_gis, which itself converts
the UN OCHA Thailand COD-AB boundaries (Royal Thai Survey Department):
https://data.humdata.org/dataset/cod-ab-tha

| Local folder         | Source path in thailand_gis repo                                              | Level                    |
|-----------------------|---------------------------------------------------------------------------------|---------------------------|
| `adm1_province/`      | `tha_adm1/tha_adm1_province_shapefile.zip`                                      | Province (ADM1), official P-code |
| `adm2_district/`      | `tha_adm2/tha_admbnda_adm2_shapefile.zip`                                       | District (ADM2), official P-code |
| `adm3_subdistrict/`   | `tambon_simplify/tha_admbnda_adm3_rtsd_20220121_shapefile/` (dated 2022-01-21)  | Subdistrict (ADM3), official P-code |
| `village_2012/`       | `village/TH_VILLAGE2012.zip`                                                    | Village, 2012 vintage, **no official P-code** |

To regenerate from scratch: `git clone https://github.com/prasertcbs/thailand_gis`
and copy the corresponding folders/zips here, then run the build script.

All shapefiles must be read with `encoding="utf-8"` (Thai `_TH` attribute columns
otherwise come out as mojibake — they're UTF-8 but lack a `.cpg` file so readers
default-guess the wrong legacy encoding).
