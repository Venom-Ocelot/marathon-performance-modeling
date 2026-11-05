# Data sources and attribution

Six Kaggle datasets contribute substantive analyses. A seventh small training dataset is audited but excluded from inference. Versions and exact input hashes are recorded in `data/raw/**/manifest.json`; source descriptions are saved in `data/catalog/`.

| Dataset | Version | License as listed | Use |
| --- | --- | --- | --- |
| [Berlin marathon data](https://www.kaggle.com/datasets/aiaiaidavid/berlin-marathons-data) | 2 | CC0 | Results and daily weather; initial analysis uses 1999–2018. |
| [Chicago Marathon Results](https://www.kaggle.com/datasets/runningwithrock/chicago-marathon-results) | 3 | MIT | Results, temperature, dew point, wind, and race dates. |
| [2010–2019 Fall Marathons](https://www.kaggle.com/datasets/runningwithrock/2010-2019-fall-marathons) | 1 | MIT | Comparisons among recurrent U.S. fall-marathon fields. |
| [Boston Marathon 2015–2017](https://www.kaggle.com/datasets/rojour/boston-results) | 4 | Unknown | Splits, pacing, and conservative previous-race linkage. No broader reuse license is inferred. |
| [Boston 2024 Weather and Splits](https://www.kaggle.com/datasets/runningwithrock/2024-boston-marathon-weather-and-splits) | 2 | CC BY 4.0 | U.S. runner results and home-area weather; ZIP and date normalization. |
| [Long-distance running dataset](https://www.kaggle.com/datasets/mexwell/long-distance-running-dataset) | 1 | CC BY 4.0 | Selected 2019 daily file; pre-race training and logged race-day duration proxy. |
| [Marathon Time Predictions](https://www.kaggle.com/datasets/girardi69/marathon-time-predictions) | 2 | CC0 | Audit only: 87 records, mileage including the race, missing fields, and a speed anomaly. |

## Original collection and authors

**Berlin:** the Kaggle uploader credits Beat Knechtle and Volker Scheer. [Knechtle and colleagues' primary research](https://pmc.ncbi.nlm.nih.gov/articles/PMC8293098/) describes organizer results and German Weather Service records. The downloaded CSVs do not independently identify station IDs or race dates.

**Chicago, fall marathons, and Boston 2024:** compiled by [Brian Rock / Running with Rock](https://runningwithrock.com/). Chicago combines organizer results and Weather Spark. Boston 2024 combines Boston Athletic Association results and Open-Meteo home-area temperatures. [The collector's Boston source explanation](https://runningwithrock.com/boston-marathon-splits-analysis/) describes the ZIP-based data. This project changes ZIP formatting, excludes race day from exposure, filters conflicting bib records, aggregates observations, and performs new statistical analyses.

**Boston 2015–2017:** Rodrigo Jourdan's Kaggle dataset, scraped from Boston Athletic Association results. Raw source records are not committed here.

**Daily running logs:** Kaggle mirror by mexwell of the research dataset by Leonardo A. Afonseca, Renato N. Watanabe, and Marcos Duarte. [Their PeerJ paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8958962/) documents the collection and approximate age groups. This project selects 2019 event windows, aggregates pre-race training, applies consistency filters, and models a logged-duration outcome. The [BAA confirms Boston's April 15, 2019 date](https://www.baa.org/news/baa-announces-2019-boston-marathon-registration-dates/); Chicago's date is in its event-weather input.

## Overlap and exclusions

Chicago 2010–2019 also appears in the fall sample. These support different questions and are not pooled as independent replications. People may recur across years and sources.

The [dataset selection audit](../outputs/expanded/tables/dataset_selection.csv) records nine expansion candidates. The Run Club candidate is excluded because its publisher identifies it as synthetic. Other result collections add coverage but were not selected for the missing measurements needed by these analyses.

Only manifests and source metadata are tracked under `data/`. Use `make download` to obtain the pinned inputs from Kaggle. Data licenses remain attached to their respective sources; this repository does not grant additional rights to source data.

## Supporting methods

- [EPA humidity algorithms](https://qed.epa.gov/hms/meteorology/humidity/algorithms/) — Magnus approximation for relative humidity from temperature and dew point.
- [World Athletics on Chicago 2007](https://worldathletics.org/news/news/ivuti-and-adere-clinch-chicago-dramatic-chica) — contemporary account of the race stoppage.
- [Berlin age-class documentation](https://github.com/hmelberg/berlin-marathon#age-classes) — numeric category interpretation.
