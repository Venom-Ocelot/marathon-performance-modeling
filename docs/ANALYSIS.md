# Analysis guide

The project asks what explains differences between race fields and what predicts an individual runner's performance. Those questions require different units of analysis and different information.

## Notebook chapters

| Chapters | Question | Design |
| --- | --- | --- |
| 1–8 | How do Berlin demographics and weather relate to finishing time? | 629,545 eligible finishes, 1999–2018; annual standardized comparisons and future-year prediction. |
| 9 | Does temperature replicate in Chicago, and what about moisture and wind? | 26 main-analysis race years; estimated RH from dew point; alternative specifications. |
| 10 | Do different race fields remain different after demographic adjustment? | 1,624,823 records from 199 supported race-years across 21 races. |
| 11 | How does Boston pacing vary, and can 10-km pace predict the remaining distance? | 2015–2017 splits; future-year prediction and an identity-overlap sensitivity check. |
| 12 | How useful is a previous marathon? | Train on 2015→2016 matched Boston transitions, test on 2016→2017. |
| 13 | Is recent home weather associated with Boston 2024 slowdown? | 17,169 eligible U.S. records; 14 pre-race days; regional clustering and sensitivity checks. |
| 14 | Which candidate sources and findings withstand scrutiny? | Source exclusions, data audits, and exploratory multiple-testing adjustment. |
| 15 | What can actual training logs add? | Strictly pre-race training windows; a logged-duration proxy; Boston-to-Chicago transfer with disjoint athlete IDs. |

## Decisions that affect interpretation

- Berlin 2019 contains only male records and is excluded from the initial study. Historic age codes are categories, not exact ages.
- Chicago 2007 was halted during the race. It is excluded from the primary weather fit and restored in sensitivity analysis.
- Chicago temperature/dew-point values are interpreted as Fahrenheit, a documented source assumption. Wind effects remain in source units because the CSV does not specify units.
- Repeated anonymized rows in Berlin cannot safely be treated as duplicate people. Boston's conflicting 2024 bib records are different: 22 rows with inconsistent outcomes are excluded from the main home-weather analysis.
- Name-based historical matches are probable identities, not verified IDs. Match ambiguity and location agreement are handled explicitly.
- Home weather is a proxy for potential exposure, not a measurement of actual outdoor training or acclimatization.
- Training logs contain zero-activity day rows and coarse estimated age groups. Race-day logged duration may include recording artifacts or other running; it is not an official finish time.
- Finisher-only datasets cannot measure how weather changes the probability of dropping out. Observational group differences do not establish causal effects.

## Verification

`make check` validates the committed notebook, aggregate results, report images, provenance manifests, and repository contents without source data. This is the check used by GitHub Actions.

`make validate` needs the raw inputs. It independently reconstructs standardized annual means, regression coefficients, simple prediction baselines, cohort counts, ZIP joins, training windows, and separation of training/test athlete IDs.

`make execute` is the full reproducibility check: it executes every notebook cell and regenerates the evidence and reports. Source hashes must match the recorded manifests.
