### Feed + Train at eight workers

Seconds, median of three runs. Lower is better.

| Workload | HF main | PR #2501 | small trainer | HF main (8 workers, 512 MiB only) |
| --- | ---: | ---: | ---: | ---: |
| Code / ByteLevel | 30.642 | **4.659** | 8.718 | 30.642 |
| English / ByteLevel | 13.692 | **2.857** | 3.907 | 13.789 |
| Chinese / ByteLevel | 472.585 | **13.888** | 24.957 | 471.759 |
| Chinese / HF Whitespace | 211.369 | **7.471** | 21.040 | 211.369 |

### Train only at eight workers

Seconds, median of three runs. Lower is better.

| Workload | HF main | PR #2501 | small trainer | HF main (8 workers, 512 MiB only) |
| --- | ---: | ---: | ---: | ---: |
| Code / ByteLevel | 23.627 | **1.364** | 2.075 | 23.627 |
| English / ByteLevel | 10.829 | **0.782** | 1.165 | 10.829 |
| Chinese / ByteLevel | 463.483 | **11.367** | 17.684 | 463.483 |
| Chinese / HF Whitespace | 205.498 | **5.421** | 14.613 | 205.498 |

### Matched-input memory

Peak RSS in GiB, eight workers, 512 MiB.

| Workload | HF main | PR #2501 | small trainer | HF main (8 workers, 512 MiB only) |
| --- | ---: | ---: | ---: | ---: |
| Code / ByteLevel | 2.483 | 0.500 | 1.018 | 2.483 |
| English / ByteLevel | 1.244 | 0.243 | 0.506 | 1.244 |
| Chinese / ByteLevel | 36.961 | 5.097 | 14.442 | 36.961 |
| Chinese / HF Whitespace | 18.754 | 2.705 | 8.679 | 18.754 |
