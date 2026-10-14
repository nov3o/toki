### Feed + Train at eight workers

Seconds, median of three runs. Lower is better.

| Workload | HF main | PR #2501 | PR #2501 + toki counting |
| --- | ---: | ---: | ---: |
| Code / ByteLevel | 30.642 | 4.659 | **2.579** |
| English / ByteLevel | 13.692 | 2.857 | **1.628** |
| Chinese / ByteLevel | 472.585 | **13.888** | 14.087 |
| Chinese / HF Whitespace | 211.369 | **7.471** | 7.539 |

### Train only at eight workers

Seconds, median of three runs. Lower is better.

| Workload | HF main | PR #2501 | PR #2501 + toki counting |
| --- | ---: | ---: | ---: |
| Code / ByteLevel | 23.627 | **1.364** | 1.384 |
| English / ByteLevel | 10.829 | **0.782** | 0.784 |
| Chinese / ByteLevel | 463.483 | **11.367** | 11.493 |
| Chinese / HF Whitespace | 205.498 | **5.421** | 5.501 |

### Matched-input memory

Peak RSS in GiB, eight workers, 512 MiB.

| Workload | HF main | PR #2501 | PR #2501 + toki counting |
| --- | ---: | ---: | ---: |
| Code / ByteLevel | 2.483 | 0.500 | 0.427 |
| English / ByteLevel | 1.244 | 0.243 | 0.295 |
| Chinese / ByteLevel | 36.961 | 5.097 | 4.929 |
| Chinese / HF Whitespace | 18.754 | 2.705 | 2.652 |
