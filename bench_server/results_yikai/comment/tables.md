### Feed + Train at eight workers

Seconds, median of three runs. Lower is better.

| Workload | HF main | PR #2501 | PR #2501 + toki counting | HF main (8 workers, 512 MiB only) |
| --- | ---: | ---: | ---: | ---: |
| Code / ByteLevel | 30.642 | 4.659 | **2.579** | 30.642 |
| English / ByteLevel | 13.692 | 2.857 | **1.628** | 13.789 |
| Chinese / ByteLevel | 472.585 | **13.888** | 14.087 | 471.759 |
| Chinese / HF Whitespace | 211.369 | **7.471** | 7.539 | 211.369 |

### Train only at eight workers

Seconds, median of three runs. Lower is better.

| Workload | HF main | PR #2501 | PR #2501 + toki counting | HF main (8 workers, 512 MiB only) |
| --- | ---: | ---: | ---: | ---: |
| Code / ByteLevel | 23.627 | **1.364** | 1.384 | 23.627 |
| English / ByteLevel | 10.829 | **0.782** | 0.784 | 10.829 |
| Chinese / ByteLevel | 463.483 | **11.367** | 11.493 | 463.483 |
| Chinese / HF Whitespace | 205.498 | **5.421** | 5.501 | 205.498 |

### Matched-input memory

Peak RSS in GiB, eight workers, 512 MiB.

| Workload | HF main | PR #2501 | PR #2501 + toki counting | HF main (8 workers, 512 MiB only) |
| --- | ---: | ---: | ---: | ---: |
| Code / ByteLevel | 2.483 | 0.500 | 0.427 | 2.483 |
| English / ByteLevel | 1.244 | 0.243 | 0.295 | 1.244 |
| Chinese / ByteLevel | 36.961 | 5.097 | 4.929 | 36.961 |
| Chinese / HF Whitespace | 18.754 | 2.705 | 2.652 | 18.754 |
