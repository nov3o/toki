"""sample_one.py <data_dir> <name> <size_GB>: draw one sample from the pools (prepare_data.sample), e.g. code 10.
name = english | multilingual | code | mix (mix = the 3 pools, 1:1:1). Lets several samples be drawn in parallel."""
import sys
src = open(__file__.replace("sample_one.py", "prepare_data.py")).read().replace("\nmain()\n", "\n")
exec(src)
data, name, size = sys.argv[1], sys.argv[2], float(sys.argv[3])
srcs = ["english", "multilingual", "code"] if name == "mix" else [name]
sample(data, srcs, int(size * GB), f"{data}/{name}_{size:g}GB.txt", seed=SEED + (1 if name == "mix" else 0))
