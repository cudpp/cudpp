#!/usr/bin/env python

import os

for cta_size in [64, 128, 256, 512]:
  for sort_exp in range(10,25):
    sort_size = pow(2, sort_exp)
    os.system("bin/win32/Release/cudpp_perfrig.exe --sort --radix --n=%d --chunkSize=%d --perf" % (sort_size, cta_size))

for sort_exp in range(10,25):
  sort_size = pow(2, sort_exp)
  os.system("bin/win32/Release/cudpp_perfrig.exe --sort --radix --n=%d --perfcpu" % sort_size)
