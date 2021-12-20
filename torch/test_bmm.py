import torch
import sys
from timeit import default_timer as timer


if len(sys.argv) != 5:
    print("too few args.")

batch = int(sys.argv[1])
m = int(sys.argv[2])
n = int(sys.argv[3])
k = int(sys.argv[4])

input1 = torch.randn(batch, m, k)
input2 = torch.randn(batch, k, n)

flop = 2.0 * batch * m * n * k * 100.0 / 1000.0 / 1000.0 / 1000.0

for i in range(5):
    res = torch.bmm(input1, input2)

time_start = timer()
for i in range(100):
    res = torch.bmm(input1, input2)
time_end = timer()
time_total = time_end - time_start

gflops = flop / time_total

print("gflop = %f" % flop)
print("gflops = %f" % gflops)
