import sys
from pylab import *
print sys.argv

fp = open(sys.argv[1], 'r')
lines = fp.readlines()
fp.close()
lines = [_.split(',')[0] for _ in lines]
lines = map(float, [_.split(':')[-1] for _ in lines])
plot(lines[:int(sys.argv[2])])
xlabel('epoch')
ylabel('loss')
savefig(sys.argv[3])
close()