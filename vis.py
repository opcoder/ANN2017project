#encoding:utf8
import sys
from pylab import *
print sys.argv

# log = sys.argv[1]
# png = sys.argv[2]
# xlim = int(sys.argv[3])
# ylim([0, 4])
# fp = open(log, 'r')
# lines = fp.readlines()
# fp.close()
# lines = [_.split(',')[0] for _ in lines]
# lines = map(float, [_.split(':')[-1] for _ in lines])
# line = plot(lines[:xlim], 'g-', linewidth = 2)
# xlabel('epoch')
# ylabel('loss')
# savefig(png)
# close()
# exit(0)


text = u'comparision of different learning rates'
files = ['log_6.txt', 'log_8.txt', 'log_5.txt', 'log_7.txt']
legs = ['learning rate: 1.0', 'learning rate: 0.2', 'learning rate: 0.1', 'learning rate: 0.5']
lr = [1, 0.2, 0.1, 0.05]
xlims = [15000, 15000, 3000, 3000]
xlabels = ['1-1', '1-2', '1-3', '1-4']

text = u'comparision of different numbers of hidden neurons'
files = ['log_1.txt', 'log_2.txt', 'log_3.txt', 'log_4.txt']
legs = ['hidden nodes: 16', 'hidden nodes: 8', 'hidden nodes: 4', 'hidden nodes: 3']
hidden_nodes = [16, 8, 4, 3]
xlims = [15000, 15000, 15000, 15000]
xlabels = ['2-1', '2-2', '2-3', '2-4']

# text = u'comparision of different optimizion algorithms'
# files = ['log_7.txt', 'log_5.txt', 'log_9.txt']
# legs = ['algorithm:Standard\nlearning rate:0.05', 'algorithm:Standard\nlearning rate:0.1', 'algorithm:Adagrad\nlearning rate:0.1']
# xlims = [2000, 2000, 2000]
# xlabels = ['3-1', '3-2', '3-3']

# text = u'comparision of different activaitons of the hidden layer'
# files = ['log_5.txt', 'log_10.txt']
# legs = ['hidden activation:tanh','hidden activation:sigmoid']
# xlims = [10000, 10000]
# xlabels = ['4-1', '4-2']

# text = u'comparision of different fillers'
# files = ['log_5.txt', 'log_12.txt']
# legs = ['filler: gaussian','filler: uniform']
# xlims = [1000, 1000]
# xlabels = ['6-1', '6-2']

# text = u'comparision of different initial weights using uniform'
# files = ['log_12.txt', 'log_13.txt']
# legs = ['weights range: [-1, 1]','weights range: [-0.01, 0.01]']
# xlims = [1000, 30000]
# xlabels = ['7-1', '7-2']

color = 'rgby'
logs = []

figtext(0.5, 0.95, text, ha='center', fontsize='large')
for i, f in enumerate(files):
    x = i / 2 + 1
    y = i % 2 + 1
    fp = open(f, 'r')
    lines = fp.readlines()
    fp.close()
    lines = [_.split(',')[0] for _ in lines]
    lines = map(float, [_.split(':')[-1] for _ in lines])
    logs.append(lines)
    subplot(221 + i)
    step = 20
    xlim = min([xlims[i], len(lines)])
    x = range(0, xlim, step)
    y = lines[:xlim: step]
    # if max(lines) > 10:
    #     ylim([0, 10])
    plot(x, y, color[i], linewidth = 2)
    legend([legs[i]], fontsize='medium')
    # legend([u'hidden neurons:{}'.format(hidden_nodes[i])])
    # legend([u'learning rate:{}'.format(lr[i])])
    # legend([u'algorithm:{}'.format(algo[i])], fontsize='medium')
    xlabel('figure {}'.format(xlabels[i]))
    # xlabel('epoch', horizontalalignment = u'right')
    ylabel('loss')
show()
