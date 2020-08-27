# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 15:00:40 2020

@author: Majin
"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class Plotter():
    def __init__(self, phases=('train','val',), metrics=('loss', 'acc',), filename='figure.jpg'):
        # phases: tuple(string), use (element,) when contain only one element
        # metrics: tuple(string), use (element,) when contain only one element
        self.phases = phases
        self.metrics = metrics
        self.filename = filename
        self.plot_data = {}
        self.plot_data['epoch'] = []
        for phase in phases:
            self.plot_data[phase] = {}
            for metric in metrics:
                self.plot_data[phase][metric] = []
    
    def update(self, epoch, phase, metric, value):
        if epoch not in self.plot_data['epoch']:
            self.plot_data['epoch'].append(epoch)
            for pi in self.phases:
                for mi in self.metrics:
                    self.plot_data[pi][mi].append(0)
        idx = self.plot_data['epoch'].index(epoch)
        self.plot_data[phase][metric][idx] = value
    
    def draw_curve(self):
        fig = plt.figure()
        num_phases = len(self.phases)
        num_metrics = len(self.metrics)
        curve_types = ('bo-','rs-','y*-', 'g^-', 'mv-', 'c+-')
        assert num_phases<len(curve_types)
        for i in range(num_metrics):
            loc = 100 + 10*num_metrics + i+1
            metric = self.metrics[i]
            ax = fig.add_subplot(loc, title = metric)
            for j in range(num_phases):
                ax.plot(self.plot_data['epoch'],
                          self.plot_data[self.phases[j]][self.metrics[i]],
                          curve_types[j],
                          label = self.phases[j])
            ax.legend()
        fig.savefig(self.filename)
        plt.close()


        