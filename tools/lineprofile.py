# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018

@author: Luciano A. Masullo
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg

class linePlotWidget(QtGui.QWidget):
        
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        graphicsLayout = pg.GraphicsLayoutWidget()
        grid = QtGui.QGridLayout()
        
        self.setLayout(grid)
        self.linePlot = graphicsLayout.addPlot(row=0, col=0, 
                                               title="Intensity line profile")
        self.linePlot.setLabels(bottom=('nm'),
                                left=('counts'))
        
        grid.addWidget(graphicsLayout, 0, 0)