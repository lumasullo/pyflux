# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 14:18:19 2018

@author: Luciano A. Masullo
"""

from pyqtgraph.Qt import QtGui
import pyqtgraph as pg
from PyQt5 import QtWidgets

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
        self.linePlot.setMenuEnabled(enableMenu=False)
        
        grid.addWidget(graphicsLayout, 0, 0)
        
        self.gauss = False

    def get_scanConnection(self, main):
        self.main = main
        
    def contextMenuEvent(self, event):
        
        cmenu = QtWidgets.QMenu(self)
        
        GaussAction = cmenu.addAction('Fit 1D Gaussian')
        DoughnutAction = cmenu.addAction('Fit 1D Doughnut')
        
        action = cmenu.exec_(self.mapToGlobal(event.pos()))
        
        if action == GaussAction:
            if self.gauss:
                self.gauss = False
            else:
                self.gauss = True
                
        if action == DoughnutAction:
            if self.doughnut:
                self.doughnut = False
            else:
                self.doughnut = True
                
        self.main.update_line_profile()