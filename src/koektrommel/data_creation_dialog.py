'''
Created on 15 Jan 2018

@author: schilsm
'''
from PyQt5 import QtCore as qt
from PyQt5 import QtWidgets as widgets
from PyQt5 import QtGui as gui

import pandas as pd, numpy as np
from scipy.stats import norm
import copy as cp

from koektrommel.crux_table_model import CruxTableModel

class DataCreationDialog(widgets.QDialog):

    def __init__(self, parent, selected_fn):
        self.function = selected_fn
        self.all_series_names = []
        self.pn_model_params = pd.Panel(major_axis=self.function.parameters, minor_axis=['Value'])
        self.pn_axes_params = pd.Panel(major_axis=self.function.independents, minor_axis=['Start', 'End'])
        self.df_series_spec = pd.DataFrame(columns=['edit_npoint', 'edit_std', 'npoints', 'std'])
        self.template = None
                
        super(DataCreationDialog, self).__init__(parent)
        self.setWindowTitle("Create a data set")
        
        # Buttonbox
        main_layout = widgets.QVBoxLayout()
        self.button_box = widgets.QDialogButtonBox()
        self.btn_add_series = widgets.QPushButton("Add series")
        self.button_box.addButton(self.btn_add_series, widgets.QDialogButtonBox.ActionRole)
        self.button_box.addButton(widgets.QDialogButtonBox.Cancel)
        self.button_box.addButton(widgets.QDialogButtonBox.Ok)
        self.btn_add_series.clicked.connect(self.add_series)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        # Function description widgets
        txt_fn = widgets.QLabel("Function: " + self.function.name)
        txt_descr = widgets.QTextEdit(self.function.long_description)
        txt_descr.setReadOnly(True)
        txt_descr.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
        
        # Series tab widget
        self.pn_series_data = None
        
        self.tab_series = widgets.QTabWidget()
        self.add_series()
        
        # Main layout
        main_layout.addWidget(txt_fn)
        main_layout.addWidget(txt_descr)
        main_layout.addWidget(self.tab_series)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)
        
    def add_series(self):
        try:
            n = len(self.all_series_names)
            name = "Series " + str(n + 1)
            self.all_series_names.append(name)
    
            lbl_npoints = widgets.QLabel("Number of data points")
            txt_npoints = widgets.QLineEdit("21")
            txt_npoints.textChanged.connect(self.on_spec_changed)
            txt_npoints.setValidator(gui.QIntValidator())
            
            lbl_noise = widgets.QLabel("Standard deviation on data points")
            txt_std = widgets.QLineEdit("0.0")
            txt_std.textChanged.connect(self.on_spec_changed)
            txt_std.setValidator(gui.QDoubleValidator())
            self.df_series_spec.loc[name, 'edit_npoints'] = txt_npoints
            self.df_series_spec.loc[name, 'edit_std'] = txt_std
            self.df_series_spec.loc[name, 'npoints'] = txt_npoints.text()
            self.df_series_spec.loc[name, 'std'] = txt_std.text()
            
            df_axes = pd.DataFrame(index=self.pn_axes_params.major_axis, columns=self.pn_axes_params.minor_axis)
            self.pn_model_params.loc[name] = df_axes
            df_axes.Start = 0.0
            df_axes.End = 10.0
            df_axes.npoints = int(txt_npoints.text())
            df_axes.std = float(txt_std.text()) 
            
            df_params = pd.DataFrame(index=self.pn_model_params.major_axis, columns=self.pn_model_params.minor_axis)
            df_params.loc[:, :] = np.ones(df_params.values.shape)
            self.pn_model_params.loc[name] = df_params
            
            mdl_axes = CruxTableModel(df_axes.loc[:, 'Start':'End'])
            mdl_pars = CruxTableModel(self.pn_model_params.loc[name])

            lbl_axes = widgets.QLabel('&Axes parameters')
            tbl_axes = widgets.QTableView()
            lbl_axes.setBuddy(tbl_axes)
            tbl_axes.setModel(mdl_axes)
            tbl_axes.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
            
            lbl_pars = widgets.QLabel('&Model parameter values')
            tbl_pars = widgets.QTableView()
            lbl_pars.setBuddy(tbl_pars)
            tbl_pars.setModel(mdl_pars)
            tbl_pars.setSizeAdjustPolicy(widgets.QAbstractScrollArea.AdjustToContents)
    
            # Put everything in a layout inside a tab widget
            w = widgets.QWidget()
            self.tab_series.addTab(w, name)
            glo1 = widgets.QGridLayout()
            glo1.addWidget(lbl_npoints, 0, 0, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(lbl_noise, 0, 1, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(txt_npoints, 1, 0, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(txt_std, 1, 1, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(lbl_axes, 2, 0, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(lbl_pars, 2, 1, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(tbl_axes, 3, 0, alignment=qt.Qt.AlignHCenter)
            glo1.addWidget(tbl_pars, 3, 1, alignment=qt.Qt.AlignHCenter)
            vlo = widgets.QVBoxLayout()
            vlo.addLayout(glo1)
            w.setLayout(vlo)
            self.tab_series.setCurrentWidget(w)
            
        except Exception as e:
            print(e)
            

    def on_spec_changed(self):
        nps, std, sel = False, False, ""
        print(self.pn_model_params)
        try:
            for sname, row in self.df_series_spec.iterrows():
                for cat, item in row.iteritems():
                    if item == self.sender():
                            if cat == 'edit_npoints':
                                nps = True
                                sel = sname              
                            elif cat == 'edit_std':
                                std = True
                                sel = sname
            if nps: # should not change matrix while iterating over it, hence this construction
                self.df_series_spec.loc[sel, 'npoints'] = int(self.sender().text())
            elif std:
                self.df_series_spec.loc[sel, 'std'] = float(self.sender().text())
            else:
                pass
            
        except Exception as e:
            print(e)
                    
    
        
            
#     def get_parameters(self):
#         """
#         Returns a pandas DataFrame with parameter values; index: series names, columns: parameter names
#         """
#         df_all_parameters = pd.DataFrame()
#         for name in self.all_series_names:
#             df_p = self.pn_model_params[name].df_data # parameters for this series
#             df_all_parameters = pd.concat([df_all_parameters, df_p], axis=1) # add to all parameters            
#         return cp.deepcopy(df_all_parameters)
#     
#     def get_series_names(self):
#         return np.array(self.all_series_names)
#     
#     def get_axes(self):
#         return np.array(self.function.independents)
#     
#     def get_series_dict(self):
#         df_all_series = pd.DataFrame()
#         series_dict = {}
#         for name in self.all_series_names:
#             df_p = self.pn_model_params[name].df_data # parameters for this series
#             df_si = self.pn_axes_params[name][0].df_data # axes start, stop, and std on data
#             n = int(self.pn_axes_params[name][1].text()) # number of points in series
#             std = float(self.pn_axes_params[name][2].text())
#             cols = df_si.index # independent axes names
#             df_s = pd.DataFrame([], index=range(n), columns=cols) # dataframe for axes values
#             for col in cols:
#                 df_s[col] = np.linspace(df_si.iloc[:,0][col], df_si.iloc[:,1][col], n)
#             x = cp.deepcopy(df_s).as_matrix().transpose() # copy axes values and transpose for use in self.function
#             params = cp.deepcopy(df_p).as_matrix()
#             vals = pd.DataFrame([], index=range(n), columns=[name]) # dataframe for dependent values
#             y = self.function.func(x, params)
#             if std > 0:
#                 y = norm.rvs(loc=y, scale=std)
#             vals[name] = y
#             df_s = pd.concat([df_s, vals], axis=1)
#             df_all_series = pd.concat([df_all_series, df_s], axis=1)
#             series_dict[name] = df_s
#         return cp.deepcopy(series_dict) # f_all_series
                    
    def accept(self):
#        self.template = self.create_template()
        widgets.QDialog.accept(self)
        
    def reject(self):
        widgets.QDialog.reject(self)
        
        

        
        