# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'crux.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1129, 802)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/TheBiscuit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setIconSize(QtCore.QSize(24, 24))
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        MainWindow.setProperty(".\\Resources", "")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setIconSize(QtCore.QSize(16, 16))
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.tabPlots = QtWidgets.QWidget()
        self.tabPlots.setObjectName("tabPlots")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.tabPlots)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.params_window = QtWidgets.QFrame(self.tabPlots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.params_window.sizePolicy().hasHeightForWidth())
        self.params_window.setSizePolicy(sizePolicy)
        self.params_window.setMinimumSize(QtCore.QSize(500, 0))
        self.params_window.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.params_window.setFrameShadow(QtWidgets.QFrame.Raised)
        self.params_window.setObjectName("params_window")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.params_window)
        self.verticalLayout.setObjectName("verticalLayout")
        self.lbl_fn_name = QtWidgets.QLabel(self.params_window)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lbl_fn_name.sizePolicy().hasHeightForWidth())
        self.lbl_fn_name.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        self.lbl_fn_name.setFont(font)
        self.lbl_fn_name.setObjectName("lbl_fn_name")
        self.verticalLayout.addWidget(self.lbl_fn_name)
        self.txt_description = QtWidgets.QTextEdit(self.params_window)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.txt_description.sizePolicy().hasHeightForWidth())
        self.txt_description.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(12)
        self.txt_description.setFont(font)
        self.txt_description.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.txt_description.setReadOnly(True)
        self.txt_description.setObjectName("txt_description")
        self.verticalLayout.addWidget(self.txt_description)
        self.label_2 = QtWidgets.QLabel(self.params_window)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.tbl_param_values = QtWidgets.QTableWidget(self.params_window)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.tbl_param_values.sizePolicy().hasHeightForWidth())
        self.tbl_param_values.setSizePolicy(sizePolicy)
        self.tbl_param_values.setObjectName("tbl_param_values")
        self.tbl_param_values.setColumnCount(0)
        self.tbl_param_values.setRowCount(0)
        self.verticalLayout.addWidget(self.tbl_param_values)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.bbox_fit = QtWidgets.QDialogButtonBox(self.params_window)
        self.bbox_fit.setStandardButtons(QtWidgets.QDialogButtonBox.NoButton)
        self.bbox_fit.setObjectName("bbox_fit")
        self.horizontalLayout_3.addWidget(self.bbox_fit)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label = QtWidgets.QLabel(self.params_window)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_4.addWidget(self.label)
        self.chk_global = QtWidgets.QCheckBox(self.params_window)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.chk_global.sizePolicy().hasHeightForWidth())
        self.chk_global.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        self.chk_global.setFont(font)
        self.chk_global.setText("")
        self.chk_global.setObjectName("chk_global")
        self.horizontalLayout_4.addWidget(self.chk_global)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.tbl_series_links = QtWidgets.QTableWidget(self.params_window)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.tbl_series_links.sizePolicy().hasHeightForWidth())
        self.tbl_series_links.setSizePolicy(sizePolicy)
        self.tbl_series_links.setObjectName("tbl_series_links")
        self.tbl_series_links.setColumnCount(0)
        self.tbl_series_links.setRowCount(0)
        self.verticalLayout.addWidget(self.tbl_series_links)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(3, 3)
        self.verticalLayout.setStretch(6, 3)
        self.horizontalLayout.addWidget(self.params_window)
        self.mpl_window = QtWidgets.QWidget(self.tabPlots)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mpl_window.sizePolicy().hasHeightForWidth())
        self.mpl_window.setSizePolicy(sizePolicy)
        self.mpl_window.setMinimumSize(QtCore.QSize(500, 0))
        self.mpl_window.setObjectName("mpl_window")
        self.mpl_layout = QtWidgets.QVBoxLayout(self.mpl_window)
        self.mpl_layout.setContentsMargins(0, 0, 0, 0)
        self.mpl_layout.setObjectName("mpl_layout")
        self.horizontalLayout.addWidget(self.mpl_window)
        self.tabWidget.addTab(self.tabPlots, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.tab_2)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.txt_info = QtWidgets.QTextEdit(self.tab_2)
        self.txt_info.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.txt_info.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.txt_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.txt_info.setObjectName("txt_info")
        self.verticalLayout_2.addWidget(self.txt_info)
        self.label_3 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.tbl_fitted_params = QtWidgets.QTableWidget(self.tab_2)
        self.tbl_fitted_params.setObjectName("tbl_fitted_params")
        self.tbl_fitted_params.setColumnCount(0)
        self.tbl_fitted_params.setRowCount(0)
        self.verticalLayout_2.addWidget(self.tbl_fitted_params)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(3, 5)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_4.addWidget(self.label_4)
        self.tbl_fitted_data = QtWidgets.QTableWidget(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(3)
        sizePolicy.setHeightForWidth(self.tbl_fitted_data.sizePolicy().hasHeightForWidth())
        self.tbl_fitted_data.setSizePolicy(sizePolicy)
        self.tbl_fitted_data.setObjectName("tbl_fitted_data")
        self.tbl_fitted_data.setColumnCount(0)
        self.tbl_fitted_data.setRowCount(0)
        self.verticalLayout_4.addWidget(self.tbl_fitted_data)
        self.label_6 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setFamily("Calibri")
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_4.addWidget(self.label_6)
        self.tbl_smooth_line = QtWidgets.QTableWidget(self.tab_2)
        self.tbl_smooth_line.setObjectName("tbl_smooth_line")
        self.tbl_smooth_line.setColumnCount(0)
        self.tbl_smooth_line.setRowCount(0)
        self.verticalLayout_4.addWidget(self.tbl_smooth_line)
        self.verticalLayout_4.setStretch(3, 2)
        self.horizontalLayout_5.addLayout(self.verticalLayout_4)
        self.tabWidget.addTab(self.tab_2, "")
        self.horizontalLayout_2.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1129, 21))
        self.menuBar.setObjectName("menuBar")
        self.menu_File = QtWidgets.QMenu(self.menuBar)
        self.menu_File.setObjectName("menu_File")
        self.menu_Analysis = QtWidgets.QMenu(self.menuBar)
        self.menu_Analysis.setObjectName("menu_Analysis")
        self.menuSettings = QtWidgets.QMenu(self.menuBar)
        self.menuSettings.setObjectName("menuSettings")
        MainWindow.setMenuBar(self.menuBar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.action_save = QtWidgets.QAction(MainWindow)
        self.action_save.setEnabled(True)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/BiscuitSave.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_save.setIcon(icon1)
        self.action_save.setObjectName("action_save")
        self.action_quit = QtWidgets.QAction(MainWindow)
        self.action_quit.setMenuRole(QtWidgets.QAction.QuitRole)
        self.action_quit.setObjectName("action_quit")
        self.action_select_function = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/BiscuitFunction.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_select_function.setIcon(icon2)
        self.action_select_function.setObjectName("action_select_function")
        self.action_create = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/BiscuitNew.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_create.setIcon(icon3)
        self.action_create.setObjectName("action_create")
        self.action_open = QtWidgets.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/BiscuitOpen.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_open.setIcon(icon4)
        self.action_open.setObjectName("action_open")
        self.action_analyze = QtWidgets.QAction(MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/BiscuitFit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_analyze.setIcon(icon5)
        self.action_analyze.setObjectName("action_analyze")
        self.action_close = QtWidgets.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/BiscuitClose.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_close.setIcon(icon6)
        self.action_close.setObjectName("action_close")
        self.action_estimate = QtWidgets.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/BiscuitEstimate.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_estimate.setIcon(icon7)
        self.action_estimate.setObjectName("action_estimate")
        self.action_apply = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/BiscuitCalculate.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_apply.setIcon(icon8)
        self.action_apply.setObjectName("action_apply")
        self.menu_File.addAction(self.action_open)
        self.menu_File.addAction(self.action_create)
        self.menu_File.addAction(self.action_close)
        self.menu_File.addSeparator()
        self.menu_File.addAction(self.action_save)
        self.menu_File.addSeparator()
        self.menu_File.addAction(self.action_quit)
        self.menu_Analysis.addAction(self.action_select_function)
        self.menu_Analysis.addSeparator()
        self.menu_Analysis.addAction(self.action_estimate)
        self.menu_Analysis.addAction(self.action_apply)
        self.menu_Analysis.addAction(self.action_analyze)
        self.menuBar.addAction(self.menu_File.menuAction())
        self.menuBar.addAction(self.menu_Analysis.menuAction())
        self.menuBar.addAction(self.menuSettings.menuAction())
        self.toolBar.addAction(self.action_open)
        self.toolBar.addAction(self.action_close)
        self.toolBar.addAction(self.action_create)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_select_function)
        self.toolBar.addAction(self.action_apply)
        self.toolBar.addAction(self.action_estimate)
        self.toolBar.addAction(self.action_analyze)
        self.toolBar.addSeparator()
        self.toolBar.addAction(self.action_save)
        self.toolBar.addSeparator()

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "The Crux of the Biscuit"))
        MainWindow.setToolTip(_translate("MainWindow", "The Crux of the Biscuit"))
        self.lbl_fn_name.setText(_translate("MainWindow", "Selected function: None"))
        self.label_2.setText(_translate("MainWindow", "Parameter values"))
        self.label.setText(_translate("MainWindow", "Share parameters between series"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabPlots), _translate("MainWindow", "The Crux"))
        self.label_5.setText(_translate("MainWindow", "Information"))
        self.label_3.setText(_translate("MainWindow", "Best fit parameters"))
        self.label_4.setText(_translate("MainWindow", "Fitted data"))
        self.label_6.setText(_translate("MainWindow", "Smooth line"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Fit details"))
        self.menu_File.setTitle(_translate("MainWindow", "File"))
        self.menu_Analysis.setTitle(_translate("MainWindow", "&Analysis"))
        self.menuSettings.setTitle(_translate("MainWindow", "Settings"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.action_save.setText(_translate("MainWindow", "Save results"))
        self.action_save.setToolTip(_translate("MainWindow", "Save the parameter values and best fit curves to a comma-separated values (csv) file"))
        self.action_quit.setText(_translate("MainWindow", "Quit"))
        self.action_select_function.setText(_translate("MainWindow", "Select function"))
        self.action_select_function.setToolTip(_translate("MainWindow", "Select an objective function from the list in the dialog that will open"))
        self.action_create.setText(_translate("MainWindow", "Create data set"))
        self.action_create.setToolTip(_translate("MainWindow", "Create a new data set (give parameter values and independent ranges in the dialog that will open)"))
        self.action_open.setText(_translate("MainWindow", "Open data set"))
        self.action_open.setToolTip(_translate("MainWindow", "Open an existing data set (Octet format, comma-separated values)"))
        self.action_analyze.setText(_translate("MainWindow", "Perform fit"))
        self.action_analyze.setToolTip(_translate("MainWindow", "Determine which parameters values model the data best"))
        self.action_close.setText(_translate("MainWindow", "Close data set"))
        self.action_close.setToolTip(_translate("MainWindow", "Close the current data set"))
        self.action_estimate.setText(_translate("MainWindow", "Estimate"))
        self.action_estimate.setToolTip(_translate("MainWindow", "Set a first estimate for the parameter values"))
        self.action_apply.setText(_translate("MainWindow", "Calculate"))
        self.action_apply.setToolTip(_translate("MainWindow", "Apply the current parameter values to the objective function"))

import crux_rc
