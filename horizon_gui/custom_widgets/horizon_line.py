from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QTabWidget, QHBoxLayout, QVBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QModelIndex
from PyQt5.QtGui import QStandardItemModel

from horizon_gui.definitions import CSS_DIR
from horizon_gui.custom_widgets.multi_slider import QMultiSlider

#
# class TabButtonWidget(QWidget):
#     def __init__(self, parent):
#         super().__init__(parent)
#         # Create button's
#         self.button_add = QPushButton("+")
#         self.button_remove = QPushButton("-")
#
#         # Set button size
#         self.button_add.setFixedSize(16, 16)
#         self.button_remove.setFixedSize(16, 16)
#
#         # Create layout
#         self.layout = QVBoxLayout()
#         self.layout.setSpacing(0)
#         self.layout.setContentsMargins(0, 0, 0, 0)
#
#         # Add button's to layout
#         self.layout.addWidget(self.button_add)
#         self.layout.addWidget(self.button_remove)
#
#         # Use layout in widget
#         self.setLayout(self.layout)

class FunctionTab(QWidget):
    nodesChanged = pyqtSignal(str, list)

    def __init__(self, name, n_nodes, options=None, parent=None):
        super().__init__(parent)

        self.name = name
        self.n_nodes = n_nodes

        self.hlayout = QHBoxLayout(self)
        self.slider = QMultiSlider(slider_range=[0, self.n_nodes - 1, 1], values=[0, self.n_nodes-1], options=options)
        self.hlayout.addWidget(self.slider)

        self.slider.slicesChanged.connect(self.on_nodes_changed)

    def on_nodes_changed(self, range_list):
        self.nodesChanged.emit(self.name, range_list)
        # adding + - push button to tab
        # self.ct_tab.tabBar().setTabButton(0, self.ct_tab.tabBar().RightSide, TabButtonWidget())

    def updateRange(self, n_nodes):
        self.n_nodes = n_nodes
        self.slider.updateRange([0, self.n_nodes-1])

class HorizonLine(QWidget):
    add_fun_horizon = pyqtSignal(dict)
    remove_fun_horizon = pyqtSignal(dict)
    repeated_fun = pyqtSignal(str)
    funNodesChanged = pyqtSignal(str, list)

    def __init__(self, horizon, fun_type, nodes=0, logger=None, parent=None):
        QWidget.__init__(self, parent)
        self.setAcceptDrops(True)

        self.horizon_receiver = horizon

        if logger:
            self.logger = logger
        # can be Constraint or Cost Function

        self.fun_type = fun_type

        self.options = dict()
        if self.fun_type == 'constraint':
            self.options['background_color'] = Qt.darkGreen
            self.options['slice_color'] = Qt.green
            self.options['minmax_color'] = Qt.darkRed
            self.options['ticks_color'] = Qt.darkCyan
        elif self.fun_type == 'costfunction':
            self.options['background_color'] = Qt.darkBlue
            self.options['slice_color'] = Qt.blue
            self.options['minmax_color'] = Qt.darkRed
            self.options['ticks_color'] = Qt.darkCyan



        self.setWindowState(Qt.WindowMaximized)
        self.showMaximized()
        # define main layout
        self.main_layout = QHBoxLayout(self)

        # define TabWidget
        self.fun_tab = QTabWidget(self)

        with open(CSS_DIR + '/tab.css', 'r') as f:
            self.fun_tab.setStyleSheet(f.read())

        self.fun_tab.setTabPosition(1)
        self.fun_tab.setTabsClosable(True)

        # add TabWidget to main layout
        self.main_layout.addWidget(self.fun_tab)

        # add a layout to TabWidget
        self.fun_tab_layout = QGridLayout(self.fun_tab)

        self.n_nodes = nodes
        self._updateBoxNodes()

        # self.rubberband = QRubberBand(QRubberBand.Rectangle, self)
        # self.setMouseTracking(True)

        self.fun_tab.tabCloseRequested.connect(self.removeActiveFunctionRequest)

    def setBoxNodes(self, nodes):
        self.n_nodes = nodes
        self._updateBoxNodes()

    def _updateBoxNodes(self):
        # Add nodes to the layout

        for i in reversed(range(self.fun_tab_layout.count())):
            self.fun_tab_layout.itemAt(i).widget().deleteLater()

        if self.n_nodes == 0:
            self.setDisabled(True)
        else:
            self.setDisabled(False)

        for i in range(self.n_nodes):
            n_i = QPushButton("{}".format(i), self)
            self.fun_tab_layout.addWidget(n_i, 0, i, 1, 1, alignment=Qt.AlignTop)


        for i in range(self.fun_tab.count()):
            self.fun_tab.widget(i).findChild(FunctionTab).updateRange(self.n_nodes)
            node_box_width = self.fun_tab_layout.itemAt(0).widget().width()
            self.intab_layout.setContentsMargins(node_box_width / 2 - 2, 20, node_box_width / 2 - 5, 0)

    def removeActiveFunctionRequest(self, index):

        flag, signal = self.horizon_receiver.removeActiveFunction(self.fun_tab.tabText(index))

        if flag:
            self.fun_tab.removeTab(index)

        self.logger.info(signal)

    def dragEnterEvent(self, event):
        # source_Widget = event.source()
        # items = source_Widget.selectedItems()
        # for i in items:
        #     source_Widget.takeItem(source_Widget.indexFromItem(i).row())
        #     # self.addItem(i)
        #     print(i.text())
        #     print('drop event')
        # standard format of mimeData from QListWidget
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        source_item = QStandardItemModel()
        source_item.dropMimeData(event.mimeData(), Qt.CopyAction, 0, 0, QModelIndex())
        fun_name = source_item.item(0, 0).text()
        # fun = source_item.item(0, 0).data(Qt.UserRole)

        self.addFunctionToHorizon(fun_name)

    @pyqtSlot()
    def on_repeated_fun(self, str):
        self.repeated_fun.emit(str)

    @pyqtSlot()
    def on_fun_nodes_changed(self, name, ranges):
        self.funNodesChanged.emit(name, ranges)

    def emitFunctionNodes(self, name, ranges):
        self.on_fun_nodes_changed(name, ranges)

    def addFunctionToGUI(self, fun_name):

        node_box_width = self.fun_tab_layout.itemAt(0).widget().width()
        self.ft = FunctionTab(fun_name, self.n_nodes, options=self.options)
        self.ft.nodesChanged.connect(self.emitFunctionNodes)
        verticalSpacer = QSpacerItem(20, 200, QSizePolicy.Minimum, QSizePolicy.Fixed)

        # todo hardcoded bottom margin
        self.intab_layout = QVBoxLayout()

        # don't fucking ask me why, these are magic numbers (-2, -5) in margins for alignment
        self.intab_layout.setContentsMargins(node_box_width / 2 - 2, 20, node_box_width / 2 - 5, 0)
        self.intab_layout.addWidget(self.ft, stretch=3)
        # self.intab_layout.addWidget(self.ct_max_lim, stretch=1)
        # self.intab_layout.addWidget(self.ct_min_lim, stretch=1)
        self.intab_layout.addSpacerItem(verticalSpacer)

        self.tab = QWidget()
        self.tab.setLayout(self.intab_layout)

        self.fun_tab.addTab(self.tab, str(fun_name))

    def addFunctionToHorizon(self, name):
        flag, signal = self.horizon_receiver.activateFunction(name, self.fun_type)

        if flag:
            self.addFunctionToGUI(name)
            self.logger.info(signal)
        else:
            self.logger.warning(signal)

    def getFunctionNodes(self, name):

        for i in range(self.fun_tab.count()):
            if self.fun_tab.tabText(i) == name:
                print('Nodes of function {}: {}'.format(name, self.fun_tab.widget(i).getNodes()))
                return self.fun_tab.widget(i).getNodes()
            else:
                pass



    # def mousePressEvent(self, event):
    #     self.origin = event.pos()
    #     self.rubberband.setGeometry(QRect(self.origin, QSize()))
    #     self.rubberband.show()
    #     QWidget.mousePressEvent(self, event)
    #
    # def mouseMoveEvent(self, event):
    #     if self.rubberband.isVisible():
    #         self.rubberband.setGeometry(QRect(self.origin, event.pos()).normalized())
    #     QWidget.mouseMoveEvent(self, event)
    #
    # def mouseReleaseEvent(self, event):
    #         if self.rubberband.isVisible():
    #             self.rubberband.hide()
    #             selected = []
    #             rect = self.rubberband.geometry()
    #             if self.ct_tab.widget(self.ct_tab.currentIndex()) is not None:
    #                 for child in self.ct_tab.widget(self.ct_tab.currentIndex()).findChildren(QCheckBox):
    #                     # get absolute geometry and then map it to self (HorizonLine)
    #                     # the problem was that .geometry() returns the position w.r.t. the parent:
    #                     # I was comparing the geometry of the rubberband taken in the HorizonLine's coordinates with
    #                     # the geometry of the QCheckBox taken in the QGridLayout's coordinates.
    #
    #                     gp = child.mapToGlobal(QPoint(0, 0))
    #                     b_a = self.mapFromGlobal(gp)
    #
    #                     # todo check if possible, right now im subtracting a small value to make the geometry of the checkbox smaller
    #                     if rect.intersects(QRect(b_a.x(), b_a.y(), child.geometry().width()-20, child.geometry().height())):
    #                         selected.append(child)
    #
    #             for elem in selected:
    #                 if event.button() == Qt.RightButton:
    #                     elem.setChecked(False)
    #                 elif event.button() == Qt.LeftButton:
    #                     elem.setChecked(True)
    #
    #         QWidget.mouseReleaseEvent(self, event)