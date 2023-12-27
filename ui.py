# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1321, 771)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.ImgFolderBtn = QtWidgets.QPushButton(self.centralwidget)
        self.ImgFolderBtn.setGeometry(QtCore.QRect(20, 390, 261, 49))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(12)
        self.ImgFolderBtn.setFont(font)
        self.ImgFolderBtn.setStyleSheet("border-radius:20px;\n"
"border:1px solid #000;\n"
" background:#ffb6c1;")
        self.ImgFolderBtn.setObjectName("ImgFolderBtn")
        self.FolderPathLabel = QtWidgets.QLabel(self.centralwidget)
        self.FolderPathLabel.setEnabled(True)
        self.FolderPathLabel.setGeometry(QtCore.QRect(20, 440, 261, 49))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(10)
        self.FolderPathLabel.setFont(font)
        self.FolderPathLabel.setAutoFillBackground(False)
        self.FolderPathLabel.setText("")
        self.FolderPathLabel.setTextFormat(QtCore.Qt.AutoText)
        self.FolderPathLabel.setScaledContents(False)
        self.FolderPathLabel.setWordWrap(True)
        self.FolderPathLabel.setObjectName("FolderPathLabel")
        self.DetectBtn = QtWidgets.QPushButton(self.centralwidget)
        self.DetectBtn.setGeometry(QtCore.QRect(20, 600, 261, 49))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(12)
        self.DetectBtn.setFont(font)
        self.DetectBtn.setStyleSheet("border-radius:20px;\n"
"border:1px solid #000;\n"
" background:#ffb6c1;")
        self.DetectBtn.setObjectName("DetectBtn")
        self.SegmentBtn = QtWidgets.QPushButton(self.centralwidget)
        self.SegmentBtn.setGeometry(QtCore.QRect(20, 660, 261, 49))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(12)
        self.SegmentBtn.setFont(font)
        self.SegmentBtn.setStyleSheet("border-radius:20px;\n"
"border:1px solid #000;\n"
" background:#ffb6c1;")
        self.SegmentBtn.setObjectName("SegmentBtn")
        self.OriginalImgBox = QtWidgets.QLabel(self.centralwidget)
        self.OriginalImgBox.setGeometry(QtCore.QRect(20, 70, 280, 280))
        self.OriginalImgBox.setStyleSheet("border:1px solid #000;")
        self.OriginalImgBox.setText("")
        self.OriginalImgBox.setObjectName("OriginalImgBox")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(660, 380, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(20)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(650, 20, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(20)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.SegmentationImgBox = QtWidgets.QLabel(self.centralwidget)
        self.SegmentationImgBox.setGeometry(QtCore.QRect(660, 430, 280, 280))
        self.SegmentationImgBox.setStyleSheet("border:1px solid #000;")
        self.SegmentationImgBox.setText("")
        self.SegmentationImgBox.setObjectName("SegmentationImgBox")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 30, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.DetectionImgBox = QtWidgets.QLabel(self.centralwidget)
        self.DetectionImgBox.setGeometry(QtCore.QRect(660, 70, 280, 280))
        self.DetectionImgBox.setStyleSheet("border:1px solid #000;")
        self.DetectionImgBox.setText("")
        self.DetectionImgBox.setObjectName("DetectionImgBox")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(980, 70, 301, 281))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.layoutWidget.setFont(font)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.CurrentLabel = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.CurrentLabel.setFont(font)
        self.CurrentLabel.setObjectName("CurrentLabel")
        self.verticalLayout.addWidget(self.CurrentLabel)
        self.FPSLabel = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.FPSLabel.setFont(font)
        self.FPSLabel.setObjectName("FPSLabel")
        self.verticalLayout.addWidget(self.FPSLabel)
        self.GTLabel = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.GTLabel.setFont(font)
        self.GTLabel.setObjectName("GTLabel")
        self.verticalLayout.addWidget(self.GTLabel)
        self.PredictLabel = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.PredictLabel.setFont(font)
        self.PredictLabel.setObjectName("PredictLabel")
        self.verticalLayout.addWidget(self.PredictLabel)
        self.IOULabel = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.IOULabel.setFont(font)
        self.IOULabel.setObjectName("IOULabel")
        self.verticalLayout.addWidget(self.IOULabel)
        self.DiceCoefficientLabel = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.DiceCoefficientLabel.setFont(font)
        self.DiceCoefficientLabel.setObjectName("DiceCoefficientLabel")
        self.verticalLayout.addWidget(self.DiceCoefficientLabel)
        self.ImgBtn = QtWidgets.QPushButton(self.centralwidget)
        self.ImgBtn.setGeometry(QtCore.QRect(20, 490, 261, 49))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(12)
        self.ImgBtn.setFont(font)
        self.ImgBtn.setStyleSheet("border-radius:20px;\n"
"border:1px solid #000;\n"
" background:#ffb6c1;")
        self.ImgBtn.setObjectName("ImgBtn")
        self.DetectionGTImgBox = QtWidgets.QLabel(self.centralwidget)
        self.DetectionGTImgBox.setGeometry(QtCore.QRect(360, 70, 280, 280))
        self.DetectionGTImgBox.setStyleSheet("border:1px solid #000;")
        self.DetectionGTImgBox.setText("")
        self.DetectionGTImgBox.setIndent(-1)
        self.DetectionGTImgBox.setObjectName("DetectionGTImgBox")
        self.SegmentationGTImgBox = QtWidgets.QLabel(self.centralwidget)
        self.SegmentationGTImgBox.setGeometry(QtCore.QRect(360, 430, 280, 280))
        self.SegmentationGTImgBox.setStyleSheet("border:1px solid #000;")
        self.SegmentationGTImgBox.setText("")
        self.SegmentationGTImgBox.setObjectName("SegmentationGTImgBox")
        self.ImgLabel = QtWidgets.QLabel(self.centralwidget)
        self.ImgLabel.setEnabled(True)
        self.ImgLabel.setGeometry(QtCore.QRect(30, 540, 261, 49))
        font = QtGui.QFont()
        font.setFamily("Arial Rounded MT Bold")
        font.setPointSize(10)
        self.ImgLabel.setFont(font)
        self.ImgLabel.setAutoFillBackground(False)
        self.ImgLabel.setText("")
        self.ImgLabel.setTextFormat(QtCore.Qt.AutoText)
        self.ImgLabel.setScaledContents(False)
        self.ImgLabel.setWordWrap(True)
        self.ImgLabel.setObjectName("ImgLabel")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(980, 430, 301, 281))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.widget.setFont(font)
        self.widget.setObjectName("widget")
        self.formLayout_2 = QtWidgets.QFormLayout(self.widget)
        self.formLayout_2.setLabelAlignment(QtCore.Qt.AlignCenter)
        self.formLayout_2.setFormAlignment(QtCore.Qt.AlignCenter)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setSpacing(12)
        self.formLayout_2.setObjectName("formLayout_2")
        self.EvaluationLabel = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.EvaluationLabel.setFont(font)
        self.EvaluationLabel.setObjectName("EvaluationLabel")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.EvaluationLabel)
        self.MeanLabel = QtWidgets.QLabel(self.widget)
        self.MeanLabel.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.MeanLabel.setFont(font)
        self.MeanLabel.setObjectName("MeanLabel")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.MeanLabel)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignCenter)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignCenter)
        self.formLayout.setHorizontalSpacing(12)
        self.formLayout.setVerticalSpacing(11)
        self.formLayout.setObjectName("formLayout")
        self.UncoverLabel = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.UncoverLabel.setFont(font)
        self.UncoverLabel.setObjectName("UncoverLabel")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.UncoverLabel)
        self.UnevenLabel = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.UnevenLabel.setFont(font)
        self.UnevenLabel.setObjectName("UnevenLabel")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.UnevenLabel)
        self.ScratchLabel = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(18)
        self.ScratchLabel.setFont(font)
        self.ScratchLabel.setObjectName("ScratchLabel")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.ScratchLabel)
        self.formLayout_2.setLayout(2, QtWidgets.QFormLayout.FieldRole, self.formLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ImgFolderBtn.setText(_translate("MainWindow", "Select Image Folder"))
        self.DetectBtn.setText(_translate("MainWindow", "Detect defects"))
        self.SegmentBtn.setText(_translate("MainWindow", "Segment"))
        self.label_3.setText(_translate("MainWindow", "Segmentation result"))
        self.label_2.setText(_translate("MainWindow", "Detection result"))
        self.label.setText(_translate("MainWindow", "Original image"))
        self.CurrentLabel.setText(_translate("MainWindow", "Current Image :  "))
        self.FPSLabel.setText(_translate("MainWindow", "FPS:"))
        self.GTLabel.setText(_translate("MainWindow", "Type(GT)"))
        self.PredictLabel.setText(_translate("MainWindow", "Predict: "))
        self.IOULabel.setText(_translate("MainWindow", "IoU :"))
        self.DiceCoefficientLabel.setText(_translate("MainWindow", "Dice Coefficient : "))
        self.ImgBtn.setText(_translate("MainWindow", "Select Image"))
        self.EvaluationLabel.setText(_translate("MainWindow", "Evaluation Metric"))
        self.MeanLabel.setText(_translate("MainWindow", "Folder (Mean)"))
        self.UncoverLabel.setText(_translate("MainWindow", "AP50(uncover):"))
        self.UnevenLabel.setText(_translate("MainWindow", "AP50(uneven):"))
        self.ScratchLabel.setText(_translate("MainWindow", "AP50(scratch)"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
