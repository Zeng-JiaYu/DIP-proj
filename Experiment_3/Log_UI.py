# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Log_UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(432, 148)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 90, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 30, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setTextFormat(QtCore.Qt.PlainText)
        self.label.setObjectName("label")
        self.lineEdit_a = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_a.setGeometry(QtCore.QRect(50, 30, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.lineEdit_a.setFont(font)
        self.lineEdit_a.setObjectName("lineEdit_a")
        self.lineEdit_b = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_b.setGeometry(QtCore.QRect(190, 30, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.lineEdit_b.setFont(font)
        self.lineEdit_b.setObjectName("lineEdit_b")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(150, 30, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setTextFormat(QtCore.Qt.PlainText)
        self.label_2.setObjectName("label_2")
        self.lineEdit_c = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_c.setGeometry(QtCore.QRect(320, 30, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.lineEdit_c.setFont(font)
        self.lineEdit_c.setObjectName("lineEdit_c")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(290, 30, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setTextFormat(QtCore.Qt.PlainText)
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "指数变换参数设置"))
        self.label.setText(_translate("Dialog", "a:"))
        self.label_2.setText(_translate("Dialog", "b:"))
        self.label_3.setText(_translate("Dialog", "c:"))
