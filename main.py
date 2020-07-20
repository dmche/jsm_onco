# -*- coding: utf-8 -*-
# pyuic5 /Users/dmitrychebanov/untitled002/design08plusF.ui -o /Users/dmitrychebanov/Googledrive/python/PycharmProjects/untitled4/myvenv5/design.py

import sys
from PyQt5 import QtWidgets

import design

from jA006m import *

'''
class MyPopup(QWidget):
    def __init__(self):
        QWidget.__init__(self)

    def paintEvent(self, e):
        dc = QPainter(self)
        dc.drawLine(0, 0, 100, 100)
        dc.drawLine(100, 0, 0, 100)
        
'''
#sign = 1
class ExampleApp(QtWidgets.QMainWindow, design.Ui_LatticeNavi):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        # options: pressing different buttons calls different functions:

        # CHAINS

        # Выполнить функцию calculate_hyps при нажатии кнопки
        self.CalcButLower.clicked.connect(self.calc_borders)  #both borders itself, depends on radio
        self.LowLeftOnly.textChanged.connect(self.left_only)  #on text change in Input Lower
        self.LowLeftOnly_2.textChanged.connect(self.left_only)  # on text change 'Without' in Input Lower
        # self.LowMovUp.clicked.connect(self.level_up)  # level up for all
        # self.UppMovDown.clicked.connect(self.level_down)  # level up for all


        # FULL HYPOTHESES
        self.CalcButAll.clicked.connect(self.calc_all)  # newly all amounts of hyps

        self.Load.clicked.connect(self.load_fromfile)  #button press - print on window

        self.Show.setEnabled(False) # by default it is nothing to show, so the button is disabled

        self.radioButtonPlus.toggled.connect(self.filter_preview)

        self.Show.clicked.connect(self.show_window)  #button press - print on window



        # filters of all

        self.FiltLeftOnly.textChanged.connect(self.left_only_all)  # on text change in Input Lower
        self.FiltLeftNot.textChanged.connect(self.left_only_all)  # on text change 'Without' in Input Lower
        self.LenHypsFilter.valueChanged.connect(self.left_only_all)  # on text change 'Without' in Input Lower



        # NAVIGATION block
        self.NavInputHypHint.textChanged.connect(self.nav_status)  # on text change in Navigation
        self.NavUp.clicked.connect(self.nav_up)  #on a level up
        self.NavDown.clicked.connect(self.nav_down)  #on a level down
        self.CalcNeighbors.clicked.connect(self.get_delta)  # Neighbors search
        #self.checkBox.stateChanged.connect(self.get_delta)  # checkbox for more
        #self.checkBox_2.stateChanged.connect(self.get_delta)  # checkbox for less
        self.LowLeftOnly_3.textChanged.connect(self.left_only_nav)  # on text change in Input Lower
        self.LowLeftOnly_4.textChanged.connect(self.left_only_nav)  # on text change 'Without' in Input Lower


        # Encryption block
        self.DecryptAll.clicked.connect(self.crypt_print_all)  #all the reasons and their numbers
        self.DecryptByNumber.textChanged.connect(self.crypt_reas_nums_to_names)  # on text change in Navigation
        # fill dropdown menu for reasons-numbers:
        self.comboBox.addItem("bage")
        self.comboBox.addItem("mage")
        # cycle

        #self.DecryptByNumber.textChanged.connect(self.left_only_nav) #
        self.comboBox.currentTextChanged.connect(self.crypt_select_byone_reas_names_to_nums)  # crypt by names with dropdown window




    def calc_borders(self):
        '''
        :return: calculates not only lower, but also both borders
        '''
        self.set_sign() # function call from  another place here
        # for any other functions could use  border, and not to calculate again:
        global border

        if self.lowradioButton.isChecked():
            # implement function, and sorting the list:
            if self.radioButtonSortOrb.isChecked():
                low_bor = lower1(sign)
            else:
                low_bor = sorted(lower1(sign), key=len)
            border = low_bor
        else:
            if self.radioButtonSortOrb.isChecked():
                upp_bor = upper(sign)
            else:
                upp_bor = sorted(upper(sign), key=len)
            border = upp_bor

        # result output
        self.LowBord.clear()  # На случай, если в окне вывода уже есть элементы
        # Not allow wrapping of hypotheses output:
        self.LowBord.setLineWrapMode(False)
        # printing text into output form:
        for el in border:
            self.LowBord.append(str(el).replace('[', '').replace(']', ''))

        # stay at top and not scroll to the bottom:
        self.LowBord.verticalScrollBar().setValue(0)

        # show length of border:
        self.LowBordLen.clear()
        self.LowBordLen.setText(str(len(border)))

        # clear masks and filters:
        self.LowLeftOnly.clear()
        self.LowLeftOnly_2.clear()

    def set_sign(self):
        global sign
        if self.Radiobutton_plus.isChecked():
            sign = 1
            print('i am working - plus')
        else:
            sign = -1
            print('i am working - minus')

    def left_only(self):
        #take all the text in Output window:
        #all_hyps = self.LowBord.toPlainText() #they disappear after first change of input window
        #parse by end of string:
        #all_hyps_list = all_hyps.split('\n')

        all_hyps_list = border

        # text that appear in With input window, remove spaces:
        mask = self.LowLeftOnly.text().replace(' ', '')
        # text that appear in Without input window, remove spaces:
        maskWithout = self.LowLeftOnly_2.text().replace(' ', '')

        #check if anything inputted in the mask window:
        if (len(mask) == 0) and (len(maskWithout) == 0):
            # delete all the masks - print back whole the current border:
            for el in all_hyps_list:
                self.LowBord.append(str(el).replace('[', '').replace(']', ''))
                self.LowBordLen.setText(str(len(all_hyps_list)))

        else:
            new_after_mask = []

            def mask_elems(maska):
                # divide inputted by ',':
                mask_el_str = maska.split(',')
                # deleting empty symbols:
                mask_el_str = list(filter(None, mask_el_str))
                # make integer:
                return [int(x) for x in mask_el_str]

            if (len(mask) > 0) and (len(maskWithout) == 0):
                for hyp in all_hyps_list:
                    if all(x in hyp for x in mask_elems(mask)):
                        new_after_mask.append(hyp)

            elif (len(mask) == 0) and (len(maskWithout) > 0):
                for hyp in all_hyps_list:
                    if not any(x in hyp for x in mask_elems(maskWithout)):
                        new_after_mask.append(hyp)

            elif (len(mask) > 0) and (len(maskWithout) > 0):
                for hyp in all_hyps_list:
                    if (not any(x in hyp for x in mask_elems(maskWithout))) and (all(x in hyp for x in mask_elems(mask))):
                        new_after_mask.append(hyp)

            # and finally print result in output:
            self.LowBord.clear()
            for el in new_after_mask:
                self.LowBord.append(str(el).replace('[', '').replace(']', ''))
            # show length of masked list:
            self.LowBordLen.setText(str(len(new_after_mask)))
            # stay at top and not scroll to the bottom:
            self.LowBord.verticalScrollBar().setValue(0)


    def calc_all(self):
        '''
        :return: calculates not only lower, but also both borders
        '''

        # number of plus hypotheses:
        global all_hyps
        # here should depend on the sign:
        all_hyps = comb(1)

        # show length of plus sign hyps:
        self.PlusHypsTotal.clear()
        self.PlusHypsTotal.setText(str(len(all_hyps)))

        if self.radioButtonPlus.isChecked():
            # implement function, and sorting the list:
            self.LeftNumberFilters.clear()
            self.LeftNumberFilters.setText(str(len(all_hyps)))


    def load_fromfile(self):
        # Loading instead of newly calculating
        global all_hyps
        all_hyps = []
        get_fromfile(1, all_hyps, '_reas')

        # show length of plus sign hyps:
        self.PlusHypsTotal.clear()
        self.PlusHypsTotal.setText(str(len(all_hyps)))




    def filter_preview(self):

        # when checkbox about sign is selected:
        self.Show.setEnabled(True)
        self.LeftNumberFilters.clear()
        self.LeftNumberFilters.setText(str(len(all_hyps)))


    def show_window(self):
        # shows hyps when button Show being pressed
        # result output
        self.DisplayFilter.clear()  # На случай, если в окне вывода уже есть элементы
        # Not allow wrapping of hypotheses output:
        self.DisplayFilter.setLineWrapMode(False)
        # display the list with a name all
        for el in all_hyps:
            self.DisplayFilter.append(str(el).replace('[', '').replace(']', '').replace("'", ""))
        # stay at top and not scroll to the bottom:
        self.DisplayFilter.verticalScrollBar().setValue(0)


    def left_only_all(self):
        # Filter for all

        all_hyps_list = all_hyps
        #get_fromfile(1, all_hyps_list, '_reas')


        # text that appear in With input window, remove spaces:
        mask = self.FiltLeftOnly.text().replace(' ', '')
        # text that appear in Without input window, remove spaces:
        maskWithout = self.FiltLeftNot.text().replace(' ', '')
        # text in length window
        length = self.LenHypsFilter.value()

        #check if anything inputted in the mask window:
        if (len(mask) == 0) and (len(maskWithout) == 0) and (length == 0):
            # delete all the masks - print back whole the current border:
            for el in all_hyps_list:
                self.DisplayFilter.append(str(el).replace('[', '').replace(']', '').replace("'", ""))
                self.LeftNumberFilters.setText(str(len(all_hyps_list)))

        else:
            # if filters not empty - make new list:
            new_after_mask = []

            def mask_elems(maska):
                # divide inputted by ',':
                mask_el_str = maska.split(',')
                # deleting empty symbols:
                mask_el_str = list(filter(None, mask_el_str))
                # not make integer:
                return [int(x) for x in mask_el_str]

            if (len(mask) > 0) and (len(maskWithout) == 0):
                for hyp in all_hyps_list:
                    if all(x in hyp for x in mask_elems(mask)):
                        new_after_mask.append(hyp)

            elif (len(mask) == 0) and (len(maskWithout) > 0):
                for hyp in all_hyps_list:
                    if not any(x in hyp for x in mask_elems(maskWithout)):
                        new_after_mask.append(hyp)

            elif (len(mask) > 0) and (len(maskWithout) > 0):
                for hyp in all_hyps_list:
                    if (not any(x in hyp for x in mask_elems(maskWithout))) and (all(x in hyp for x in mask_elems(mask))):
                        new_after_mask.append(hyp)


            # now dealing with length

            if length > 0:
                print(length)
                for hyp in new_after_mask:
                    if len(hyp) != length:
                        new_after_mask.remove(hyp)




            # and finally print result in output:
            self.DisplayFilter.clear()
            for el in new_after_mask:
                self.DisplayFilter.append(str(el).replace('[', '').replace(']', ''))
            # show length of masked list:
            self.LeftNumberFilters.setText(str(len(new_after_mask)))
            # stay at top and not scroll to the bottom:
            self.DisplayFilter.verticalScrollBar().setValue(0)



    def nav_status(self):
        '''
        works on change Status window
        return status in hint-statusbar
        '''
        # button becomes disabled while function start:
        nav_butt_active = []

        # check status of the inputted hypothesis (hint):
        hint = self.NavInputHypHint.text()
        hint = hint.replace(' ', '')
        hint = hint.split(',')
        hint = list(filter(None, hint))
        hint = sorted([int(x) for x in hint])  # sorted added!!

        # put result in output window:
        # makin' Status window ready:
        if len(hint) == 0:
            self.NavNextLevel.clear()
            nav_butt_active = [1]

        # enable wrapping
        self.NavNextLevel.setLineWrapMode(True)
        self.NavNextLevel.setText(orbit_closure(hint, sign, nav_butt_active))#.replace('[', '').replace(']', ''))
        if len(nav_butt_active) == 0:
            self.NavUp.setEnabled(False)
            self.NavDown.setEnabled(False)
        else:
            self.NavUp.setEnabled(True)
            self.NavDown.setEnabled(True)

        # make zero delta counter, by the way:
        self.spinBox.setValue(0)

        # clear masks and filters:
        self.LowLeftOnly_3.clear()
        self.LowLeftOnly_4.clear()

    def nav_up(self):
        self.NavNextLevel.clear()
        # fragment will make as func as in previous
        hint = self.NavInputHypHint.text()
        hint = hint.replace(' ', '')
        hint = hint.split(',')
        hint = list(filter(None, hint))
        hint = [int(x) for x in hint]

        # makin output window ready:
        if len(hint) == 0:
            self.NavNextLevel.clear()
        self.NavNextLevel.setLineWrapMode(False)
        # printing text into output form:
        global navbuttonpressed
        navbuttonpressed = 1
        global after_nav_up
        after_nav_up = sorted(go_up_reasons(hint, sign), key=len)
        for el in after_nav_up:
            self.NavNextLevel.append(str(el).replace('[', '').replace(']', ''))
        # stay at top and not scroll to the bottom:
        self.NavNextLevel.verticalScrollBar().setValue(0)

    def nav_down(self):
        self.NavNextLevel.clear()
        # fragment will make as func as in previous
        hint = self.NavInputHypHint.text()
        hint = hint.replace(' ', '')
        hint = hint.split(',')
        hint = list(filter(None, hint))
        hint = [int(x) for x in hint]

        # makin output window ready:
        if len(hint) == 0:
            self.NavNextLevel.clear()
        self.NavNextLevel.setLineWrapMode(False)
        # printing text into output form:
        global navbuttonpressed
        navbuttonpressed = 0
        global after_nav_down
        after_nav_down = sorted(go_down_objects(hint, sign), key=len)
        for el in after_nav_down:
            self.NavNextLevel.append(str(el).replace('[', '').replace(']', ''))
        # stay at top and not scroll to the bottom:
        self.NavNextLevel.verticalScrollBar().setValue(0)

    def left_only_nav(self):
        if navbuttonpressed == 0:
            all_hyps_list = after_nav_down
        elif navbuttonpressed == 1:
            all_hyps_list = after_nav_up
        else:
            return 'Вычислите следующий уровень, чтобы отфильтровать гипотезы из него'

        # text that appear in With input window, remove spaces:
        mask = self.LowLeftOnly_3.text().replace(' ', '')
        # text that appear in Without input window, remove spaces:
        maskWithout = self.LowLeftOnly_4.text().replace(' ', '')

        #check if anything inputted in the mask window:
        if (len(mask) == 0) and (len(maskWithout) == 0):
            # delete all the masks - print back whole the current border:
            for el in all_hyps_list:
                self.NavNextLevel.append(str(el).replace('[', '').replace(']', ''))
                #self.LowBordLen.setText(str(len(all_hyps_list)))

        else:
            new_after_mask = []

            def mask_elems(maska):
                # divide inputted by ',':
                mask_el_str = maska.split(',')
                # deleting empty symbols:
                mask_el_str = list(filter(None, mask_el_str))
                # make integer:
                return [int(x) for x in mask_el_str]

            if (len(mask) > 0) and (len(maskWithout) == 0):
                for hyp in all_hyps_list:
                    if all(x in hyp for x in mask_elems(mask)):
                        new_after_mask.append(hyp)

            elif (len(mask) == 0) and (len(maskWithout) > 0):
                for hyp in all_hyps_list:
                    if not any(x in hyp for x in mask_elems(maskWithout)):
                        new_after_mask.append(hyp)

            elif (len(mask) > 0) and (len(maskWithout) > 0):
                for hyp in all_hyps_list:
                    if (not any(x in hyp for x in mask_elems(maskWithout))) and (all(x in hyp for x in mask_elems(mask))):
                        new_after_mask.append(hyp)

            # and finally print result in output:
            self.NavNextLevel.clear()
            for el in new_after_mask:
                self.NavNextLevel.append(str(el).replace('[', '').replace(']', ''))
            # show length of masked list:
            #self.LowBordLen.setText(str(len(new_after_mask)))
            # stay at top and not scroll to the bottom:
            self.NavNextLevel.verticalScrollBar().setValue(0)

    def get_delta(self):
        '''
        :return: partial hypotheses with more or less reasons (county)
        '''
        delta = self.spinBox.value()

        # check status of the inputted hypothesis (hint):
        hint = self.NavInputHypHint.text()
        hint = hint.replace(' ', '')
        hint = hint.split(',')
        hint = list(filter(None, hint))
        hint = sorted([int(x) for x in hint])

        self.NavNextLevel.clear()
        self.NavNextLevel.setLineWrapMode(False)

        for el in sorted(get_partial(hint, sign, delta), key=len):
            self.NavNextLevel.append(str(el).replace('[', '').replace(']', ''))

        # stay at top and not scroll to the bottom:
        self.LowBord.verticalScrollBar().setValue(0)




        #algorithm

        #take all the text in Output window:
        #all_hyps = [self.LowBord.toPlainText().split('\n')]
        #bigl = all_hyps[0]
        #print(bigl)  # ['0, 8, 36, 38', '4, 8, 36, 40', '4, 21, 22, 36, 38',
        #print(bigl[0])  # 0, 8, 36, 38

        #in_window = [all_hyps[0][0].split(',')]
        #print (len(in_window))
        #in cycle:
        #int_w = [int(x) for x in in_window[1]]
        #print (int_w, int_w[0], type(int_w[0]))  # gives [0, 8, 36, 38]


    #reason_converter:

    def crypt_print_all(self):
        self.DecryptOutput.clear()
        for string in crypt_names():
            self.DecryptOutput.append(str(string).replace('[', '').replace(']', '').replace("'", ""))

    def crypt_reas_nums_to_names(self):
        hyp = self.DecryptByNumber.text().replace(' ', '')
        # divide inputted by ',':
        hyp_el_str = hyp.split(',')
        # deleting empty symbols:
        hyp_el_str = list(filter(None, hyp_el_str))
        # make integer:
        hyp_int = [int(x) for x in hyp_el_str]

        self.DecryptOutput.clear()
        for item in hyp_int:
            self.DecryptOutput.append(str(item) + ' - ' + str(decrypt_numbers(item)))

    def crypt_select_byone_reas_names_to_nums(self):
        text = str(self.comboBox.currentText())
        print(text)


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()


