import sys
import os
import json
from typing import Dict, Any
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QFontDatabase, QIcon
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QStackedWidget, QFrame, QHBoxLayout, QSizePolicy, QGridLayout, QTableWidget, QTableWidgetItem
)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from scipy.interpolate import interp1d
from sympy import symbols, simplify, expand

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

class Pages(QWidget):
    def __init__(self, custom_font_family):
        super().__init__()
        self.setWindowTitle("Переключение страниц")
        self.showFullScreen()
        self.custom = custom_font_family
        self.flags = [
            ("США", resource_path("flags/usa.png")),
            ("Казахстан", resource_path("flags/kaz.jpg")),
            ("Россия", resource_path("flags/rus.jpg")),
            ("Китай", resource_path("flags/chi.jpg")),
            ("Британия", resource_path("flags/bri.jpg")),
            ("Корея", resource_path("flags/kor.jpg"))
        ]

        self.data = self.load_all_country_data()
        self.filename = resource_path("../data/spec.json")
        self.data_spec = self.load_all_specializations()
            
        screen = QFrame()
        screen.setStyleSheet("""
            background: #EDFCFF;
        """)
        main_layout = QHBoxLayout()
        m_layout =QVBoxLayout()
        main_layout.setContentsMargins(200, 0, 100,0)
        button_layout = QVBoxLayout()
        maintext_label = QLabel("Главный экран")
        maintext_label.setStyleSheet(f"""
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 60px;
            font-family: "{custom_font_family}";
        """)
        maintext_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_layout.addWidget(maintext_label)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.buttons = {}
        image_map = {
            "Аналитика": resource_path("images/analitics.png"),
            "Прогноз": resource_path("images/scrutiny.png"),
            "О программе": resource_path("images/info.png"),
            "Связаться с нами": resource_path("images/operator.png"),
            "Выход": resource_path("images/logout.png")
        }

        for name, image_path in image_map.items():
            btn = QPushButton(name)
            btn.setFixedSize(360, 60)
            btn.setStyleSheet(f"""
                QPushButton {{
                    padding: 10px; 
                    font-size: 18px; 
                    font-family: {custom_font_family}; 
                    border: 0px; 
                    border-radius: 15px; 
                    background: #42AAD0; 
                    color: #fff;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: #3399CC;
                }}
            """)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.enterEvent = lambda e, path=image_path: self.show_image(path)
            button_layout.addWidget(btn)
            self.buttons[name] = btn
        self.buttons["Аналитика"].clicked.connect(self.analiticsPage)
        self.buttons["Прогноз"].clicked.connect(self.prognoz)
        self.buttons["Связаться с нами"].clicked.connect(self.contactPage)
        self.buttons["О программе"].clicked.connect(self.aboutPage)
        self.buttons["Выход"].clicked.connect(self.exitApp)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.under_image = QLabel()
        self.under_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.under_image.setStyleSheet(f"""
            font-size: 18px;
            color: #2F4F4F;
            font-weight: 700;
            font-family: "{custom_font_family}";
            padding: 10px;
        """)
        self.show_image(resource_path("images/doctor.png"))

        main_layout.addLayout(button_layout, 1)
        m_layout.addStretch(1)
        m_layout.addWidget(self.image_label, 3)
        m_layout.addWidget(self.under_image)
        m_layout.addStretch(1)
        main_layout.addStretch(2)
        main_layout.addLayout(m_layout)
        main_layout.addStretch(1)

        screen.setLayout(main_layout)

        self.stack = QStackedWidget()
        self.stack.addWidget(screen)
        self.stack.setCurrentIndex(0)

        self.analitics_screen()
        self.createContactPage()
        self.createAboutPage()
        self.prognoz_screen()
        self.open_kazakhstan_details()
        self.open_kazakhstan_details_v2()

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.addWidget(self.stack)
        self.setLayout(main_layout)

    def load_country_data(self, country_name: str) -> Dict[int, int]:
        filename = resource_path(f"../data/{country_name}.json")
        try:
            os.makedirs(resource_path("../data"), exist_ok=True)
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    return {int(year): value for year, value in data.items()}
        except (json.JSONDecodeError, IOError):
            pass
        
        base_value = 5 + hash(country_name) % 10
        return {year: year * base_value for year in range(2016, 2025)}

    def load_all_country_data(self) -> Dict[str, Dict[int, int]]:
        countries = ["США", "Казахстан", "Россия", "Китай", "Британия", "Корея"]
        return {country: self.load_country_data(country) for country in countries}

    def save_country_data(self, country_name: str, data: Dict[int, int]):
        filename = resource_path(f"../data/{country_name}.json")
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
        except IOError as e:
            print(f"Ошибка сохранения данных: {e}")

    def load_specialization_data(self, spec_name: str) -> Dict[int, int]:
        try:
            os.makedirs(resource_path("../data"), exist_ok=True)
            if os.path.exists(self.filename):
                with open(self.filename, 'r') as f:
                    all_data = json.load(f)
                    if spec_name in all_data:
                        return {int(year): value for year, value in all_data[spec_name].items()}
        except (json.JSONDecodeError, IOError):
            pass
        
        base_value = 1000 + hash(spec_name) % 1000
        return {year: base_value + (year - 2016) * 50 for year in range(2016, 2025)}

    def load_all_specializations(self) -> Dict[str, Dict[int, int]]:
        try:
            os.makedirs(resource_path("../data"), exist_ok=True)
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
                    return {
                        spec: {int(year): value for year, value in year_data.items()}
                        for spec, year_data in raw_data.items()
                    }
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    def save_specialization_data(self, spec_name: str, data: Dict[int, int]):
        all_data = self.load_all_specializations()
        all_data[spec_name] = {str(year): value for year, value in data.items()}
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=4, ensure_ascii=False)
        except IOError as e:
            print(f"Ошибка при сохранении: {e}")


    def analiticsPage(self):
        self.stack.setCurrentIndex(1)
    def prognoz(self):
        self.stack.setCurrentIndex(4)
    def backToMain(self):
        self.stack.setCurrentIndex(1)
    def backToProg(self):
        self.stack.setCurrentIndex(4)
    def vrachi(self):
        self.stack.setCurrentIndex(5)
    def vrachi_v2(self):
        self.stack.setCurrentIndex(6)
    def backToAnalitics(self):
        self.stack.setCurrentIndex(0)

    def make_open_country_handler(self, country_name):
        def handler():
            self.open_country_page(country_name)
        return handler
    
    def make_open_prognoz(self, country_name):
        def handler():
            self.open_country_prognoz(country_name)
        return handler
    def make_open_page(self, country_name):
        def handler():
            self.open_country_page(country_name)
        return handler

    def analitics_screen(self):
        screen = QWidget()
        screen.setStyleSheet("""
            background: #EDFCFF;
        """)
        layout = QVBoxLayout(screen)
        grid = QGridLayout()
        vid = QLabel("Выбор страны")
        vid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vid.setStyleSheet(f"""
            font-size: 36px;
            font-weight: 700;
            font-family: "{self.custom}";
            color: #2F4F4F;
        """)

        for i, (name, icon_path) in enumerate(self.flags):
            btn = QPushButton()
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(274, 154))
            btn.setToolTip(name)
            btn.setFixedSize(278, 158)

            btn.setStyleSheet(f"""
                QPushButton {{
                    border: none;
                    background-color: transparent;
                }}
                QPushButton:hover {{
                    background-color: rgba(0, 0, 0, 40);
                }}
            """)
            btn.clicked.connect(self.make_open_country_handler(name))

            row = i // 3
            col = i % 3
            grid.addWidget(btn, row, col)

        layout.addStretch(7)
        layout.addWidget(vid)
        layout.addStretch(1)
        layout.addLayout(grid)

        back_btn = QPushButton("Назад")
        back_btn.setFixedSize(120, 40)
        back_btn.setStyleSheet(f"""
                QPushButton {{
                    padding: 10px; 
                    font-size: 18px; 
                    font-family: {self.custom}; 
                    border: 0px; 
                    border-radius: 15px; 
                    background: #42AAD0; 
                    color: #fff;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: #3399CC; 
                }}
            """)
        back_btn.clicked.connect(self.backToAnalitics)
        layout.addStretch(1)
        back_layout = QHBoxLayout()
        back_layout.addStretch()
        back_layout.addWidget(back_btn)
        back_layout.addStretch()
        layout.addLayout(back_layout)
        layout.addStretch(5)

        self.stack.addWidget(screen)

    def open_country_page(self, country_name):
        screen = QWidget()
        screen.setStyleSheet("""
            background: #EDFCFF;
        """)
        layout = QVBoxLayout(screen)
        
        content_layout = QHBoxLayout()
        content_layoutV = QVBoxLayout()
        left_panel = QVBoxLayout()
        left_panel.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Год", "Численность"])
        self.info_table.setFixedWidth(220)
        self.info_table.setStyleSheet("""
            QTableWidget {
                font-size: 16px;
                border-radius: 10px;
            }
            QHeaderView::section {
                background-color: #B2EBF2;
                font-weight: bold;
            }
        """)
        
        def populate_table():
            if country_name=="США" or country_name=="Казахстан" or country_name=="Россия" or country_name=="Китай" or country_name== "Британия" or country_name== "Корея":
                self.info_table.setRowCount(len(self.data[country_name]))
                for row, (year, value) in enumerate(sorted(self.data[country_name].items())):
                    year_item = QTableWidgetItem(str(year))
                    year_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    value_item = QTableWidgetItem(str(value))
                    self.info_table.setItem(row, 0, year_item)
                    self.info_table.setItem(row, 1, value_item)
            else:
                self.info_table.setRowCount(len(self.data_spec[country_name]))
                for row, (year, value) in enumerate(sorted(self.data_spec[country_name].items())):
                    year_item = QTableWidgetItem(str(year))
                    year_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    value_item = QTableWidgetItem(str(value))
                    self.info_table.setItem(row, 0, year_item)
                    self.info_table.setItem(row, 1, value_item)

        populate_table()
        left_panel.addWidget(self.info_table)
    
        self.poly_label = QLabel()
        self.poly_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.poly_label.setStyleSheet("""
            font-size: 14px;
            color: #333;
            margin-top: 10px;
        """)
        
        poly_btn = QPushButton("Провести полиному")
        self.poly_label.setFixedHeight(0)
        poly_btn.setStyleSheet("""
            QPushButton {
                padding: 10px; 
                font-size: 16px; 
                font-family: %s; 
                border: 0px; 
                border-radius: 15px; 
                background: #42AAD0; 
                color: #fff;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3399CC; 
            }
        """ % self.custom)
        
        polyline_btn = QPushButton("Показать полином")
        polyline_btn.setFixedSize(220, 40)
        polyline_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 16px;
                font-family: Arial;
                border: 0px;
                border-radius: 15px;
                background: #42AAD0; 
                color: #fff;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3399CC;
            }
        """)
        
        left_panel.addStretch()
        
        content_layout.addLayout(left_panel)
        
        figure = plt.figure(figsize=(6, 4))
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)
        if country_name=="США" or country_name=="Казахстан" or country_name=="Россия" or country_name=="Китай" or country_name== "Британия" or country_name== "Корея":
            years = list(self.data[country_name].keys())
            values = list(self.data[country_name].values())
        else:
            years = list(self.data_spec[country_name].keys())
            values = list(self.data_spec[country_name].values())
        
        ax.plot(years, values, 'o', label='Данные')
        if country_name=="США" or country_name=="Казахстан" or country_name=="Россия" or country_name=="Китай" or country_name== "Британия" or country_name== "Корея":
            ax.set_title(f"Численность врачей в {country_name}")
        else:
            ax.set_title(f"Численность врачей ({country_name})")
        ax.set_xlabel("Год")
        ax.set_ylabel("Численность")
        ax.grid(True)
        ax.set_facecolor("#EDFCFF") 
        figure.set_facecolor("#EDFCFF")
        
        content_layoutV.addWidget(self.poly_label)
        content_layoutV.addWidget(canvas)
        content_layout.addLayout(content_layoutV)
        layout.addLayout(content_layout)
        

        back_btn = QPushButton("Назад")
        back_btn.setFixedSize(120, 40)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 10px; 
                font-size: 18px; 
                font-family: {self.custom}; 
                border: 0px; 
                border-radius: 15px; 
                background: #42AAD0; 
                color: #fff;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #3399CC; 
            }}
        """)
        
        back_layout = QHBoxLayout()
        back_layout.addStretch()
        back_layout.addWidget(back_btn)
        # back_layout.addWidget(polyline_btn)
        back_layout.addWidget(poly_btn)
        if country_name == "Казахстан" or country_name == "Терапевт" or country_name == "Педиатр"  or country_name == "Кардиолог"  or country_name == "Невролог"  or country_name == "Хирург"  or country_name == "Гинеколог"  or country_name == "Психиатр"  or country_name == "Стоматолог"  or country_name == "Офтальмолог"  or country_name == "Отоларинголог"  or country_name == "Уролог" or country_name == "Эндокринолог":
            kz_button = QPushButton("Виды врачей")
            kz_button.setFixedSize(220, 40)
            kz_button.setStyleSheet(f"""
                QPushButton {{
                    padding: 10px;
                    font-size: 16px;
                    font-family: {self.custom};
                    border: 0px;
                    border-radius: 15px;
                    background: #66BB6A;
                    color: #fff;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: #558B2F;
                }}
            """)
            kz_button.clicked.connect(self.vrachi_v2)
        if country_name == "Казахстан" or country_name == "Терапевт" or country_name == "Педиатр"  or country_name == "Кардиолог"  or country_name == "Невролог"  or country_name == "Хирург"  or country_name == "Гинеколог"  or country_name == "Психиатр"  or country_name == "Стоматолог"  or country_name == "Офтальмолог"  or country_name == "Отоларинголог"  or country_name == "Уролог" or country_name == "Эндокринолог":
            back_layout.addWidget(kz_button)
        back_layout.addStretch()
        layout.addLayout(back_layout)
        
        poly_shown = [False]
        polynomial_shown = [False]
        
        def update_data_from_table():
            new_data = {}
            for row in range(self.info_table.rowCount()):
                year = int(self.info_table.item(row, 0).text())
                try:
                    value = int(self.info_table.item(row, 1).text())
                except ValueError:
                    value = 0
                new_data[year] = value
            if country_name=="США" or country_name=="Казахстан" or country_name=="Россия" or country_name=="Китай" or country_name== "Британия" or country_name== "Корея":
                self.data[country_name] = new_data
            else:
                self.data_spec[country_name] = new_data
            self.save_country_data(country_name, new_data)
            redraw_graph()

        self.info_table.cellChanged.connect(update_data_from_table)
        
        def poly_to_string(coeffs):
            terms = []
            degree = len(coeffs) - 1
            for i, coef in enumerate(coeffs):
                power = degree - i
                coef_str = f"{coef:.5f}"
                if power == 0:
                    term = f"{coef_str}"
                elif power == 1:
                    term = f"{coef_str}x"
                else:
                    term = f"{coef_str}x^{power}"
                terms.append(term)
            return " + ".join(terms).replace("+ -", "- ")


        def redraw_graph():
            ax.clear()
            ax.set_title(f"Численность врачей в {country_name}")
            ax.set_xlabel("Год")
            ax.set_ylabel("Численность")
            ax.grid(True)
            ax.set_facecolor("#EDFCFF")
            figure.set_facecolor("#EDFCFF")

            if country_name=="США" or country_name=="Казахстан" or country_name=="Россия" or country_name=="Китай" or country_name== "Британия" or country_name== "Корея":
                current_data = self.data[country_name]
            else:
                current_data = self.data_spec[country_name]
            years = list(current_data.keys())
            values = list(current_data.values())

            ax.plot(years, values, 'o', label='Фактические данные')

            if poly_shown[0]:
                interp = interp1d(years, values, kind='cubic')
                x_interp = np.linspace(min(years), max(years), 200)
                y_interp = interp(x_interp)
                ax.plot(x_interp, y_interp, 'r--', label='Интерполяция')

            if polynomial_shown[0]:
                coeffs = np.polyfit(years, values, len(years) - 1)
                poly = np.poly1d(coeffs)
                x_poly = np.linspace(min(years), max(years), 200)
                y_poly = poly(x_poly)
                ax.plot(x_poly, y_poly, 'g-', label='Полином')

            canvas.draw()

        def on_poly_btn_clicked():
            poly_shown[0] = not poly_shown[0]
            if country_name=="США" or country_name=="Казахстан" or country_name=="Россия" or country_name=="Китай" or country_name== "Британия" or country_name== "Корея":
                current_data = self.data[country_name]
            else:
                current_data = self.data_spec[country_name]
            years = list(current_data.keys())
            values = list(current_data.values())

            if poly_shown[0]:
                x = symbols('x')
                P = 0
                for i in range(len(years)):
                    xi, yi = years[i], values[i]
                    Li = 1
                    for j in range(len(years)):
                        if i != j:
                            xj = years[j]
                            Li *= (x - xj)/(xi - xj)
                    P += yi * Li

                P = simplify(expand(P))

                P_str = str(P.evalf(4)).replace("**", "^")
                self.poly_label.setText(f"P(x) = {P_str}")
                self.poly_label.setFixedHeight(30)
                self.poly_label.setStyleSheet(f"""
                    font-family: {self.custom}; 
                    border-radius: 14px;
                    background:rgb(137, 198, 198);
                """)
                poly_btn.setText("Убрать полиному")
                poly_btn.setStyleSheet("""
                    QPushButton {
                        padding: 10px;
                        font-size: 16px;
                        font-family: Arial;
                        border: 0px;
                        border-radius: 15px;
                        background: #E53935;
                        color: #fff;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background-color: #D32F2F;
                    }
                """)
            else:
                self.poly_label.setText("")
                self.poly_label.setStyleSheet("""
                    border-radius: 14px;
                    background: #EDFCFF;
                """)
                poly_btn.setText("Провести полиному")
                poly_btn.setStyleSheet("""
                    QPushButton {
                        padding: 10px;
                        font-size: 16px;
                        font-family: Arial;
                        border: 0px;
                        border-radius: 15px;
                        background: #42AAD0;
                        color: #fff;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background-color: #3399CC;
                    }
                """)

            redraw_graph()



        def on_polyline_btn_clicked():
            polynomial_shown[0] = not polynomial_shown[0]
            if polynomial_shown[0]:
                polyline_btn.setText("Убрать полином")
                polyline_btn.setStyleSheet("""
                    QPushButton {
                        padding: 10px;
                        font-size: 16px;
                        font-family: Arial;
                        border: 0px;
                        border-radius: 15px;
                        background: #E53935;
                        color: #fff;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background-color: #D32F2F;
                    }
                """)
            else:
                polyline_btn.setText("Показать полином")
                polyline_btn.setStyleSheet("""
                    QPushButton {
                        padding: 10px;
                        font-size: 16px;
                        font-family: Arial;
                        border: 0px;
                        border-radius: 15px;
                        background: #42AAD0;
                        color: #fff;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background-color: #3399CC;
                    }
                """)
            redraw_graph()

        poly_btn.clicked.connect(on_poly_btn_clicked)
        polyline_btn.clicked.connect(on_polyline_btn_clicked)
        back_btn.clicked.connect(self.backToMain)

        self.stack.addWidget(screen)
        self.stack.setCurrentWidget(screen)

    def createContactPage(self):
        self.contact_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(100, 50, 100, 50)
        layout.setSpacing(25)

        label = QLabel("Связаться с нами")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"""
            font-size: 36px;
            font-weight: 700;
            font-family: "{self.custom}";
            color: #2F4F4F;
        """)

        info = QLabel(
            "Email: asset@gmail.com\n"
            "Телефон: +7 707 007 1403\n\n"
            "Instagram: @sset\n"
            "Telegram: @sset\n"
            "WhatsApp: +7 707 007 1403"
        )
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet(f"""
            font-size: 24px;
            font-family: "{self.custom}";
            color: #555555;
        """)
        info.setWordWrap(True)

        back_btn = QPushButton("Назад")
        back_btn.setFixedSize(180, 50)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #42AAD0;
                color: white;
                font-family: "{self.custom}";
                font-size: 18px;
                font-weight: 600;
                border-radius: 15px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #3399CC;
            }}
        """)
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.clicked.connect(self.goToMainScreen)

        layout.addStretch(1)
        layout.addWidget(label)
        layout.addWidget(info)
        layout.addStretch(1)
        layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.contact_widget.setLayout(layout)
        self.stack.addWidget(self.contact_widget)

    def createAboutPage(self):
        self.about_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(100, 50, 100, 50)
        layout.setSpacing(25)

        label = QLabel("О программе")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"""
            font-size: 36px;
            font-weight: 700;
            font-family: "{self.custom}";
            color: #2F4F4F;
        """)

        about_text = (
            "Приложение предназначено для глубокого анализа численности врачей по различным странам и медицинским специализациям. "
            "Основная цель — предоставить пользователю мощный инструмент для визуализации, прогнозирования и стратегического планирования "
            "медицинских кадров на основе актуальных статистических данных.\n\n"
            "В программе используются следующие технологии и библиотеки:\n"
            "• Python — основной язык разработки\n"
            "• PyQt6 — создание графического интерфейса\n"
            "• TK Canvas — построение графиков\n"
            "• NumPy — численные вычисления и интерполяция\n"
            "Основные функциональные возможности:\n\n"
            "• Отображение численности врачей по странам: США, Казахстан, Россия, Китай, Южная Корея и Великобритания\n"
            "• Отдельный анализ численности врачей по специализациям (например, терапевты, хирурги и т. д.) для Казахстана\n"
            "• Графики с интерполяцией и полиномиальной аппроксимацией, позволяющие оценить тренды по годам\n"
            "• Прогноз численности врачей на 2025–2026 годы, основанный на моделях интерполяции и математических вычислениях\n"
            "• Использование метода Гаусса для вычисления коэффициентов полинома и построения точных прогнозных графиков\n"
            "• Интерактивный график с возможностью сравнения динамики между странами и специализациями\n"
            "• Информационная таблица по всем странам с данными по годам и значениями численности врачей\n\n"
            "Программа будет полезна аналитикам в сфере здравоохранения, исследователям, государственным и частным организациям, "
            "занятым в мониторинге, оценке и стратегическом планировании медицинских кадров. Благодаря визуальному интерфейсу и мощным аналитическим возможностям, "
            "приложение позволяет быстро принимать решения, основанные на точных данных и прогнозах."
        )
        info = QLabel(about_text)
        info.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        info.setStyleSheet(f"""
            font-size: 20px;
            font-family: "{self.custom}";
            color: #555555;
        """)
        info.setWordWrap(True)

        back_btn = QPushButton("Назад")
        back_btn.setFixedSize(180, 50)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #42AAD0;
                color: white;
                font-family: "{self.custom}";
                font-size: 18px;
                font-weight: 600;
                border-radius: 15px;
                border: none;
            }}
            QPushButton:hover {{
                background-color: #3399CC;
            }}
        """)
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.clicked.connect(self.goToMainScreen)

        layout.addStretch(1)
        layout.addWidget(label)
        layout.addWidget(info)
        layout.addStretch(1)
        layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.about_widget.setLayout(layout)
        self.stack.addWidget(self.about_widget)

    def goToMainScreen(self):
        self.stack.setCurrentIndex(0)
    
    def contactPage(self):
        index = self.stack.indexOf(self.contact_widget)
        if index != -1:
            self.stack.setCurrentIndex(index)

    def aboutPage(self):
        index = self.stack.indexOf(self.about_widget)
        if index != -1:
            self.stack.setCurrentIndex(index)

    def exitApp(self):
        self.close()

    def prognoz_screen(self):
        screen = QWidget()
        screen.setStyleSheet("""
            background: #EDFCFF;
        """)
        layout = QVBoxLayout(screen)
        grid = QGridLayout()
        vid = QLabel("Выбор страны")
        vid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vid.setStyleSheet(f"""
            font-size: 36px;
            font-weight: 700;
            font-family: "{self.custom}";
            color: #2F4F4F;
        """)

        for i, (name, icon_path) in enumerate(self.flags):
            btn = QPushButton()
            btn.setIcon(QIcon(icon_path))
            btn.setIconSize(QSize(274, 154))
            btn.setToolTip(name)
            btn.setFixedSize(278, 158)

            btn.setStyleSheet(f"""
                QPushButton {{
                    border: none;
                    background-color: transparent;
                }}
                QPushButton:hover {{
                    background-color: rgba(0, 0, 0, 40);
                }}
            """)
            btn.clicked.connect(self.make_open_prognoz(name))

            row = i // 3
            col = i % 3
            grid.addWidget(btn, row, col)

        layout.addStretch(7)
        layout.addWidget(vid)
        layout.addStretch(1)
        layout.addLayout(grid)
        back_btn = QPushButton("Назад")
        back_btn.setFixedSize(120, 40)
        back_btn.setStyleSheet(f"""
                QPushButton {{
                    padding: 10px; 
                    font-size: 18px; 
                    font-family: {self.custom}; 
                    border: 0px; 
                    border-radius: 15px; 
                    background: #42AAD0; 
                    color: #fff;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: #3399CC; 
                }}
            """)
        back_btn.clicked.connect(self.backToAnalitics)
        layout.addStretch(1)
        back_layout = QHBoxLayout()
        back_layout.addStretch()
        back_layout.addWidget(back_btn)
        back_layout.addStretch()
        layout.addLayout(back_layout)
        layout.addStretch(5)

        self.stack.addWidget(screen)
    
    def open_country_prognoz(self, country_name):
        screen = QWidget()
        screen.setStyleSheet("""
            background: #EDFCFF;
        """)
        layout = QVBoxLayout(screen)
        content_layout = QHBoxLayout()
        content_layoutV = QVBoxLayout()
        left_layout = QVBoxLayout()
        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["Год", "Численность"])
        self.info_table.setFixedWidth(240)
        self.info_table.setFixedHeight(420)
        self.info_table.setStyleSheet("""
            QTableWidget {
                font-size: 16px;
                border-radius: 2px;
            }
            QHeaderView::section {
                background-color: #B2EBF2;
                font-weight: bold;
            }
        """)

        if country_name=="США" or country_name=="Казахстан" or country_name=="Россия" or country_name=="Китай" or country_name== "Британия" or country_name== "Корея":
            def populate_table():
                self.info_table.setRowCount(len(self.data[country_name]))
                for row, (year, value) in enumerate(sorted(self.data[country_name].items())):
                    year_item = QTableWidgetItem(str(year))
                    year_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    value_item = QTableWidgetItem(str(value))
                    self.info_table.setItem(row, 0, year_item)
                    self.info_table.setItem(row, 1, value_item)
        else:
            def populate_table():
                self.info_table.setRowCount(len(self.data_spec[country_name]))
                for row, (year, value) in enumerate(sorted(self.data_spec[country_name].items())):
                    year_item = QTableWidgetItem(str(year))
                    year_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    value_item = QTableWidgetItem(str(value))
                    self.info_table.setItem(row, 0, year_item)
                    self.info_table.setItem(row, 1, value_item)
        populate_table()
        left_layout.addWidget(self.info_table)
        
        self.poly_label = QLabel()
        self.poly_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.poly_label.setStyleSheet("""
            font-size: 14px;
            color: #333;
            margin-top: 10px;
            font-family: Arial;
        """)
        left_layout.addStretch()
        
        content_layout.addLayout(left_layout)

        figure = plt.figure(figsize=(6, 4))
        canvas = FigureCanvas(figure)
        ax = figure.add_subplot(111)
        content_layoutV.addWidget(self.poly_label) 
        content_layoutV.addWidget(canvas, stretch=1)
        content_layout.addLayout(content_layoutV) 
        layout.addLayout(content_layout)

        def update_data_from_table():
            new_data = {}
            for row in range(self.info_table.rowCount()):
                year = int(self.info_table.item(row, 0).text())
                if year in (2025, 2026):
                    continue 
                try:
                    value = int(self.info_table.item(row, 1).text())
                except ValueError:
                    value = 0
                new_data[year] = value
            if country_name=="США" or country_name=="Казахстан" or country_name=="Россия" or country_name=="Китай" or country_name== "Британия" or country_name== "Корея":
                self.data[country_name] = new_data
                self.save_country_data(country_name, new_data)
                if poly_shown[0]:
                    on_poly_btn_clicked()
            else:
                self.data_spec[country_name] = new_data
                self.save_specialization_data(country_name, new_data)
                if poly_shown[0]:
                    on_poly_btn_clicked()
            redraw_graph()

        self.info_table.cellChanged.connect(update_data_from_table)
        
        def redraw_graph():
            ax.clear()
            ax.set_title(f"Численность врачей в {country_name}")
            if country_name == "Терапевт" or country_name == "Педиатр"  or country_name == "Кардиолог"  or country_name == "Невролог"  or country_name == "Хирург"  or country_name == "Гинеколог"  or country_name == "Психиатр"  or country_name == "Стоматолог"  or country_name == "Офтальмолог"  or country_name == "Отоларинголог"  or country_name == "Уролог" or country_name == "Эндокринолог":
                ax.set_title(f"Численность врачей({country_name})")
            ax.set_xlabel("Год")
            ax.set_ylabel("Численность")
            ax.grid(True)
            ax.set_facecolor("#EDFCFF")
            figure.set_facecolor("#EDFCFF")
            if country_name=="США" or country_name=="Казахстан" or country_name=="Россия" or country_name=="Китай" or country_name== "Британия" or country_name== "Корея":
                years = list(self.data[country_name].keys())
                values = list(self.data[country_name].values())
            else:
                years = list(self.data_spec[country_name].keys())
                values = list(self.data_spec[country_name].values())
            ax.plot(years, values, 'o', label='Фактические данные')

            if forecast_shown[0]:
                coeffs = np.polyfit([int(y) for y in years], [float(v) for v in values], 2)
                poly = np.poly1d(coeffs)
                forecast_years = [2025, 2026]
                forecast_values = [poly(y) for y in forecast_years]
                ax.plot(forecast_years, forecast_values, 'gs', label='Прогноз')
                for year, val in zip(forecast_years, forecast_values):
                    ax.text(year, val, f"{int(val)}", fontsize=10, ha='left', va='bottom', color='green')

            canvas.draw()
        if country_name=="США" or country_name=="Казахстан" or country_name=="Россия" or country_name=="Китай" or country_name== "Британия" or country_name== "Корея":
            years = list(self.data[country_name].keys())
            values = list(self.data[country_name].values())
        else:
            years = list(self.data_spec[country_name].keys())
            values = list(self.data_spec[country_name].values())
        ax.plot(years, values, 'o', label='Данные')
        ax.set_title(f"Численность врачей в {country_name}")
        if country_name == "Терапевт" or country_name == "Педиатр"  or country_name == "Кардиолог"  or country_name == "Невролог"  or country_name == "Хирург"  or country_name == "Гинеколог"  or country_name == "Психиатр"  or country_name == "Стоматолог"  or country_name == "Офтальмолог"  or country_name == "Отоларинголог"  or country_name == "Уролог" or country_name == "Эндокринолог":
            ax.set_title(f"Численность врачей({country_name})")
        ax.set_xlabel("Год")
        ax.set_ylabel("Численность")
        ax.grid(True)
        ax.set_facecolor("#EDFCFF") 
        figure.set_facecolor("#EDFCFF")  

        def poly_to_string(coeffs):
            terms = []
            degree = len(coeffs) - 1
            for i, coef in enumerate(coeffs):
                power = degree - i
                coef_str = f"{coef:.5f}"
                if power == 0:
                    term = f"{coef_str}"
                elif power == 1:
                    term = f"{coef_str}x"
                else:
                    term = f"{coef_str}x^{power}"
                terms.append(term)
            return " + ".join(terms).replace("+ -", "- ")

        poly_shown = [False]
        forecast_shown = [False]

        def on_forecast_btn_clicked():
            forecast_shown[0] = not forecast_shown[0]

            self.info_table.blockSignals(True)

            if forecast_shown[0]:
                forecast_years = [2025, 2026]
                if country_name=="США" or country_name=="Казахстан" or country_name=="Россия" or country_name=="Китай" or country_name== "Британия" or country_name== "Корея":
                    coeffs = np.polyfit(list(self.data[country_name].keys()), list(self.data[country_name].values()), 2)
                else:
                    coeffs = np.polyfit(list(self.data_spec[country_name].keys()), list(self.data_spec[country_name].values()), 2)
                poly = np.poly1d(coeffs)
                forecast_values = [int(poly(year)) for year in forecast_years]

                current_row_count = self.info_table.rowCount()
                self.info_table.setRowCount(current_row_count + 2)
                for i, year in enumerate(forecast_years):
                    year_item = QTableWidgetItem(str(year))
                    year_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    value_item = QTableWidgetItem(str(forecast_values[i]))
                    value_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    self.info_table.setItem(current_row_count + i, 0, year_item)
                    self.info_table.setItem(current_row_count + i, 1, value_item)

            else:
                for row in reversed(range(self.info_table.rowCount())):
                    year_item = self.info_table.item(row, 0)
                    if year_item and year_item.text() in ("2025", "2026"):
                        self.info_table.removeRow(row)

            self.info_table.blockSignals(False)
            redraw_graph()

            if poly_shown[0]:
                poly_shown[0]=False
                on_poly_btn_clicked()

            if forecast_shown[0]:
                forecast_btn.setText("Убрать прогноз")
                forecast_btn.setStyleSheet("""
                    QPushButton {
                        padding: 10px;
                        font-size: 16px;
                        font-family: Arial;
                        border: 0px;
                        border-radius: 15px;
                        background: #E53935;
                        color: #fff;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background-color: #D32F2F;
                    }
                """)
            else:
                forecast_btn.setText("Показать прогноз")
                forecast_btn.setStyleSheet("""
                    QPushButton {
                        padding: 10px;
                        font-size: 16px;
                        font-family: Arial;
                        border: 0px;
                        border-radius: 15px;
                        background: #42AAD0; 
                        color: #fff;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background-color: #3399CC; 
                    }
                """)

        def on_poly_btn_clicked():
            poly_shown[0] = not poly_shown[0]

            ax.clear()
            ax.set_title(f"Численность врачей в {country_name}")
            if country_name in ["Терапевт", "Педиатр", "Кардиолог", "Невролог", "Хирург", "Гинеколог", "Психиатр", "Стоматолог", "Офтальмолог", "Отоларинголог", "Уролог", "Эндокринолог"]:
                ax.set_title(f"Численность врачей({country_name})")

            ax.set_xlabel("Год")
            ax.set_ylabel("Численность")
            ax.grid(True)
            ax.set_facecolor("#EDFCFF")
            figure.set_facecolor("#EDFCFF")

            if country_name in ["США", "Казахстан", "Россия", "Китай", "Британия", "Корея"]:
                current_data = self.data[country_name]
            else:
                current_data = self.data_spec[country_name]

            years = list(current_data.keys())
            values = list(current_data.values())
            ax.plot(years, values, 'o', label='Фактические данные')

            if poly_shown[0]:
                from sympy import symbols, simplify, expand

                x = symbols('x')
                P = 0
                for i in range(len(years)):
                    xi, yi = years[i], values[i]
                    Li = 1
                    for j in range(len(years)):
                        if i != j:
                            xj = years[j]
                            Li *= (x - xj) / (xi - xj)
                    P += yi * Li

                P = simplify(expand(P))
                P_str = str(P.evalf(4)).replace("**", "^")
                self.poly_label.setText(f"P(x) = {P_str}")
                self.poly_label.setFixedHeight(30)
                self.poly_label.setStyleSheet(f"""
                    font-family: {self.custom}; 
                    border-radius: 14px;
                    background:rgb(137, 198, 198);
                """)

                coeffs = np.polyfit([int(y) for y in years], [float(v) for v in values], 2)
                interp_years = years[:]
                interp_values = values[:]

                if forecast_shown[0]:
                    poly = np.poly1d(coeffs)
                    forecast_years = [2025, 2026]
                    forecast_values = [poly(y) for y in forecast_years]
                    interp_years += forecast_years
                    interp_values += forecast_values

                interp = interp1d(interp_years, interp_values, kind='cubic', fill_value="extrapolate")
                x_interp = np.linspace(min(interp_years), max(interp_years), 300)
                y_interp = interp(x_interp)
                ax.plot(x_interp, y_interp, 'r--', label='Интерполяция по точкам')

                poly_btn.setText("Удалить полиному")
                poly_btn.setStyleSheet("""
                    QPushButton {
                        padding: 10px;
                        font-size: 16px;
                        font-family: Arial;
                        border: 0px;
                        border-radius: 15px;
                        background: #E53935;
                        color: #fff;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background-color: #D32F2F;
                    }
                """)
            else:
                self.poly_label.setText("")
                self.poly_label.setStyleSheet("""
                    border-radius: 14px;
                    background: #EDFCFF;
                """)
                poly_btn.setText("Провести полиному")
                poly_btn.setStyleSheet("""
                    QPushButton {
                        padding: 10px;
                        font-size: 16px;
                        font-family: Arial;
                        border: 0px;
                        border-radius: 15px;
                        background: #42AAD0;
                        color: #fff;
                        font-weight: 600;
                    }
                    QPushButton:hover {
                        background-color: #3399CC;
                    }
                """)

            if forecast_shown[0]:
                coeffs = np.polyfit([int(y) for y in years], [float(v) for v in values], 2)
                poly = np.poly1d(coeffs)
                forecast_years = [2025, 2026]
                forecast_values = [poly(y) for y in forecast_years]
                ax.plot(forecast_years, forecast_values, 'gs', label='Прогноз')
                for year, val in zip(forecast_years, forecast_values):
                    ax.text(year, val, f"{int(val)}", fontsize=10, ha='left', va='bottom', color='green')

            canvas.draw()

            canvas.draw()

        poly_btn = QPushButton("Провести полиному")
        poly_btn.setFixedSize(220, 40)
        poly_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 10px; 
                font-size: 16px; 
                font-family: {self.custom}; 
                border: 0px; 
                border-radius: 15px; 
                background: #42AAD0; 
                color: #fff;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #3399CC; 
            }}
        """)
        poly_btn.clicked.connect(on_poly_btn_clicked)
        
        back_btn = QPushButton("Назад")
        back_btn.setFixedSize(120, 40)
        back_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 10px; 
                font-size: 18px; 
                font-family: {self.custom}; 
                border: 0px; 
                border-radius: 15px; 
                background: #42AAD0; 
                color: #fff;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #3399CC; 
            }}
        """)
        
        forecast_btn = QPushButton("Показать прогноз")
        forecast_btn.setFixedSize(220, 40)
        forecast_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 10px;
                font-size: 16px;
                font-family: {self.custom};
                border: 0px;
                border-radius: 15px;
                background: #42AAD0; 
                color: #fff;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #3399CC; 
            }}
        """)
        forecast_btn.clicked.connect(on_forecast_btn_clicked)
        
        back_btn.clicked.connect(self.backToProg)
        if country_name == "Казахстан" or country_name == "Терапевт" or country_name == "Педиатр"  or country_name == "Кардиолог"  or country_name == "Невролог"  or country_name == "Хирург"  or country_name == "Гинеколог"  or country_name == "Психиатр"  or country_name == "Стоматолог"  or country_name == "Офтальмолог"  or country_name == "Отоларинголог"  or country_name == "Уролог" or country_name == "Эндокринолог":
            kz_button = QPushButton("Виды врачей")
            kz_button.setFixedSize(220, 40)
            kz_button.setStyleSheet(f"""
                QPushButton {{
                    padding: 10px;
                    font-size: 16px;
                    font-family: {self.custom};
                    border: 0px;
                    border-radius: 15px;
                    background: #66BB6A;
                    color: #fff;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: #558B2F;
                }}
            """)
            kz_button.clicked.connect(self.vrachi)
    
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(back_btn)
        button_layout.addWidget(forecast_btn)
        button_layout.addWidget(poly_btn)
        if country_name == "Казахстан" or country_name == "Терапевт" or country_name == "Педиатр"  or country_name == "Кардиолог"  or country_name == "Невролог"  or country_name == "Хирург"  or country_name == "Гинеколог"  or country_name == "Психиатр"  or country_name == "Стоматолог"  or country_name == "Офтальмолог"  or country_name == "Отоларинголог"  or country_name == "Уролог" or country_name == "Эндокринолог":
            button_layout.addWidget(kz_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.stack.addWidget(screen)
        self.stack.setCurrentWidget(screen)

    def open_kazakhstan_details(self):
        screen = QWidget()
        screen.setStyleSheet("""
            background: #EDFCFF;
        """)
        layout = QVBoxLayout(screen)
        layout.setContentsMargins(400,0,400,0)
        grid = QGridLayout()
        layout.addStretch(4)
        vid = QLabel("Выбор видов врачей")
        vid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vid.setStyleSheet(f"""
            font-size: 36px;
            font-weight: 700;
            font-family: "{self.custom}";
            color: #2F4F4F;
        """)
        layout.addWidget(vid)

        for i, (name, years) in enumerate(self.data_spec.items()):
            btn = QPushButton(name)
            btn.setFixedSize(220, 60)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 10px;
                    font-size: 16px;
                    font-family: Arial;
                    border: 0px;
                    border-radius: 15px;
                    background: #42AAD0;
                    color: #fff;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: #3399CC;
                }
            """)
            btn.clicked.connect(self.make_open_prognoz(name))
            row = i // 3
            col = i % 3
            grid.addWidget(btn, row, col)

        layout.addStretch(2)
        layout.addLayout(grid)
        obshi_btn = QPushButton("Общее")
        obshi_btn.setFixedSize(700, 60)
        obshi_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 10px; 
                font-size: 18px; 
                font-family: {self.custom}; 
                border: 0px; 
                border-radius: 15px; 
                background: #42AAD0; 
                color: #fff;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #3399CC; 
            }}
        """)
        obshi_btn.clicked.connect(self.make_open_prognoz("Казахстан"))

        back_btn = QPushButton("Назад")
        back_btn.setFixedSize(120, 40)
        back_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 16px;
                font-family: Arial;
                border: 0px;
                border-radius: 15px;
                background: #E53935;
                color: #fff;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)
        back_btn.clicked.connect(self.make_open_prognoz("Казахстан"))
        back_layoutv = QHBoxLayout()
        back_layoutv.addStretch()
        back_layoutv.addWidget(obshi_btn)
        back_layoutv.addStretch()
        layout.addLayout(back_layoutv)
        layout.addStretch(1)
        back_layout = QHBoxLayout()
        back_layout.addStretch()
        back_layout.addWidget(back_btn)
        back_layout.addStretch()
        layout.addLayout(back_layout)
        layout.addStretch(5)

        self.stack.addWidget(screen)

    def open_kazakhstan_details_v2(self):
        screen = QWidget()
        screen.setStyleSheet("""
            background: #EDFCFF;
        """)
        layout = QVBoxLayout(screen)
        layout.setContentsMargins(400,0,400,0)
        grid = QGridLayout()
        layout.addStretch(4)
        vid = QLabel("Выбор видов врачей")
        vid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vid.setStyleSheet(f"""
            font-size: 36px;
            font-weight: 700;
            font-family: "{self.custom}";
            color: #2F4F4F;
        """)
        layout.addWidget(vid)

        for i, (name, years) in enumerate(self.data_spec.items()):
            btn = QPushButton(name)
            btn.setFixedSize(220, 60)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 10px;
                    font-size: 16px;
                    font-family: Arial;
                    border: 0px;
                    border-radius: 15px;
                    background: #42AAD0;
                    color: #fff;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: #3399CC;
                }
            """)
            btn.clicked.connect(self.make_open_page(name))
            row = i // 3
            col = i % 3
            grid.addWidget(btn, row, col)

        layout.addStretch(2)
        layout.addLayout(grid)
        obshi_btn = QPushButton("Общее")
        obshi_btn.setFixedSize(700, 60)
        obshi_btn.setStyleSheet(f"""
            QPushButton {{
                padding: 10px; 
                font-size: 18px; 
                font-family: {self.custom}; 
                border: 0px; 
                border-radius: 15px; 
                background: #42AAD0; 
                color: #fff;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #3399CC; 
            }}
        """)
        obshi_btn.clicked.connect(self.make_open_page("Казахстан"))

        back_btn = QPushButton("Назад")
        back_btn.setFixedSize(120, 40)
        back_btn.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-size: 16px;
                font-family: Arial;
                border: 0px;
                border-radius: 15px;
                background: #E53935;
                color: #fff;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #D32F2F;
            }
        """)
        back_btn.clicked.connect(self.make_open_page("Казахстан"))
        back_layoutv = QHBoxLayout()
        back_layoutv.addStretch()
        back_layoutv.addWidget(obshi_btn)
        back_layoutv.addStretch()
        layout.addLayout(back_layoutv)
        layout.addStretch(1)
        back_layout = QHBoxLayout()
        back_layout.addStretch()
        back_layout.addWidget(back_btn)
        back_layout.addStretch()
        layout.addLayout(back_layout)

        layout.addStretch(5)

        self.stack.addWidget(screen)

    def show_image(self, image_path):
        abs_path = resource_path(image_path)
        if os.path.exists(abs_path):
            pixmap = QPixmap(abs_path)
            self.image_label.setPixmap(pixmap.scaled(
                400, 400,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            explanation_map = {
                resource_path("images/analitics.png"): "Раздел аналитики: просмотр графиков и данных по численности врачей.",
                resource_path("images/scrutiny.png"): "Раздел прогноза: оценка будущей численности врачей по странам.",
                resource_path("images/info.png"): "Информация о программе: цели, разработчик, технологии.",
                resource_path("images/operator.png"): "Контакты: как с нами связаться и задать вопрос.",
                resource_path("images/logout.png"): "Выход из приложения.",
                resource_path("images/doctor.png"): "Добро пожаловать! Выберите нужный раздел слева."
            }

            self.under_image.setText(explanation_map.get(abs_path, ""))
        else:
            self.image_label.setText(f"Изображение не найдено: {abs_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    font_path = resource_path("ttf/Raleway-Regular.ttf")
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        font_families = QFontDatabase.applicationFontFamilies(font_id)
        selected_font = font_families[0] if font_families else None
    else:
        selected_font = None

    window = Pages(custom_font_family=selected_font)
    window.show()
    sys.exit(app.exec())
