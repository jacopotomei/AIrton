from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QGridLayout, QLineEdit, QTabWidget, QFileDialog
import sys

# Struttura app
# self.audio  > contiene il file audio grezzo importato dall'utente
# self.video  > contiene il file video grezzo importato dall'utente
# self.rpm    > contiene tutti i dati relativi all'analisi giri
# self.gear   > contiene tutti i dati relativi all'analisi dei cambi marcia
# self.speed  > contiene tutti i dati relativi all'analisi della velocità
# self.steer  > contiene tutti i dati relativi all'analisi dell'angolo di sterzo
# self.lean   > contiene tutti i dati relativi all'analisi dell'angolo di piega

# Subclass QMainWindow
class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Titolo finestra
        self.setWindowTitle("AIrton")
        # Dimensioni finestra
        # self.setGeometry(50,50,1820,980)
        # Creazione componenti
        self.creaWidgets()
        # Finestra visibile
        self.showMaximized()

    def creaWidgets(self):
        # Layout principale
        self.mainLayout = QGridLayout()
        # Importazione file audio
        self.cmdAudio = QPushButton("Import audio")
        self.cmdAudio.clicked.connect(self.importaAudio)
        self.mainLayout.addWidget(self.cmdAudio,0,0)
        self.txtAudio = QLineEdit()
        self.mainLayout.addWidget(self.txtAudio,0,1)
        # Importazione file video
        self.cmdVideo = QPushButton("Import video")
        self.cmdVideo.clicked.connect(self.importaVideo)
        self.mainLayout.addWidget(self.cmdVideo,1,0)
        self.txtVideo = QLineEdit()
        self.mainLayout.addWidget(self.txtVideo,1,1)
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.West)
        self.tabs.setMovable(False)
        self.tbRPMAnalysis = QWidget()
        self.tabs.addTab(self.tbRPMAnalysis,"Engine speed analysis")
        self.tbGearAnalysis = QWidget()
        self.tabs.addTab(self.tbGearAnalysis,"Gear analysis")
        self.tbSpeedAnalysis = QWidget()
        self.tabs.addTab(self.tbSpeedAnalysis,"Vehicle speed analysis")
        self.tbSteerAnalysis = QWidget()
        self.tabs.addTab(self.tbSteerAnalysis,"Steer analysis")
        self.tbLeanAnalysis = QWidget()
        self.tabs.addTab(self.tbLeanAnalysis,"Lean analysis")
        self.mainLayout.addWidget(self.tabs,2,0,1,2)
        # Cosmetica grid layout
        self.mainLayout.setColumnStretch(0, 1)
        self.mainLayout.setColumnStretch(1, 10)
        self.mainLayout.setRowStretch(0, 2)
        self.mainLayout.setRowStretch(1, 2)
        self.mainLayout.setRowStretch(2, 20)
        # Impostazione main widget
        w = QWidget()
        w.setLayout(self.mainLayout)
        self.setCentralWidget(w)

    def importaAudio(self):
        # File dialog
        nome = QFileDialog.getOpenFileName(self,"Import audio file", "/home", "WAV Audio Files (*.wav)")
        if nome[0] != "":
            # Impostazione del nome file
            self.txtAudio.setText(nome[0])
            # Lettura del file
            self.audio

    def importaVideo(self):
        # File dialog
        nome = QFileDialog.getOpenFileName(self,"Import video file", "/home", "Video Files (*.mp4 *.avi *.webm)")
        if nome[0] != "":
            self.txtVideo.setText(nome[0])
        
# Creazione istanza app
app = QApplication(sys.argv)
# Creazione main window
window = MyMainWindow()
# Start app
app.exec()