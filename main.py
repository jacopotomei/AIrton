from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QGridLayout, QLineEdit, QTabWidget, QFileDialog, QComboBox
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
        if sys.platform == "win32":
            self.tbRPMAnalysis.setStyleSheet("background-color: whitesmoke")
        self.tabs.addTab(self.tbRPMAnalysis,"Engine speed analysis")
        self.tbGearAnalysis = QWidget()
        if sys.platform == "win32":
            self.tbGearAnalysis.setStyleSheet("background-color: whitesmoke")
        self.tabs.addTab(self.tbGearAnalysis,"Gear analysis")
        self.tbSpeedAnalysis = QWidget()
        if sys.platform == "win32":
            self.tbSpeedAnalysis.setStyleSheet("background-color: whitesmoke")
        self.tabs.addTab(self.tbSpeedAnalysis,"Vehicle speed analysis")
        self.tbSteerAnalysis = QWidget()
        if sys.platform == "win32":
            self.tbSteerAnalysis.setStyleSheet("background-color: whitesmoke")
        self.tabs.addTab(self.tbSteerAnalysis,"Steer analysis")
        self.tbLeanAnalysis = QWidget()
        if sys.platform == "win32":
            self.tbLeanAnalysis.setStyleSheet("background-color: whitesmoke")
        self.tabs.addTab(self.tbLeanAnalysis,"Lean analysis")
        self.mainLayout.addWidget(self.tabs,2,0,1,2)
        # Cosmetica grid layout
        self.mainLayout.setColumnStretch(0, 1)
        self.mainLayout.setColumnStretch(1, 10)
        self.mainLayout.setRowStretch(0, 2)
        self.mainLayout.setRowStretch(1, 2)
        self.mainLayout.setRowStretch(2, 20)
        # Tab <Engine speed analysis>
        self.tbRPMAnalysisLayout = QGridLayout()
        self.tbRPMAnalysis.setLayout(self.tbRPMAnalysisLayout)
        label = QLabel("Engine type")
        self.tbRPMAnalysisLayout.addWidget(label,0,0)
        self.tbRPMAnalysis_engineType = QComboBox()
        self.tbRPMAnalysis_engineType.addItems(["F1","MotoGP","GT"])
        self.tbRPMAnalysis_engineType.currentIndexChanged.connect(self.cambiaEngineType)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_engineType,0,1)
        label = QLabel("Sampling rate")
        self.tbRPMAnalysisLayout.addWidget(label,1,0)
        self.tbRPMAnalysis_samplingRate = QLineEdit()
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_samplingRate,1,1)
        label = QLabel("Overlap")
        self.tbRPMAnalysisLayout.addWidget(label,2,0)
        self.tbRPMAnalysis_overlap = QLineEdit("0.1")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_overlap,2,1)
        label = QLabel("Window length")
        self.tbRPMAnalysisLayout.addWidget(label,3,0)
        self.tbRPMAnalysis_windowLength = QLineEdit("1024")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_windowLength,3,1)
        label = QLabel("FFT points")
        self.tbRPMAnalysisLayout.addWidget(label,4,0)
        self.tbRPMAnalysis_nFFT = QLineEdit("4096")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_nFFT,4,1)
        #label = QLabel("Hop length")
        #self.tbRPMAnalysisLayout.addWidget(label,5,0)
        #self.tbRPMAnalysis_hopLength = QLineEdit("410")
        #self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_hopLength,5,1)
        label = QLabel("Main harmonics")
        self.tbRPMAnalysisLayout.addWidget(label,5,0)
        self.tbRPMAnalysis_baseHarmonics = QLineEdit("9")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_baseHarmonics,5,1)
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

    def cambiaEngineType(self):
        if self.sender().currentIndex() == 0:
            # F1
            self.tbRPMAnalysis_overlap.setText("0.1")
            self.tbRPMAnalysis_windowLength.setText("1024")
            self.tbRPMAnalysis_nFFT.setText("4096")
        elif self.sender().currentIndex() == 1:
            # MotoGP
            self.tbRPMAnalysis_overlap.setText("0.025")
            self.tbRPMAnalysis_windowLength.setText("2048")
            self.tbRPMAnalysis_nFFT.setText("16384")
        elif self.sender().currentIndex() == 2:
            # GT
            self.tbRPMAnalysis_overlap.setText("0.1")
            self.tbRPMAnalysis_windowLength.setText("1024")
            self.tbRPMAnalysis_nFFT.setText("4096")
        
        
# Creazione istanza app
app = QApplication(sys.argv)
# Creazione main window
window = MyMainWindow()
# Start app
app.exec()