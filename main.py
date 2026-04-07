from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QLabel, QMessageBox,
    QGridLayout, QLineEdit, QTabWidget, QFileDialog, QComboBox, QCheckBox, QVBoxLayout, QDialog)
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib import pyplot as plt
import re
import sys
import torch
import pickle
import numpy as np
import librosa as lib
import scipy.signal as ss
import scipy.fft as sfft

# Struttura app
# self.audio  > contiene il file audio grezzo importato dall'utente
# self.video  > contiene il file video grezzo importato dall'utente
# self.rpm    > contiene tutti i dati relativi all'analisi giri
# self.gear   > contiene tutti i dati relativi all'analisi dei cambi marcia
# self.speed  > contiene tutti i dati relativi all'analisi della velocità
# self.steer  > contiene tutti i dati relativi all'analisi dell'angolo di sterzo
# self.lean   > contiene tutti i dati relativi all'analisi dell'angolo di piega

# Class FigureCanvas di matplotlib
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

# Class QMainWindow
class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Titolo finestra
        self.setWindowTitle("AIrton")
        # Parametri applicazione
        if sys.platform == "win32":
            self.pwd = __file__[:len(__file__)-__file__[::-1].find("\\")]
        else:
            self.pwd = __file__[:len(__file__)-__file__[::-1].find("/")]
        self.vettoreFrequenze = [8000,16000,22050,44100,48000]
        self.audio_raw = np.array([])
        self.audio_fs = 8000
        # Inizializzazione struttura app
        self.audio_fft = {"f":np.array([]),"t":np.array([]),"stft":np.array([[]])}
        self.rpm = {"i":np.array([]),"t":np.array([]),"y":np.array([]),"yraw":np.array([])}
        self.gear = {"iup":np.array([]),"idown":np.array([]),"dmarciaup":np.array([]),"dmarciadown":np.array([]),"dgiriup":np.array([]),"dgiridown":np.array([])}
        # Creazione componenti
        self.creaWidgets()
        # Disattivo le shortcut di matplotlib
        for k in plt.rcParams:
            if k.startswith("keymap"):
                [plt.rcParams[k].remove(h) for h in plt.rcParams[k]]
        # Finestra visibile
        self.showMaximized()

    def creaWidgets(self):
        # Layout principale
        self.mainLayout = QGridLayout()
        # GridLayout per importazione
        w = QWidget()
        self.mainLayout.addWidget(w,0,0,1,10)
        importLayout = QGridLayout()
        w.setLayout(importLayout)
        importTab = QTabWidget()
        importLayout.addWidget(importTab,0,0)
        # Import audio
        audioTab = QWidget()
        importTab.addTab(audioTab,"Audio")
        audioTabLayout = QGridLayout()
        audioTab.setLayout(audioTabLayout)
        self.cmdAudioImport = QPushButton("Import audio")
        self.cmdAudioImport.clicked.connect(self.importaAudio)
        audioTabLayout.addWidget(self.cmdAudioImport,0,0)
        self.txtAudio = QLineEdit()
        audioTabLayout.addWidget(self.txtAudio,0,1,1,10)
        l = QLabel("Frequency")
        l.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        audioTabLayout.addWidget(l,1,0)
        self.ddAudioFreq = QComboBox()
        self.ddAudioFreq.addItems([str(f)+" Hz" for f in self.vettoreFrequenze])
        audioTabLayout.addWidget(self.ddAudioFreq,1,1)
        l = QLabel("Channels")
        l.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        audioTabLayout.addWidget(l,1,2)
        self.ddAudioChannels = QComboBox()
        self.ddAudioChannels.addItems(["Mono","Stereo"])
        audioTabLayout.addWidget(self.ddAudioChannels,1,3)
        self.chkAudioMeanValueFilter = QCheckBox(text="Mean-value filter")
        self.chkAudioMeanValueFilter.setChecked(True)
        audioTabLayout.addWidget(self.chkAudioMeanValueFilter,1,4)
        self.chkAudioTrim = QCheckBox(text="Trim")
        self.chkAudioTrim.stateChanged.connect(self.abilitaAudioTrim)
        audioTabLayout.addWidget(self.chkAudioTrim,1,5)
        l = QLabel("Start time [s]")
        l.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        audioTabLayout.addWidget(l,1,6)
        self.txtAudioStartTime = QLineEdit("0")
        self.txtAudioStartTime.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.txtAudioStartTime.setEnabled(False)
        audioTabLayout.addWidget(self.txtAudioStartTime,1,7)
        l = QLabel("End time [s]")
        l.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignVCenter)
        audioTabLayout.addWidget(l,1,8)
        self.txtAudioEndTime = QLineEdit("0")
        self.txtAudioEndTime.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.txtAudioEndTime.setEnabled(False)
        audioTabLayout.addWidget(self.txtAudioEndTime,1,9)
        [audioTabLayout.setColumnStretch(c, 1) for c in range(10)]
        audioTabLayout.setColumnStretch(10,10)
        # Import video
        videoTab = QWidget()
        importTab.addTab(videoTab,"Video")
        videoTabLayout = QGridLayout()
        videoTab.setLayout(videoTabLayout)
        self.cmdVideo = QPushButton("Import video")
        self.cmdVideo.clicked.connect(self.importaVideo)
        videoTabLayout.addWidget(self.cmdVideo,0,0)
        self.txtVideo = QLineEdit()
        videoTabLayout.addWidget(self.txtVideo,0,1)
        
        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.West)
        self.tabs.setMovable(False)
        self.tbRPMAnalysis = QWidget()
        if sys.platform == "win33":
            self.tbRPMAnalysis.setStyleSheet("background-color: whitesmoke")
        self.tabs.addTab(self.tbRPMAnalysis,"Engine speed analysis")
        self.tbGearAnalysis = QWidget()
        if sys.platform == "win33":
            self.tbGearAnalysis.setStyleSheet("background-color: whitesmoke")
        self.tabs.addTab(self.tbGearAnalysis,"Gears identification")
        self.tbSpeedAnalysis = QWidget()
        if sys.platform == "win33":
            self.tbSpeedAnalysis.setStyleSheet("background-color: whitesmoke")
        self.tabs.addTab(self.tbSpeedAnalysis,"Vehicle speed analysis")
        self.tbSteerAnalysis = QWidget()
        if sys.platform == "win33":
            self.tbSteerAnalysis.setStyleSheet("background-color: whitesmoke")
        self.tabs.addTab(self.tbSteerAnalysis,"Steer analysis")
        self.tbLeanAnalysis = QWidget()
        if sys.platform == "win33":
            self.tbLeanAnalysis.setStyleSheet("background-color: whitesmoke")
        self.tabs.addTab(self.tbLeanAnalysis,"Lean analysis")
        self.mainLayout.addWidget(self.tabs,1,0,1,10)
        # Cosmetica grid layout
        self.mainLayout.setColumnStretch(0, 1)
        self.mainLayout.setColumnStretch(1, 10)
        self.mainLayout.setRowStretch(0, 1)
        self.mainLayout.setRowStretch(1, 10)
        
        # Tab <Engine speed analysis>
        self.tbRPMAnalysisLayout = QGridLayout()
        self.tbRPMAnalysis.setLayout(self.tbRPMAnalysisLayout)

        label = QLabel("Pitch shift")
        self.tbRPMAnalysisLayout.addWidget(label,0,0)
        self.tbRPMAnalysis_PitchShift = QLineEdit("1")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_PitchShift,0,1)
        
        label = QLabel("Engine type")
        self.tbRPMAnalysisLayout.addWidget(label,1,0)
        self.tbRPMAnalysis_engineType = QComboBox()
        self.tbRPMAnalysis_engineType.addItems(["F1","MotoGP","GT"])
        self.tbRPMAnalysis_engineType.currentIndexChanged.connect(self.cambiaEngineType)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_engineType,1,1)
        
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
        
        self.tbRPMAnalysis_cmdFFT = QPushButton("Calculate FFT")
        self.tbRPMAnalysis_cmdFFT.clicked.connect(self.CalcolaFFT)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_cmdFFT,5,0,1,2)
        
        label = QLabel("Analysis type")
        self.tbRPMAnalysisLayout.addWidget(label,6,0)
        self.tbRPMAnalysis_analysisType = QComboBox()
        self.tbRPMAnalysis_analysisType.addItems(["Standard","AI"])
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_analysisType,6,1)
        
        label = QLabel("Main harmonics")
        self.tbRPMAnalysisLayout.addWidget(label,7,0)
        self.tbRPMAnalysis_baseHarmonics = QLineEdit("4,6,9")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_baseHarmonics,7,1)

        label = QLabel("Maximum engine speed")
        self.tbRPMAnalysisLayout.addWidget(label,8,0)
        self.tbRPMAnalysis_maxRPM = QLineEdit("12500")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_maxRPM,8,1)
        
        label = QLabel("Minimum engine speed")
        self.tbRPMAnalysisLayout.addWidget(label,9,0)
        self.tbRPMAnalysis_minRPM = QLineEdit("5000")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_minRPM,9,1)
        
        label = QLabel("Manual multiplier")
        self.tbRPMAnalysisLayout.addWidget(label,10,0)
        self.tbRPMAnalysis_Multiplier = QLineEdit("1")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_Multiplier,10,1)
        
        label = QLabel("NN model")
        self.tbRPMAnalysisLayout.addWidget(label,11,0)
        self.tbRPMAnalysis_NNmodel = QComboBox()
        self.tbRPMAnalysis_NNmodel.addItems(["F1_DeepSpeech_5_torch","F1_DeepSpeech_9_torch"])
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_NNmodel,11,1)

        label = QLabel("Correction")
        self.tbRPMAnalysisLayout.addWidget(label,12,0)
        self.tbRPMAnalysis_Correction = QComboBox()
        self.tbRPMAnalysis_Correction.addItems(["No correction","Weighted average","Median"])
        self.tbRPMAnalysis_Correction.setCurrentIndex(2)
        self.tbRPMAnalysis_Correction.currentIndexChanged.connect(self.AnalisiAutomaticaRPM)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_Correction,12,1)

        label = QLabel("Filter")
        self.tbRPMAnalysisLayout.addWidget(label,13,0)
        self.tbRPMAnalysis_Filter = QComboBox()
        self.tbRPMAnalysis_Filter.addItems(["No filter","Rolling average","Median filter","Lowpass filter"])
        self.tbRPMAnalysis_Filter.setCurrentIndex(2)
        self.tbRPMAnalysis_Filter.currentIndexChanged.connect(self.AnalisiAutomaticaRPM)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_Filter,13,1)

        label = QLabel("Filter kernel size")
        self.tbRPMAnalysisLayout.addWidget(label,14,0)
        self.tbRPMAnalysis_KernelSize = QLineEdit("11")
        self.tbRPMAnalysis_KernelSize.editingFinished.connect(self.AnalisiAutomaticaRPM)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_KernelSize,14,1)

        label = QLabel("LP filter frequency")
        self.tbRPMAnalysisLayout.addWidget(label,15,0)
        self.tbRPMAnalysis_FilterFrequency = QLineEdit("0.1")
        self.tbRPMAnalysis_FilterFrequency.editingFinished.connect(self.AnalisiAutomaticaRPM)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_FilterFrequency,15,1)
        
        self.tbRPMAnalysis_cmdStartOver = QPushButton("Start over")
        self.tbRPMAnalysis_cmdStartOver.clicked.connect(self.CancellaTuttoRPM)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_cmdStartOver,16,0,1,2)

        self.tbRPMAnalysis_cmdAutomatedAnalysis = QPushButton("Run automated analysis")
        self.tbRPMAnalysis_cmdAutomatedAnalysis.clicked.connect(self.AnalisiAutomaticaRPM)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_cmdAutomatedAnalysis,17,0,1,2)
        
        self.tbRPMAnalysis_cmdManualAnalysis = QPushButton("Run manual analysis")
        self.tbRPMAnalysis_cmdManualAnalysis.clicked.connect(self.AnalisiManualeRPM)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_cmdManualAnalysis,18,0,1,2)

        self.tbRPMAnalysis_cmdManualAnalysis = QPushButton("Accept manual reconstruction")
        self.tbRPMAnalysis_cmdManualAnalysis.clicked.connect(self.AnalisiManualeRPM)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_cmdManualAnalysis,19,0,1,2)
        
        self.tbRPMAnalysis_cmdCaricaRPM = QPushButton("Load engine speed")
        self.tbRPMAnalysis_cmdCaricaRPM.clicked.connect(self.caricaRPM)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_cmdCaricaRPM,20,0,1,2)

        self.tbRPMAnalysis_cmdSalvaRPM = QPushButton("Save engine speed")
        self.tbRPMAnalysis_cmdSalvaRPM.clicked.connect(self.salvaRPM)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_cmdSalvaRPM,21,0,1,2)
        
        # Tab per FFT
        self.subtbRPMAnalysis = QTabWidget()
        subtbRPMAnalysis_FFT = QWidget()
        self.subtbRPMAnalysis_FFT_figure = Figure(figsize=(15,10))
        self.subtbRPMAnalysis_FFT_canvas = FigureCanvas(self.subtbRPMAnalysis_FFT_figure)
        self.subtbRPMAnalysis_FFT_toolbar = NavigationToolbar2QT(self.subtbRPMAnalysis_FFT_canvas, self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.subtbRPMAnalysis_FFT_toolbar)
        vbox.addWidget(self.subtbRPMAnalysis_FFT_canvas)
        subtbRPMAnalysis_FFT.setLayout(vbox)
        self.subtbRPMAnalysis.addTab(subtbRPMAnalysis_FFT,"FFT")
        
        # Tab per analisi giri
        subtbRPMAnalysis_Analysis = QWidget()
        self.subtbRPMAnalysis_Analysis_figure = Figure(figsize=(15,10))
        self.subtbRPMAnalysis_Analysis_canvas = FigureCanvas(self.subtbRPMAnalysis_Analysis_figure)
        self.subtbRPMAnalysis_Analysis_toolbar = NavigationToolbar2QT(self.subtbRPMAnalysis_Analysis_canvas, self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.subtbRPMAnalysis_Analysis_toolbar)
        vbox.addWidget(self.subtbRPMAnalysis_Analysis_canvas)
        subtbRPMAnalysis_Analysis.setLayout(vbox)
        self.subtbRPMAnalysis.addTab(subtbRPMAnalysis_Analysis,"Analysis")

        # Tab per correzione giri
        subtbRPMAnalysis_Correction = QWidget()
        self.subtbRPMAnalysis_Correction_figure = Figure(figsize=(15,10))
        self.subtbRPMAnalysis_Correction_canvas = FigureCanvas(self.subtbRPMAnalysis_Correction_figure)
        self.subtbRPMAnalysis_Correction_toolbar = NavigationToolbar2QT(self.subtbRPMAnalysis_Correction_canvas, self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.subtbRPMAnalysis_Correction_toolbar)
        vbox.addWidget(self.subtbRPMAnalysis_Correction_canvas)
        subtbRPMAnalysis_Correction.setLayout(vbox)
        self.subtbRPMAnalysis.addTab(subtbRPMAnalysis_Correction,"Correction")
        
        # Stretch colonne
        self.tbRPMAnalysisLayout.addWidget(self.subtbRPMAnalysis,0,2,30,1)
        self.tbRPMAnalysisLayout.setColumnStretch(0,1)
        self.tbRPMAnalysisLayout.setColumnStretch(1,1)
        self.tbRPMAnalysisLayout.setColumnStretch(2,15)

        # Tab <Gears identification>
        self.tbGearAnalysisLayout = QGridLayout()
        self.tbGearAnalysis.setLayout(self.tbGearAnalysisLayout)
        
        label = QLabel("RPM filter frequency (upshift)")
        self.tbGearAnalysisLayout.addWidget(label,0,0)
        self.tbGearAnalysis_LPupFrequency = QLineEdit("0.25")
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_LPupFrequency,0,1)
        
        label = QLabel("RPM derivative zero threshold (upshift)")
        self.tbGearAnalysisLayout.addWidget(label,1,0)
        self.tbGearAnalysis_dRPM0upThreshold = QLineEdit("50")
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_dRPM0upThreshold,1,1)
        
        label = QLabel("RPM derivative threshold (upshift)")
        self.tbGearAnalysisLayout.addWidget(label,2,0)
        self.tbGearAnalysis_dRPMupThreshold = QLineEdit("150")
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_dRPMupThreshold,2,1)
        
        label = QLabel("RPM filter frequency (downshift)")
        self.tbGearAnalysisLayout.addWidget(label,3,0)
        self.tbGearAnalysis_LPdownFrequency = QLineEdit("0.5")
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_LPdownFrequency,3,1)
        
        label = QLabel("Median filter kernel (downshift)")
        self.tbGearAnalysisLayout.addWidget(label,4,0)
        self.tbGearAnalysis_MedFiltKernel = QLineEdit("11")
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_MedFiltKernel,4,1)

        label = QLabel("Derivative filter frequency (downshift)")
        self.tbGearAnalysisLayout.addWidget(label,5,0)
        self.tbGearAnalysis_LPFiltFrequency = QLineEdit("0.25")
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_LPFiltFrequency,5,1)
        
        label = QLabel("Window for maximum search (downshift)")
        self.tbGearAnalysisLayout.addWidget(label,6,0)
        self.tbGearAnalysis_MaxSearchWindow = QLineEdit("0,8")
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_MaxSearchWindow,6,1)
        
        label = QLabel("Window for minimum search (downshift)")
        self.tbGearAnalysisLayout.addWidget(label,7,0)
        self.tbGearAnalysis_MinSearchWindow = QLineEdit("5,2")
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_MinSearchWindow,7,1)
        
        label = QLabel("Window for 2° maximum search (downshift)")
        self.tbGearAnalysisLayout.addWidget(label,8,0)
        self.tbGearAnalysis_Max2SearchWindow = QLineEdit("0,3")
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_Max2SearchWindow,8,1)
        
        label = QLabel("Engine speed minimum gradient (downshift)")
        self.tbGearAnalysisLayout.addWidget(label,9,0)
        self.tbGearAnalysis_RPMGradient = QLineEdit("220")
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_RPMGradient,9,1)

        self.tbGearAnalysis_cmdCercaUp = QPushButton("Search upshifts")
        self.tbGearAnalysis_cmdCercaUp.clicked.connect(self.CercaCambiate)
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_cmdCercaUp,10,0,1,2)

        self.tbGearAnalysis_cmdCercaDown = QPushButton("Search downshifts")
        self.tbGearAnalysis_cmdCercaDown.clicked.connect(self.CercaCambiate)
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_cmdCercaDown,11,0,1,2)

        self.tbGearAnalysis_cmdPulisciUp = QPushButton("Clear all upshifts")
        self.tbGearAnalysis_cmdPulisciUp.clicked.connect(self.CercaCambiate)
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_cmdPulisciUp,12,0,1,2)

        self.tbGearAnalysis_cmdPulisciDown = QPushButton("Clear all downshifts")
        self.tbGearAnalysis_cmdPulisciDown.clicked.connect(self.CercaCambiate)
        self.tbGearAnalysisLayout.addWidget(self.tbGearAnalysis_cmdPulisciDown,13,0,1,2)

        # Tab per analisi marce
        self.subtbGearAnalysis = QTabWidget()
        subtbGearAnalysis_Analysis = QWidget()
        self.subtbGearAnalysis_Analysis_figure = Figure(figsize=(15,10))
        self.subtbGearAnalysis_Analysis_canvas = FigureCanvas(self.subtbGearAnalysis_Analysis_figure)
        self.subtbGearAnalysis_Analysis_toolbar = NavigationToolbar2QT(self.subtbGearAnalysis_Analysis_canvas, self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.subtbGearAnalysis_Analysis_toolbar)
        vbox.addWidget(self.subtbGearAnalysis_Analysis_canvas)
        subtbGearAnalysis_Analysis.setLayout(vbox)
        self.subtbGearAnalysis.addTab(subtbGearAnalysis_Analysis,"Analysis")

        # Stretch colonne
        self.tbGearAnalysisLayout.addWidget(self.subtbGearAnalysis,0,2,30,1)
        self.tbGearAnalysisLayout.setColumnStretch(0,1)
        self.tbGearAnalysisLayout.setColumnStretch(1,1)
        self.tbGearAnalysisLayout.setColumnStretch(2,15)

        # Impostazione main widget applicazione
        w = QWidget()
        w.setLayout(self.mainLayout)
        self.setCentralWidget(w)

    def importaAudio(self):
        # File dialog
        nome = QFileDialog.getOpenFileName(self,"Import audio file", "/home", "WAV Audio Files (*.wav)")
        if nome[0] != "":
            # Pulizia plot
            self.subtbRPMAnalysis_FFT_figure.clf()
            self.subtbRPMAnalysis_FFT_canvas.draw_idle()
            self.subtbRPMAnalysis_Analysis_figure.clf()
            self.subtbRPMAnalysis_Analysis_canvas.draw_idle()
            self.subtbRPMAnalysis_Correction_figure.clf()
            self.subtbRPMAnalysis_Correction_canvas.draw_idle()
            self.subtbGearAnalysis_Analysis_figure.clf()
            self.subtbGearAnalysis_Analysis_canvas.draw_idle()
            # Impostazione del nome file
            self.txtAudio.setText(nome[0].replace("/","\\"))
            # Lettura del file
            fs = self.vettoreFrequenze[self.ddAudioFreq.currentIndex()]
            mono = self.ddAudioChannels.currentText() == "Mono"
            self.audio_raw,self.audio_fs = lib.load(nome[0],sr=fs,mono=mono)
            if self.chkAudioMeanValueFilter.isChecked():
                self.audio_raw = self.audio_raw - self.audio_raw.mean()
            if self.chkAudioTrim.isChecked():
                tstart = np.float32(self.txtAudioStartTime.text())
                tend = np.float32(self.txtAudioEndTime.text())
                if tend > tstart:
                    istart = np.min([self.audio_raw.shape[0],np.max([0,np.floor(fs*tstart)])]).astype(np.int32)
                    iend = np.min([self.audio_raw.shape[0],np.ceil(fs*tend)]).astype(np.int32)
                    self.audio_raw = self.audio_raw[istart:iend]

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
            
    def abilitaAudioTrim(self):
        if self.sender().isChecked():
            self.txtAudioStartTime.setEnabled(True)
            self.txtAudioEndTime.setEnabled(True)
        else:
            self.txtAudioStartTime.setEnabled(False)
            self.txtAudioEndTime.setEnabled(False)
    
    def CalcolaFFT(self):
        if self.audio_raw.shape[0] > 0:
            # Definizione parametri
            overlap = np.float32(self.tbRPMAnalysis_overlap.text())
            win_length = np.int32(self.tbRPMAnalysis_windowLength.text())
            n_fft = np.int32(self.tbRPMAnalysis_nFFT.text())
            hop_length = int(win_length*overlap)
            # Pitch shift
            ps = np.float32(self.tbRPMAnalysis_PitchShift.text())
            if (ps > 0) & (not (np.isnan(ps))):
                if ps != 1:
                    x = lib.effects.pitch_shift(self.audio_raw,sr=self.audio_fs,n_steps=ps,bins_per_octave=12,res_type="soxr_vhq",scale=False)
                else:
                    x = self.audio_raw
            # Calcolo STFT
            f,t,stft = ss.stft(x,fs=self.audio_fs,window="hann",nperseg=win_length,noverlap=win_length-hop_length,nfft=n_fft)
            magnitude = np.abs(stft)
            magnitude[magnitude<1e-5] = 1e-5
            # Salvataggio nei dati app
            self.audio_fft = {"f":f[:np.int32(np.floor(stft.shape[0]/2))],
                            "t":t,
                            "stft":magnitude[:np.int32(np.floor(stft.shape[0]/2)),:]}
            # Plot
            self.subtbRPMAnalysis_FFT_figure.clf()
            self.subtbRPMAnalysis_FFT_axes = self.subtbRPMAnalysis_FFT_figure.add_subplot()
            self.subtbRPMAnalysis_FFT_axes.imshow(self.audio_fft["stft"],origin="lower",aspect="auto",cmap="terrain",extent=(self.audio_fft["t"][0],self.audio_fft["t"][-1],self.audio_fft["f"][0],self.audio_fft["f"][-1]))
            self.subtbRPMAnalysis_FFT_axes.set_xlabel("time [s]")
            self.subtbRPMAnalysis_FFT_axes.set_ylabel("frequency [Hz]")
            self.subtbRPMAnalysis_FFT_axes.grid()
            self.subtbRPMAnalysis_FFT_axes1 = self.subtbRPMAnalysis_FFT_axes.twinx()
            self.subtbRPMAnalysis_FFT_axes1.plot(self.audio_fft["t"],np.zeros_like(self.audio_fft["t"]),linewidth=0)
            self.subtbRPMAnalysis_FFT_axes1.set_ylabel("engine speed [rpm]")
            self.subtbRPMAnalysis_FFT_axes1.set_ylim(self.subtbRPMAnalysis_FFT_axes.get_ylim()[0]*10,self.subtbRPMAnalysis_FFT_axes.get_ylim()[1]*10)
            self.subtbRPMAnalysis_FFT_figure.tight_layout()
            self.subtbRPMAnalysis_FFT_canvas.draw_idle()
    
    def CancellaTuttoRPM(self):
        # Messagebox di conferma
        msg = QMessageBox()
        msg.setWindowTitle("AIrton")
        msg.setText("Do you want to clear the engine speed analysis?")
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setDefaultButton(QMessageBox.StandardButton.No)
        response = msg.exec()
        if response == QMessageBox.StandardButton.Yes:
            self.rpm = {"i":np.array([]),"t":np.array([]),"y":np.array([]),"yraw":np.array([])}
            if hasattr(self,"subtbRPMAnalysis_Analysis_axes"):
                [c.remove() for c in self.subtbRPMAnalysis_Analysis_axes.get_children() if c._label in ["predicted","corrected"]]
            if hasattr(self,"subtbRPMAnalysis_Analysis_axes1"):
                [c.remove() for c in self.subtbRPMAnalysis_Analysis_axes1.get_children() if c._label in ["predicted","corrected"]]
            self.subtbRPMAnalysis_Analysis_canvas.draw_idle()
            if hasattr(self,"subtbRPMAnalysis_Correction_axes"):
                [c.remove() for c in self.subtbRPMAnalysis_Correction_axes.get_children() if c._gid in ["giri"]]
            if hasattr(self,"subtbRPMAnalysis_Correction_axes1"):
                [c.remove() for c in self.subtbRPMAnalysis_Correction_axes1.get_children() if c._gid in ["giri"]]
            self.subtbRPMAnalysis_Correction_canvas.draw_idle()
    
    def AnalisiManualeRPM(self):
        # Definizione funzioni: tab
        def tastiera(evento):
            if evento.key == " ":
                if self.subtbRPMAnalysis_Correction_axes.get_xlabel() == "sample\n[RECTANGLE]":
                    self.subtbRPMAnalysis_Correction_axes.set_xlabel("sample\n[LASSO]")
                    RS.set_active(False)
                    LS.set_active(True)
                    self.subtbRPMAnalysis_Correction_figure.set_gid(-1)
                else:
                    self.subtbRPMAnalysis_Correction_axes.set_xlabel("sample\n[RECTANGLE]")
                    LS.set_active(False)
                    RS.set_active(True)
                    self.subtbRPMAnalysis_Correction_figure.set_gid(1)
                RS.clear()
                LS.clear()
                self.subtbRPMAnalysis_Correction_canvas.draw_idle()
            elif evento.key == "control":
                if self.subtbRPMAnalysis_Correction_axes.get_xlabel() == "sample\n[RECTANGLE]":
                    RS.set_active(True)
                    LS.set_active(False)
                else:
                    RS.set_active(False)
                    LS.set_active(True)
            elif evento.key == "escape":
                RS.clear()
                LS.clear()

        # Definizione funzioni: click destro
        def destro(evento):
            if evento.button == 3:
                # Vedo che strumento ha usato l'utente
                if (self.subtbRPMAnalysis_Correction_figure.get_gid() == -1):
                    # Lasso
                    p = [p for p in self.subtbRPMAnalysis_Correction_figure.axes[0].get_lines() if (p._color=="C0") & (len(p._x)>0)]
                    p = p[0]._path
                    xmin = np.floor(p.vertices[:,0].min()).astype(np.int16) - 1
                    xmax = np.ceil(p.vertices[:,0].max()).astype(np.int16) + 1
                    ymin = np.floor(p.vertices[:,1].min()).astype(np.int16) - 1
                    ymax = np.ceil(p.vertices[:,1].max()).astype(np.int16) + 1
                    tutti = np.array([(x,y) for x in np.arange(xmin,xmax,1) for y in np.arange(ymin,ymax,1)])
                    dentro = tutti[p.contains_points(tutti)]
                    unico_x = np.unique(dentro[:,0])
                    imax = np.zeros_like(unico_x)
                    cnt = 0
                    for u in unico_x:
                        tutti_y = dentro[np.where(dentro[:,0]==u)[0]][:,1]
                        imax[cnt] = self.audio_fft["stft"][tutti_y,u].argmax(axis=0) + tutti_y[0]
                        cnt += 1                     
                    if len(evento.modifiers) > 0:
                        fattore = re.findall("[\d{1}|\w{1}]$",evento.key)
                        tasto = re.findall("^(.*)\+",evento.key)
                        if (len(fattore) > 0) & (len(tasto) > 0):
                            fattore = fattore[0]
                            if fattore == "q":
                                fattore = 12/8
                            elif fattore == "w":
                                fattore = 12/9
                            elif fattore == "e":
                                fattore = 12/10
                            elif fattore == "r":
                                fattore = 12/11
                            elif fattore == "z":
                                fattore = 13/12
                            elif fattore == "x":
                                fattore = 14/12
                            elif fattore == "c":
                                fattore = 15/12
                            elif fattore == "v":
                                fattore = 16/12
                            else:
                                fattore = 12/int(fattore)
                            tasto = tasto[0]
                            if tasto == "ctrl":
                                # Moltiplico
                                imax = np.array(np.round(imax*fattore),dtype=imax.dtype)
                            elif tasto == "alt":
                                # Divido
                                imax = np.array(np.round(imax/fattore),dtype=imax.dtype)
                            elif tasto == "shift":
                                # Moltiplico per la radice quadrata
                                imax = np.array(np.round(imax*np.sqrt(fattore)),dtype=imax.dtype)
                            elif tasto == "ctrl+alt":
                                # Divido per la radice quadrata
                                imax = np.array(np.round(imax/np.sqrt(fattore)),dtype=imax.dtype)
                    if len([t.get_gid() for t in self.subtbRPMAnalysis_Correction_figure.axes[0].get_lines() if t.get_gid()=="giri"]) == 0:
                        self.subtbRPMAnalysis_Correction_figure.axes[0].plot(unico_x,imax,'.-r',gid="giri")
                    else:
                        linea = [t for t in self.subtbRPMAnalysis_Correction_figure.axes[0].get_lines() if t.get_gid()=="giri"][0]
                        nuovax = np.concatenate((linea.get_xdata(),unico_x))
                        nuovay = np.concatenate((linea.get_ydata(),imax))
                        nuovax,indici = np.unique(nuovax,return_index=True)
                        nuovay = nuovay[indici]
                        linea.set_xdata(nuovax)
                        linea.set_ydata(nuovay)
                elif (self.subtbRPMAnalysis_Correction_figure.get_gid() == 1):
                    # Rettangolo
                    vertici_x = np.array((np.max((0,np.floor(RS.corners[0].min()-1))),np.min((self.audio_fft["stft"].shape[1],np.ceil(RS.corners[0].max()+1)))),dtype="int")
                    vertici_y = np.array((np.max((0,np.floor(RS.corners[1].min()-1))),np.min((self.audio_fft["stft"].shape[0],np.ceil(RS.corners[1].max()+1)))),dtype="int")
                    imax = self.audio_fft["stft"][vertici_y[0]:vertici_y[1],vertici_x[0]:vertici_x[1]].argmax(axis=0) + vertici_y[0]
                    if len(evento.modifiers) > 0:
                        fattore = re.findall("[\d{1}|\w{1}]$",evento.key)
                        tasto = re.findall("^(.*)\+",evento.key)
                        if (len(fattore) > 0) & (len(tasto) > 0):
                            fattore = fattore[0]
                            if fattore == "q":
                                fattore = 12/8
                            elif fattore == "w":
                                fattore = 12/9
                            elif fattore == "e":
                                fattore = 12/10
                            elif fattore == "r":
                                fattore = 12/11
                            elif fattore == "z":
                                fattore = 13/12
                            elif fattore == "x":
                                fattore = 14/12
                            elif fattore == "c":
                                fattore = 15/12
                            elif fattore == "v":
                                fattore = 16/12
                            else:
                                fattore = 12/int(fattore)
                            tasto = tasto[0]
                            if tasto == "ctrl":
                                # Moltiplico
                                imax = np.array(np.round(imax*fattore),dtype=imax.dtype)
                            elif tasto == "alt":
                                # Divido
                                imax = np.array(np.round(imax/fattore),dtype=imax.dtype)
                            elif tasto == "shift":
                                # Moltiplico per la radice quadrata
                                imax = np.array(np.round(imax*np.sqrt(fattore)),dtype=imax.dtype)
                            elif tasto == "ctrl+alt":
                                # Divido per la radice quadrata
                                imax = np.array(np.round(imax/np.sqrt(fattore)),dtype=imax.dtype)
                    if len([t.get_gid() for t in self.subtbRPMAnalysis_Correction_figure.axes[0].get_lines() if t.get_gid()=="giri"]) == 0:
                        self.subtbRPMAnalysis_Correction_figure.axes[0].plot(np.linspace(vertici_x[0],vertici_x[1]-1,imax.shape[0]),imax,'.-r',gid="giri")
                    else:
                        linea = [t for t in self.subtbRPMAnalysis_Correction_figure.axes[0].get_lines() if t.get_gid()=="giri"][0]
                        nuovax = np.concatenate((linea.get_xdata(),np.linspace(vertici_x[0],vertici_x[1]-1,imax.shape[0])))
                        nuovay = np.concatenate((linea.get_ydata(),imax))
                        nuovax,indici = np.unique(nuovax,return_index=True)
                        nuovay = nuovay[indici]
                        linea.set_xdata(nuovax)
                        linea.set_ydata(nuovay)
                RS.clear()
                LS.clear()
                evento.canvas.draw_idle()
            elif evento.button == 2:
                # Controllo che non ci siano punti dentro al rettangolo: nel caso, li cancello
                linea = [t for t in self.subtbRPMAnalysis_Correction_figure.axes[0].get_lines() if t.get_gid()=="giri"]
                if len(linea) > 0:
                    linea = linea[0]
                    # indici_linea = np.array([i for i in range(len(linea._x)) if (linea._x[i]>=np.floor(RS.corners[0].min())) & (linea._x[i]<=np.ceil(RS.corners[0].max())) & (linea._y[i]>=np.floor(RS.corners[1].min())) & (linea._y[i]<=np.ceil(RS.corners[1].max()))])
                    # indici_linea = np.array([i for i in np.arange(int(np.floor(RS.corners[0][0])-25),int(np.floor(RS.corners[0][1])+25)) if (linea._x[i]>=np.floor(RS.corners[0].min())) & (linea._x[i]<=np.ceil(RS.corners[0].max())) & (linea._y[i]>=np.floor(RS.corners[1].min())) & (linea._y[i]<=np.ceil(RS.corners[1].max()))])
                    indici_x = np.array([ix for ix in range(len(linea._x)) if linea._x[ix] in np.arange(int(np.floor(RS.corners[0][0])-5),int(np.floor(RS.corners[0][1])+5))],dtype="int")
                    indici_y = np.array([iy for iy in indici_x if linea._y[iy] in np.arange(int(np.floor(RS.corners[1][0])-5),int(np.floor(RS.corners[1][2])+5))])
                    if len(indici_y) > 0:
                        nuovax = np.delete(linea._x,indici_y)
                        nuovay = np.delete(linea._y,indici_y)
                        linea.set_data(nuovax,nuovay)
                        evento.canvas.draw_idle()
        # Definizione funzioni: dummy per rettangolo
        def dummy(eclick, erelease):
            # Disegno il rettangolo
            # patches.Rectangle((np.max((0,np.floor(RS.corners[0].min()-1))),np.max((0,np.floor(RS.corners[1].min()-1)))),int(np.ceil(RS.corners[1][2]-RS.corners[1][0])),
            #                 int(np.ceil(RS.corners[0][1]-RS.corners[0][0])),linewidth=1,alpha=0.5,facecolor="green",gid="suca")
            eclick.canvas.figure.set_gid(1)
        # Definizione funzioni: dummy per lazo
        def dummy_lasso(eclick,verts):
            # Disegno il lasso
            # path = Path(verts)
            eclick.canvas.figure.set_gid(-1)
        # Callback figura
        def attivaLS(evento):
            if evento.key == "control":
                LS.set_active(True)
                RS.set_active(False)
        def disattivaLS(evento):
            if evento.key == "control":
                LS.set_active(False)
                RS.set_active(True)
            elif evento.key == "escape":
                RS.clear()
                LS.clear()

        if self.sender().text() == "Run manual analysis":
            # Correzione manuale giri su plot: attivo la tab corrispondente
            self.subtbRPMAnalysis.setCurrentIndex(2)
            # Plotto i giri e definisco i callback per interagire con il grafico
            self.subtbRPMAnalysis_Correction_figure.clf()
            self.subtbRPMAnalysis_Correction_figure.set_gid(1)
            self.subtbRPMAnalysis_Correction_axes = self.subtbRPMAnalysis_Correction_figure.add_subplot()
            self.subtbRPMAnalysis_Correction_axes.set_gid(0)
            self.subtbRPMAnalysis_Correction_axes1 = self.subtbRPMAnalysis_Correction_axes.twinx()
            self.subtbRPMAnalysis_Correction_axes1.plot(np.array([self.audio_fft["t"][0],self.audio_fft["t"][-1]]),np.array([10*self.audio_fft["f"][0],10*self.audio_fft["f"][-1]]),linewidth=0)
            self.subtbRPMAnalysis_Correction_axes1.set_ylim((10*self.audio_fft["f"][0],10*self.audio_fft["f"][-1]))
            self.subtbRPMAnalysis_Correction_axes1.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False)
            self.subtbRPMAnalysis_Correction_axes.imshow(self.audio_fft["stft"],origin="lower",aspect="auto",cmap="terrain")
            if self.rpm["i"].shape[0] > 0:
                self.subtbRPMAnalysis_Correction_axes.plot(self.rpm["i"],self.rpm["y"],'.-r',gid="giri")
            self.subtbRPMAnalysis_Correction_axes.grid()
            self.subtbRPMAnalysis_Correction_axes.set_xlabel("sample\n[RECTANGLE]")
            self.subtbRPMAnalysis_Correction_axes.set_ylabel("frequency [Hz]")
            self.subtbRPMAnalysis_Correction_axes1.set_ylabel("engine speed [rpm]")
            self.subtbRPMAnalysis_Correction_axes.set_title("Q=8° (1.5)    -    W=9° (1.33)    -    E=10° (1.2)    -    R=11° (1.09)    -    Z=13° (1.08)    -    X=14° (1.16)    -    C=15° (1.25)    -    V=16° (1.33)")
            RS = RectangleSelector(self.subtbRPMAnalysis_Correction_axes, dummy, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=False)
            LS = LassoSelector(self.subtbRPMAnalysis_Correction_axes, onselect=dummy_lasso, useblit=True, button=[1])
            LS.active = False
            # Associazione callback
            # self.subtbRPMAnalysis_Correction_figure.canvas.mpl_connect("key_release_event",disattivaLS)
            # self.subtbRPMAnalysis_Correction_figure.canvas.mpl_connect("key_press_event",attivaLS)
            self.subtbRPMAnalysis_Correction_figure.canvas.mpl_connect("key_press_event",tastiera)
            self.subtbRPMAnalysis_Correction_figure.canvas.mpl_connect("button_release_event",destro)
            self.subtbRPMAnalysis_Correction_canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
            self.subtbRPMAnalysis_Correction_canvas.setFocus()
            self.subtbRPMAnalysis_Correction_figure.tight_layout()
            self.subtbRPMAnalysis_Correction_canvas.draw_idle()
        elif self.sender().text() == "Accept manual reconstruction":
            # Salvo i giri ricostruiti nella struttura dell'app
            linea = [t for t in self.subtbRPMAnalysis_Correction_figure.axes[0].get_lines() if t.get_gid()=="giri"]
            if len(linea) > 0:
                self.rpm["i"] = linea[0]._x.astype(np.int64)
                self.rpm["t"] = self.audio_fft["t"][self.rpm["i"]]
                self.rpm["y"] = linea[0]._y.astype(np.int16)
                self.rpm["yraw"] = linea[0]._y.astype(np.int16)

    def AnalisiAutomaticaRPM(self):
        # Detection automatica dei giri tramite armoniche o tramite rete neurale
        if self.audio_fft["stft"].shape[1] > 0:
            if self.sender().__class__ == QPushButton:
                # Attivo la tab corrispondente
                self.subtbRPMAnalysis.setCurrentIndex(1)
                if self.tbRPMAnalysis_analysisType.currentIndex() == 0:
                    # Standard: ricerco i massimi sulle armoniche impostate
                    armoniche = np.array([np.int8(d) for d in re.findall("\d",self.tbRPMAnalysis_baseHarmonics.text())])
                    if armoniche.shape[0] == 0:
                        armoniche = np.array([4])
                    rpmmax = np.float32(self.tbRPMAnalysis_maxRPM.text())
                    rpmmin = np.float32(self.tbRPMAnalysis_minRPM.text())
                    Y_predict = np.zeros((armoniche.shape[0],self.audio_fft["stft"].shape[1]))
                    for it in range(Y_predict.shape[1]):
                        for ia in range(Y_predict.shape[0]):
                            i0 = np.where(self.audio_fft["f"]>=rpmmin*armoniche[ia]/120)[0][0]
                            i1 = np.where(self.audio_fft["f"]>=rpmmax*armoniche[ia]/120)[0][0]
                            Y_predict[ia,it] = (np.argmax(self.audio_fft["stft"][i0:i1,it]) + i0 - 1)*12/armoniche[ia]
                    Y_predict = np.median(Y_predict,axis=0).astype(np.int16)
                    # Applico il moltiplicatore manuale
                    multi = np.float32(self.tbRPMAnalysis_Multiplier.text())
                    if (multi is None) | (np.isnan(multi)):
                        multi = 1
                    Y_predict = (Y_predict*multi).astype(np.int16)
                elif self.tbRPMAnalysis_analysisType.currentIndex() == 1:
                    # Rete neurale: carico il modello
                    nomerete = self.tbRPMAnalysis_NNmodel.currentText()
                    if nomerete == "F1_DeepSpeech_5_torch":
                        media = 541.25
                        devst = 63.71
                    elif nomerete == "F1_DeepSpeech_9_torch":
                        media = 538.91
                        devst = 65.11
                    if torch.cuda.is_available():
                        mymodel = torch.load(self.pwd+nomerete,weights_only=False,map_location=torch.device("cuda"))
                    else:
                        mymodel = torch.load(self.pwd+nomerete,weights_only=False,map_location=torch.device("cpu"))
                    mymodel.eval()
                    # Preprocessing dello spettro
                    magnitude = self.audio_fft["stft"]
                    magnitude = (magnitude-magnitude.mean())/magnitude.std()
                    X = np.expand_dims(magnitude.T,1)
                    # Controllo che la dimensione dello spettro sia compatibile con il primo layer della rete
                    if (X.shape[1] != 1) | (X.shape[2] != 1024):
                        dlg = QMessageBox(self)
                        dlg.setWindowTitle("AIrton")
                        dlg.setText("Current FFT configuration does not match with the input shape required by the neural network.\nIf you continue, input array will be reshaped accordingly. Do you want to proceed?")
                        dlg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                        dlg.setIcon(QMessageBox.Icon.Critical)
                        button = dlg.exec()
                        if button == 65536:
                            return
                        elif button == 16384:
                            if X.shape[-1] > 1024:
                                X = X[:,:,:1024]
                            else:
                                X = np.concat((X,np.zeros((X.shape[0],X.shape[1],1024-X.shape[-1]))),axis=2)
                    # Previsione della rete
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            Y_predict = mymodel(torch.Tensor(X).to("cuda")).cpu().numpy()[:,0]
                        else:
                            Y_predict = mymodel(torch.Tensor(X)).numpy()[:,0]
                        Y_predict = (Y_predict*devst + media).astype(np.int16)
                # Sistemo tutte le frequenze utilizzando la ricerca manuale del massimo per ogni punto previsto (dalla rete o dal metodo manuale)
                if self.tbRPMAnalysis_Correction.currentIndex() > 0:
                    ordini = np.array([12/14,12/13,12/9,  2,  3,  4])
                    Y = np.zeros(Y_predict.shape,dtype=np.int16)
                    for iy in range(Y_predict.shape[0]):
                        # Cerco il massimo sull'ordine della ricostruzione
                        ir = np.argmax(self.audio_fft["stft"][Y_predict[iy]-15:Y_predict[iy]+15,iy]) + Y_predict[iy] - 15
                        # Cerco il massimo su tutti gli ordini disponibili
                        i0 = np.zeros((ordini.shape[0],))
                        pesi = np.zeros((ordini.shape[0],))
                        for io in range(ordini.shape[0]):
                            if int(Y_predict[iy]/ordini[io]) < (self.audio_fft["stft"].shape[0]-16):
                                i0[io] = np.argmax(self.audio_fft["stft"][int(Y_predict[iy]/ordini[io])-15:int(Y_predict[iy]/ordini[io])+15,iy]) + int(Y_predict[iy]/ordini[io]) - 15
                                if i0[io]*ordini[io] > ir:
                                    pesi[io] = np.max([np.min([1-(i0[io]*ordini[io]-ir)/15,1]),0])
                                elif i0[io]*ordini[io] < ir:
                                    pesi[io] = np.max([np.min([1-(ir-i0[io]*ordini[io])/15,1]),0])
                                else:
                                    pesi[io] = 1
                        # Calcolo la frequenza finale
                        if self.tbRPMAnalysis_Correction.currentIndex() == 1:
                            # Media pesata
                            Y[iy] = np.round(np.average(np.hstack((ir,i0*ordini)),weights=np.hstack((2,pesi)))).astype(np.int16)
                        elif self.tbRPMAnalysis_Correction.currentIndex() == 2:
                            # Mediana
                            Y[iy] = (np.median(i0[i0>0]*ordini[i0>0])).astype(np.int16)
                else:
                        Y = Y_predict
                # Filtro i giri ricalcolati
                # ["No filter","Rolling average","Median filter","Lowpass filter (light)","Lowpass filter (medium)","Lowpass filter (strong)"]
                if self.tbRPMAnalysis_Filter.currentIndex() > 0:
                    if self.tbRPMAnalysis_Filter.currentIndex() == 1:
                        # Media mobile
                        window_size = np.int16(self.tbRPMAnalysis_KernelSize.text())
                        if (window_size < 0) | (window_size is None) | (window_size == np.nan):
                            window_size = 11
                        Y = np.convolve(Y, np.ones(window_size) / window_size, mode='valid').astype(np.int16)
                    if self.tbRPMAnalysis_Filter.currentIndex() == 2:
                        # Mediana mobile
                        window_size = np.int16(self.tbRPMAnalysis_KernelSize.text())
                        if (window_size < 0) | (window_size is None) | (window_size == np.nan):
                            window_size = 11
                        Y = ss.medfilt(Y,window_size).astype(np.int16)
                    elif self.tbRPMAnalysis_Filter.currentIndex() == 3:
                        # Lowpass
                        fc = np.float32(self.tbRPMAnalysis_FilterFrequency.text())
                        if (fc < 0) | (fc > 1) | (fc is None) | (fc == np.nan):
                            fc = 0.1
                            self.tbRPMAnalysis_FilterFrequency.setText("0.1")
                        b,a = ss.butter(1,fc)
                        Y = ss.filtfilt(b,a,Y).astype(np.int16)
                # Salvataggio dati nella struttura app
                self.rpm["t"] = self.audio_fft["t"]
                self.rpm["i"] = np.arange(Y.shape[0])
                self.rpm["y"] = Y
                self.rpm["yraw"] = Y_predict
                # Plot
                self.subtbRPMAnalysis_Analysis_figure.clf()
                self.subtbRPMAnalysis_Analysis_axes = self.subtbRPMAnalysis_Analysis_figure.add_subplot()
                self.subtbRPMAnalysis_Analysis_axes.imshow(self.audio_fft["stft"],origin="lower",aspect="auto",cmap="terrain",extent=(self.audio_fft["t"][0],self.audio_fft["t"][-1],self.audio_fft["f"][0],self.audio_fft["f"][-1]))
                self.subtbRPMAnalysis_Analysis_axes.plot(self.audio_fft["t"],self.audio_fft["f"][Y_predict],'.w',label="predicted")
                self.subtbRPMAnalysis_Analysis_axes.plot(self.audio_fft["t"],self.audio_fft["f"][Y],'.-r',label="corrected")
                self.subtbRPMAnalysis_Analysis_axes.legend()
                self.subtbRPMAnalysis_Analysis_axes.set_xlabel("time [s]")
                self.subtbRPMAnalysis_Analysis_axes.set_ylabel("frequency [Hz]")
                self.subtbRPMAnalysis_Analysis_axes.grid()
                self.subtbRPMAnalysis_Analysis_axes1 = self.subtbRPMAnalysis_Analysis_axes.twinx()
                self.subtbRPMAnalysis_Analysis_axes1.plot(self.audio_fft["t"],10*self.audio_fft["f"][Y],linewidth=0)
                self.subtbRPMAnalysis_Analysis_axes1.set_ylabel("engine speed [rpm]")
                self.subtbRPMAnalysis_Analysis_axes1.set_ylim(self.subtbRPMAnalysis_Analysis_axes.get_ylim()[0]*10,self.subtbRPMAnalysis_Analysis_axes.get_ylim()[1]*10)
                self.subtbRPMAnalysis_Analysis_figure.tight_layout()
                self.subtbRPMAnalysis_Analysis_canvas.draw_idle()
            else:
                # Controllo se i giri sono già stati calcolati: ricalcolo i giri corretti partendo dai giri previsti
                if self.rpm["yraw"].shape[0] > 0:
                    Y_predict = self.rpm["yraw"]
                    # Sistemo tutte le frequenze utilizzando la ricerca manuale del massimo per ogni punto previsto (dalla rete o dal metodo manuale)
                    if self.tbRPMAnalysis_Correction.currentIndex() > 0:
                        ordini = np.array([12/14,12/13,12/9,  2,  3,  4])
                        Y = np.zeros(Y_predict.shape,dtype=np.int16)
                        for iy in range(Y_predict.shape[0]):
                            # Cerco il massimo sull'ordine della ricostruzione
                            ir = np.argmax(self.audio_fft["stft"][Y_predict[iy]-15:Y_predict[iy]+15,iy]) + Y_predict[iy] - 15
                            # Cerco il massimo su tutti gli ordini disponibili
                            i0 = np.zeros((ordini.shape[0],))
                            pesi = np.zeros((ordini.shape[0],))
                            for io in range(ordini.shape[0]):
                                if int(Y_predict[iy]/ordini[io]) < (self.audio_fft["stft"].shape[0]-16):
                                    i0[io] = np.argmax(self.audio_fft["stft"][int(Y_predict[iy]/ordini[io])-15:int(Y_predict[iy]/ordini[io])+15,iy]) + int(Y_predict[iy]/ordini[io]) - 15
                                    if i0[io]*ordini[io] > ir:
                                        pesi[io] = np.max([np.min([1-(i0[io]*ordini[io]-ir)/15,1]),0])
                                    elif i0[io]*ordini[io] < ir:
                                        pesi[io] = np.max([np.min([1-(ir-i0[io]*ordini[io])/15,1]),0])
                                    else:
                                        pesi[io] = 1
                            # Calcolo la frequenza finale
                            if self.tbRPMAnalysis_Correction.currentIndex() == 1:
                                # Media pesata
                                Y[iy] = np.round(np.average(np.hstack((ir,i0*ordini)),weights=np.hstack((2,pesi)))).astype(np.int16)
                            elif self.tbRPMAnalysis_Correction.currentIndex() == 2:
                                # Mediana
                                Y[iy] = (np.median(i0[i0>0]*ordini[i0>0])).astype(np.int16)
                    else:
                            Y = Y_predict
                    # Filtro i giri ricalcolati
                    # ["No filter","Rolling average","Median filter","Lowpass filter (light)","Lowpass filter (medium)","Lowpass filter (strong)"]
                    if self.tbRPMAnalysis_Filter.currentIndex() > 0:
                        if self.tbRPMAnalysis_Filter.currentIndex() == 1:
                            # Media mobile
                            window_size = np.int16(self.tbRPMAnalysis_KernelSize.text())
                            if (window_size < 0) | (window_size is None) | (window_size == np.nan):
                                window_size = 11
                            Y = np.convolve(Y, np.ones(window_size) / window_size, mode='valid').astype(np.int16)
                        if self.tbRPMAnalysis_Filter.currentIndex() == 2:
                            # Mediana mobile
                            window_size = np.int16(self.tbRPMAnalysis_KernelSize.text())
                            if (window_size < 0) | (window_size is None) | (window_size == np.nan):
                                window_size = 11
                            Y = ss.medfilt(Y,window_size).astype(np.int16)
                        elif self.tbRPMAnalysis_Filter.currentIndex() == 3:
                            # Lowpass
                            fc = np.float32(self.tbRPMAnalysis_FilterFrequency.text())
                            if (fc < 0) | (fc >= 1) | (fc is None) | (fc == np.nan):
                                fc = 0.1
                                self.tbRPMAnalysis_FilterFrequency.setText("0.1")
                            b,a = ss.butter(1,fc)
                            Y = ss.filtfilt(b,a,Y).astype(np.int16)
                    # Salvataggio dati nella struttura app
                    self.rpm["y"] = Y
                    # Plot
                    self.subtbRPMAnalysis_Analysis_figure.clf()
                    self.subtbRPMAnalysis_Analysis_axes = self.subtbRPMAnalysis_Analysis_figure.add_subplot()
                    self.subtbRPMAnalysis_Analysis_axes.imshow(self.audio_fft["stft"],origin="lower",aspect="auto",cmap="terrain",extent=(self.audio_fft["t"][0],self.audio_fft["t"][-1],self.audio_fft["f"][0],self.audio_fft["f"][-1]))
                    self.subtbRPMAnalysis_Analysis_axes.plot(self.audio_fft["t"],self.audio_fft["f"][Y_predict],'.w',label="predicted")
                    self.subtbRPMAnalysis_Analysis_axes.plot(self.audio_fft["t"],self.audio_fft["f"][Y],'.-r',label="corrected")
                    self.subtbRPMAnalysis_Analysis_axes.legend()
                    self.subtbRPMAnalysis_Analysis_axes.set_xlabel("time [s]")
                    self.subtbRPMAnalysis_Analysis_axes.set_ylabel("frequency [Hz]")
                    self.subtbRPMAnalysis_Analysis_axes.grid()
                    self.subtbRPMAnalysis_Analysis_axes1 = self.subtbRPMAnalysis_Analysis_axes.twinx()
                    self.subtbRPMAnalysis_Analysis_axes1.plot(self.audio_fft["t"],10*self.audio_fft["f"][Y],linewidth=0)
                    self.subtbRPMAnalysis_Analysis_axes1.set_ylabel("engine speed [rpm]")
                    self.subtbRPMAnalysis_Analysis_axes1.set_ylim(self.subtbRPMAnalysis_Analysis_axes.get_ylim()[0]*10,self.subtbRPMAnalysis_Analysis_axes.get_ylim()[1]*10)
                    self.subtbRPMAnalysis_Analysis_figure.tight_layout()
                    self.subtbRPMAnalysis_Analysis_canvas.draw_idle()
            
    def salvaRPM(self):
        # File dialog
        nome = QFileDialog.getSaveFileName(self,"Save engine speed data","/home","Python pickle data (*.pickle)")
        if nome[0] != "":
            with open(nome[0]+".pickle", "ab") as f:
                try:
                    pickle.dump([self.audio_fft,self.rpm],f,pickle.HIGHEST_PROTOCOL)
                    msgBox = QMessageBox()
                    msgBox.setIcon(QMessageBox.Icon.Information)
                    msgBox.setText("Data saved successfully.")
                    msgBox.setWindowTitle("AIrton")
                    msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
                    msgBox.exec()
                except:
                    msgBox = QMessageBox()
                    msgBox.setIcon(QMessageBox.Icon.Critical)
                    msgBox.setText("Error occurred during saving process.")
                    msgBox.setWindowTitle("AIrton")
                    msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
                    msgBox.exec()
    
    def caricaRPM(self):
        # File dialog
        nome = QFileDialog.getOpenFileName(self,"Load engine speed data","/home","Python pickle data (*.pickle)")
        if nome[0] != "":
            with open(nome[0], "rb") as f:
                try:
                    dati = pickle.load(f)
                    if dati is None:
                        # Dati KO
                        msgBox = QMessageBox()
                        msgBox.setIcon(QMessageBox.Icon.Critical)
                        msgBox.setText("No data was loaded. Selected file is empty or not valid.")
                        msgBox.setWindowTitle("AIrton")
                        msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
                        msgBox.exec()
                    else:
                        if len(dati) < 2:
                            # Dati KO
                            msgBox = QMessageBox()
                            msgBox.setIcon(QMessageBox.Icon.Critical)
                            msgBox.setText("No data was loaded. Selected file is empty or not valid.")
                            msgBox.setWindowTitle("AIrton")
                            msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
                            msgBox.exec()
                        else:
                            # Dati OK
                            self.audio_fft = dati[0]
                            self.rpm = dati[1]
                            # Plot FFT
                            self.subtbRPMAnalysis_FFT_figure.clf()
                            self.subtbRPMAnalysis_FFT_axes = self.subtbRPMAnalysis_FFT_figure.add_subplot()
                            self.subtbRPMAnalysis_FFT_axes.imshow(self.audio_fft["stft"],origin="lower",aspect="auto",cmap="terrain",extent=(self.audio_fft["t"][0],self.audio_fft["t"][-1],self.audio_fft["f"][0],self.audio_fft["f"][-1]))
                            self.subtbRPMAnalysis_FFT_axes.set_xlabel("time [s]")
                            self.subtbRPMAnalysis_FFT_axes.set_ylabel("frequency [Hz]")
                            self.subtbRPMAnalysis_FFT_axes.grid()
                            self.subtbRPMAnalysis_FFT_axes1 = self.subtbRPMAnalysis_FFT_axes.twinx()
                            self.subtbRPMAnalysis_FFT_axes1.plot(self.audio_fft["t"],np.zeros_like(self.audio_fft["t"]),linewidth=0)
                            self.subtbRPMAnalysis_FFT_axes1.set_ylabel("engine speed [rpm]")
                            self.subtbRPMAnalysis_FFT_axes1.set_ylim(self.subtbRPMAnalysis_FFT_axes.get_ylim()[0]*10,self.subtbRPMAnalysis_FFT_axes.get_ylim()[1]*10)
                            self.subtbRPMAnalysis_FFT_figure.tight_layout()
                            self.subtbRPMAnalysis_FFT_canvas.draw_idle()
                            # Plot giri
                            self.subtbRPMAnalysis_Analysis_figure.clf()
                            self.subtbRPMAnalysis_Analysis_axes = self.subtbRPMAnalysis_Analysis_figure.add_subplot()
                            self.subtbRPMAnalysis_Analysis_axes.imshow(self.audio_fft["stft"],origin="lower",aspect="auto",cmap="terrain",extent=(self.audio_fft["t"][0],self.audio_fft["t"][-1],self.audio_fft["f"][0],self.audio_fft["f"][-1]))
                            self.subtbRPMAnalysis_Analysis_axes.plot(self.audio_fft["t"],self.audio_fft["f"][self.rpm["yraw"]],'.w',label="predicted")
                            self.subtbRPMAnalysis_Analysis_axes.plot(self.audio_fft["t"],self.audio_fft["f"][self.rpm["y"]],'.-r',label="corrected")
                            self.subtbRPMAnalysis_Analysis_axes.legend()
                            self.subtbRPMAnalysis_Analysis_axes.set_xlabel("time [s]")
                            self.subtbRPMAnalysis_Analysis_axes.set_ylabel("frequency [Hz]")
                            self.subtbRPMAnalysis_Analysis_axes.grid()
                            self.subtbRPMAnalysis_Analysis_axes1 = self.subtbRPMAnalysis_Analysis_axes.twinx()
                            self.subtbRPMAnalysis_Analysis_axes1.plot(self.audio_fft["t"],10*self.audio_fft["f"][self.rpm["y"]],linewidth=0)
                            self.subtbRPMAnalysis_Analysis_axes1.set_ylabel("engine speed [rpm]")
                            self.subtbRPMAnalysis_Analysis_axes1.set_ylim(self.subtbRPMAnalysis_Analysis_axes.get_ylim()[0]*10,self.subtbRPMAnalysis_Analysis_axes.get_ylim()[1]*10)
                            self.subtbRPMAnalysis_Analysis_figure.tight_layout()
                            self.subtbRPMAnalysis_Analysis_canvas.draw_idle()
                            msgBox = QMessageBox()
                            msgBox.setIcon(QMessageBox.Icon.Information)
                            msgBox.setText("Data loaded successfully.")
                            msgBox.setWindowTitle("AIrton")
                            msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
                            msgBox.exec()
                except:
                    msgBox = QMessageBox()
                    msgBox.setIcon(QMessageBox.Icon.Critical)
                    msgBox.setText("Error occurred during loading process.")
                    msgBox.setWindowTitle("AIrton")
                    msgBox.setStandardButtons(QMessageBox.StandardButton.Ok)
                    msgBox.exec()

    def CercaCambiate(self):
        if np.shape(self.rpm["y"])[0] > 0:
            if self.sender().text().find("Search") >= 0:
                if self.sender().text().find("upshifts") >= 0:
                    # ANALISI UPSHIFT
                    # Pulisco la struttura gear
                    self.gear["iup"] = np.array([])
                    self.gear["dmarciaup"] = np.array([])
                    self.gear["dgiriup"] = np.array([])
                    # Frequenza di filtraggio dei giri motore
                    fcut = np.float32(self.tbGearAnalysis_LPupFrequency.text())
                    if (fcut is None) | (np.isnan(fcut)) | (fcut <= 0) | (fcut >= 1):
                        fcut = 0.25
                    b,a = ss.butter(1,fcut)
                    # Soglia annullamento derivata
                    zeroth = np.int16(self.tbGearAnalysis_dRPM0upThreshold.text())
                    if (zeroth is None) | (np.isnan(zeroth)):
                        zeroth = np.int16(50)
                    # Soglia derivata per individuazione cambiate
                    upth = np.int16(self.tbGearAnalysis_dRPMupThreshold.text())
                    if (upth is None) | (np.isnan(upth)) | (upth == 0):
                        upth = np.int16(150)
                    if upth < 0:
                        upth = -upth
                    # Filtro i giri motore
                    fgirif = ss.filtfilt(b,a,self.rpm["y"]).astype("int")
                    girif = 10*self.audio_fft["f"][fgirif]
                    # Calcolo la derivata dei giri
                    dgiri = np.hstack((0,np.diff(girif)))
                    dgiri[np.abs(dgiri)<zeroth] = 0
                    self.gear["dgiriup"] = dgiri
                    # Inizializzo la struttura degli indici degli upshift
                    iup_all = np.where((dgiri[:-1]>-upth) & (dgiri[1:]<-upth))[0]
                    iup = np.zeros((iup_all.shape[0],2),dtype="int")
                    cnt = 0
                    for i in iup_all:
                        if cnt > 0:
                            if ((np.argmin(girif[i-10:i+50]) + i - 10) - iup[cnt-1,1]) < 10:
                                print("Doppione UPSHIFT")
                            else:
                                iup[cnt,0] = np.argmax(girif[i-10:i+50]) + i - 10
                                iup[cnt,1] = np.argmin(girif[i-10:i+50]) + i - 10
                                cnt += 1
                        else:
                            iup[cnt,0] = np.argmax(girif[i-10:i+50]) + i - 10
                            iup[cnt,1] = np.argmin(girif[i-10:i+50]) + i - 10
                            cnt += 1
                    if np.where(iup[:,1]==0)[0].shape[0] > 0:
                        iup = iup[:np.where(iup[:,1]==0)[0][0],:]
                    self.gear["iup"] = iup
                    self.gear["dmarciaup"] = np.ones((iup.shape[0],),dtype="int")
                elif self.sender().text().find("downshifts") >= 0:
                    # ANALISI DOWNSHIFT
                    # Pulisco la struttura gear
                    self.gear["idown"] = np.array([])
                    self.gear["dmarciadown"] = np.array([])
                    self.gear["dgiridown"] = np.array([])
                    # Frequenza di filtraggio dei giri motore
                    fcut = np.float32(self.tbGearAnalysis_LPdownFrequency.text())
                    if (fcut is None) | (np.isnan(fcut)) | (fcut <= 0) | (fcut >= 1):
                        fcut = 0.5
                    b,a = ss.butter(1,fcut)
                    # Filtro i giri motore
                    fgirif = ss.filtfilt(b,a,self.rpm["y"]).astype("int")
                    girif = 10*self.audio_fft["f"][fgirif]
                    # Calcolo la derivata dei giri
                    dgiri = np.hstack((0,np.diff(girif)))
                    # Filtro con mediana mobile
                    kernel = np.int16(self.tbGearAnalysis_MedFiltKernel.text())
                    if (kernel is None) | (np.isnan(kernel)):
                        kernel = np.int16(11)
                    if kernel > 0:
                        if np.mod(kernel,2) == 0:
                            kernel += 1
                        dgirif = ss.medfilt(dgiri,11)
                    else:
                        dgirif = np.copy(dgiri)
                    # Filtro la mediana mobile con passa basso
                    fcut = np.float32(self.tbGearAnalysis_LPFiltFrequency.text())
                    if (fcut is None) | (np.isnan(fcut)) | (fcut <= 0) | (fcut >= 1):
                        fcut = 0.25
                    b,a = ss.butter(1,fcut)
                    dgiriff = ss.filtfilt(b,a,dgirif)
                    self.gear["dgiridown"] = dgiri
                    # Prendo dalla GUI i parametri per la ricerca delle scalate (finestra ricerca minimo)
                    finestraminimo = self.tbGearAnalysis_MinSearchWindow.text()
                    finestraminimo = re.findall("(\d+),(\d+)",finestraminimo)[0]
                    if len(finestraminimo) > 0:
                        finestraminimo = np.array([np.uint8(f) if (np.int8(f)>0) else 1 for f in finestraminimo])
                    else:
                        finestraminimo = np.array([5,2],dtype=np.uint8)
                    # Prendo dalla GUI i parametri per la ricerca delle scalate (finestra ricerca massimo)
                    finestramassimo = self.tbGearAnalysis_MaxSearchWindow.text()
                    finestramassimo = re.findall("(\d+),(\d+)",finestramassimo)[0]
                    if len(finestramassimo) > 0:
                        finestramassimo = np.array([np.uint8(f) if (np.int8(f)>0) else 1 for f in finestramassimo])
                    else:
                        finestramassimo = np.array([0,8],dtype=np.uint8)
                    # Prendo dalla GUI i parametri per la ricerca delle scalate (finestra ricerca massimo 2)
                    finestramassimo2 = self.tbGearAnalysis_Max2SearchWindow.text()
                    finestramassimo2 = re.findall("(\d+),(\d+)",finestramassimo2)[0]
                    if len(finestramassimo2) > 0:
                        finestramassimo2 = np.array([np.uint8(f) if (np.int8(f)>0) else 1 for f in finestramassimo2])
                    else:
                        finestramassimo2 = np.array([0,3],dtype=np.uint8)
                    # Cerco i punti in cui la derivata cambia segno
                    izerocross = np.where((dgiriff[:-1]<0) & (dgiriff[1:]>0))[0]
                    if izerocross[0] == 0:
                        izerocross = izerocross[1:]
                    # Basando la ricerca sui punti di 0-cross, vedo qual è il gradiente massimo della cambiata per eliminare i falsi downshift
                    idown = np.zeros((izerocross.shape[0],2),dtype="int")
                    cnt = 0
                    for i in izerocross:
                        igmin = np.max([np.argmin(girif[np.max([0,i-finestraminimo[0]]):np.min([girif.shape[0]-1,i+finestraminimo[1]])])+np.max([0,i-finestraminimo[0]]),0])
                        gmin = girif[igmin]
                        igmax = np.min([np.argmax(girif[np.max([0,i-finestramassimo[0]]):np.min([girif.shape[0]-1,i+finestramassimo[1]])])+np.max([0,i-finestramassimo[0]]),girif.shape[0]-1])
                        gmax = girif[igmax]
                        gmaxmax = girif[np.min([girif.shape[0]-1,igmax+finestramassimo2[1]])]
                        if ((gmax-gmin) > 220) & (gmaxmax < gmax):
                            # Si tratta effettivamente di un downshift: lo salvo
                            idown[cnt,:] = np.array([igmin,igmax])
                            cnt += 1
                    if np.where(idown[:,0]>0)[0].shape[0] > 0:
                        idown = idown[:np.where(idown[:,0]>0)[0][-1]+1,:]
                    self.gear["idown"] = idown
                    self.gear["dmarciadown"] = -1*np.ones((idown.shape[0],),dtype="int")
            elif self.sender().text().find("Clear") >= 0:
                if self.sender().text().find("upshifts") >= 0:
                    self.gear["iup"] = np.array([])
                    self.gear["dmarciaup"] = np.array([])
                    self.gear["dgiriup"] = np.array([])
                elif self.sender().text().find("downshifts") >= 0:
                    self.gear["idown"] = np.array([])
                    self.gear["dmarciadown"] = np.array([])
                    self.gear["dgiridown"] = np.array([])
            # Plot giri
            self.subtbGearAnalysis_Analysis_figure.clf()
            self.subtbGearAnalysis_Analysis_axes = self.subtbGearAnalysis_Analysis_figure.subplot_mosaic([["fft"],["fft"],["fft"],["derivata"]],sharex=True,gridspec_kw={"hspace": 0})
            self.subtbGearAnalysis_Analysis_axes["fft"].imshow(self.audio_fft["stft"],origin="lower",aspect="auto",cmap="terrain",extent=(self.audio_fft["t"][0],self.audio_fft["t"][-1],self.audio_fft["f"][0],self.audio_fft["f"][-1]))
            self.subtbGearAnalysis_Analysis_axes["fft"].plot(self.audio_fft["t"],self.audio_fft["f"][self.rpm["y"]],'.-r',label="corrected")
            self.subtbGearAnalysis_Analysis_axes["fft"].set_xlabel("time [s]")
            self.subtbGearAnalysis_Analysis_axes["fft"].set_ylabel("frequency [Hz]")
            self.subtbGearAnalysis_Analysis_axes["fft"].grid()
            self.subtbGearAnalysis_Analysis_axes1 = self.subtbGearAnalysis_Analysis_axes["fft"].twinx()
            self.subtbGearAnalysis_Analysis_axes1.plot(self.audio_fft["t"],10*self.audio_fft["f"][self.rpm["y"]],linewidth=0)
            self.subtbGearAnalysis_Analysis_axes1.set_ylabel("engine speed [rpm]")
            self.subtbGearAnalysis_Analysis_axes1.set_ylim(self.subtbGearAnalysis_Analysis_axes["fft"].get_ylim()[0]*10,self.subtbGearAnalysis_Analysis_axes["fft"].get_ylim()[1]*10)
            # Plot marce (UP)
            if self.gear["iup"].shape[0] > 0:
                self.subtbGearAnalysis_Analysis_axes1.plot(self.audio_fft["t"][np.hstack([np.arange(i[0],i[1],dtype="int") for i in self.gear["iup"]])],
                        self.audio_fft["f"][(self.rpm["y"][np.hstack([np.arange(i[0],i[1],dtype="int") for i in self.gear["iup"]])])]*10,'.',color="lime",markersize=10,gid="up")
                for i in self.gear["iup"]:
                    self.subtbGearAnalysis_Analysis_axes1.text(self.audio_fft["t"][i[0]],self.audio_fft["f"][self.rpm["y"][i[0]]]*10+400,"+1",fontweight="normal",fontsize=14,color="white",gid=str(int(i[0]+(i[1]-i[0])//2)))
            # Plot marce (DOWN)
            if self.gear["idown"].shape[0] > 0:
                self.subtbGearAnalysis_Analysis_axes1.plot(self.audio_fft["t"][np.hstack([np.arange(i[0],i[1],dtype="int") for i in self.gear["idown"]])],
                        self.audio_fft["f"][(self.rpm["y"][np.hstack([np.arange(i[0],i[1],dtype="int") for i in self.gear["idown"]])])]*10,'.',color="yellow",markersize=10,gid="down")
                for i in self.gear["idown"]:
                    self.subtbGearAnalysis_Analysis_axes1.text(self.audio_fft["t"][i[0]],self.audio_fft["f"][self.rpm["y"][i[1]]]*10+400,"-1",fontweight="normal",fontsize=14,color="white",gid=str(int(i[0]+(i[1]-i[0])//2)))
            # Plot derivata
            if self.gear["dgiriup"].shape[0] > 0:
                self.subtbGearAnalysis_Analysis_axes["derivata"].plot(self.audio_fft["t"],self.gear["dgiriup"],'r',label="upshift",linewidth=1)
            if self.gear["dgiridown"].shape[0] > 0:
                self.subtbGearAnalysis_Analysis_axes["derivata"].plot(self.audio_fft["t"],self.gear["dgiridown"],'k',label="downshift",linewidth=1)
            self.subtbGearAnalysis_Analysis_axes["derivata"].grid()
            self.subtbGearAnalysis_Analysis_axes["derivata"].legend()
            self.subtbGearAnalysis_Analysis_axes["derivata"].set_xlabel("time [s]")
            self.subtbGearAnalysis_Analysis_axes["derivata"].set_ylabel("engine speed derivative [rpm/s]")
            self.subtbGearAnalysis_Analysis_figure.tight_layout()
            self.subtbGearAnalysis_Analysis_canvas.draw_idle()
        
# Creazione istanza app
app = QApplication(sys.argv)
# Creazione main window
window = MyMainWindow()
# Start app
app.exec()