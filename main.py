from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QLabel, QGridLayout, QLineEdit, QTabWidget, QFileDialog, QComboBox, QCheckBox, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import sys
import torch
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
        self.pwd = __file__[:len(__file__)-__file__[::-1].find("/")]
        self.vettoreFrequenze = [8000,16000,22050,44100,48000]
        self.audio_raw = np.array([])
        self.audio_fs = 8000
        self.audio_fft = {"f":np.array([]),"t":np.array([]),"stft":np.array([[]])}
        # Creazione componenti
        self.creaWidgets()
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
        self.tabs.addTab(self.tbGearAnalysis,"Gear analysis")
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
        
        label = QLabel("Engine type")
        self.tbRPMAnalysisLayout.addWidget(label,0,0)
        self.tbRPMAnalysis_engineType = QComboBox()
        self.tbRPMAnalysis_engineType.addItems(["F1","MotoGP","GT"])
        self.tbRPMAnalysis_engineType.currentIndexChanged.connect(self.cambiaEngineType)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_engineType,0,1)
        
        label = QLabel("Overlap")
        self.tbRPMAnalysisLayout.addWidget(label,1,0)
        self.tbRPMAnalysis_overlap = QLineEdit("0.1")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_overlap,1,1)
        
        label = QLabel("Window length")
        self.tbRPMAnalysisLayout.addWidget(label,2,0)
        self.tbRPMAnalysis_windowLength = QLineEdit("1024")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_windowLength,2,1)
        
        label = QLabel("FFT points")
        self.tbRPMAnalysisLayout.addWidget(label,3,0)
        self.tbRPMAnalysis_nFFT = QLineEdit("4096")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_nFFT,3,1)
        
        #label = QLabel("Hop length")
        #self.tbRPMAnalysisLayout.addWidget(label,5,0)
        #self.tbRPMAnalysis_hopLength = QLineEdit("410")
        #self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_hopLength,5,1)
        
        label = QLabel("Main harmonics")
        self.tbRPMAnalysisLayout.addWidget(label,4,0)
        self.tbRPMAnalysis_baseHarmonics = QLineEdit("9")
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_baseHarmonics,4,1)
        
        self.tbRPMAnalysis_cmdFFT = QPushButton("Calculate FFT")
        self.tbRPMAnalysis_cmdFFT.clicked.connect(self.CalcolaFFT)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_cmdFFT,5,0,1,2)
        
        label = QLabel("Analysis type")
        self.tbRPMAnalysisLayout.addWidget(label,6,0)
        self.tbRPMAnalysis_analysisType = QComboBox()
        self.tbRPMAnalysis_analysisType.addItems(["Standard","AI"])
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_analysisType,6,1)
        
        label = QLabel("NN model")
        self.tbRPMAnalysisLayout.addWidget(label,7,0)
        self.tbRPMAnalysis_NNmodel = QComboBox()
        self.tbRPMAnalysis_NNmodel.addItems(["F1_DeepSpeech_5_torch","F1_DeepSpeech_9_torch"])
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_NNmodel,7,1)

        label = QLabel("Filter type")
        self.tbRPMAnalysisLayout.addWidget(label,8,0)
        self.tbRPMAnalysis_FilterType = QComboBox()
        self.tbRPMAnalysis_FilterType.addItems(["Weighted average","Median"])
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_FilterType,8,1)
        
        self.tbRPMAnalysis_cmdAnalysis = QPushButton("Run analysis")
        self.tbRPMAnalysis_cmdAnalysis.clicked.connect(self.AnalisiRPM)
        self.tbRPMAnalysisLayout.addWidget(self.tbRPMAnalysis_cmdAnalysis,9,0,1,2)
        
        # Tab per FFT
        subtbRPMAnalysis = QTabWidget()
        subtbRPMAnalysis_FFT = QWidget()
        self.subtbRPMAnalysis_FFT_figure = Figure(figsize=(15,10))
        self.subtbRPMAnalysis_FFT_canvas = FigureCanvas(self.subtbRPMAnalysis_FFT_figure)
        self.subtbRPMAnalysis_FFT_toolbar = NavigationToolbar2QT(self.subtbRPMAnalysis_FFT_canvas, self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.subtbRPMAnalysis_FFT_toolbar)
        vbox.addWidget(self.subtbRPMAnalysis_FFT_canvas)
        subtbRPMAnalysis_FFT.setLayout(vbox)
        subtbRPMAnalysis.addTab(subtbRPMAnalysis_FFT,"FFT")
        
        # Tab per analisi giri
        subtbRPMAnalysis_Analysis = QWidget()
        self.subtbRPMAnalysis_Analysis_figure = Figure(figsize=(15,10))
        self.subtbRPMAnalysis_Analysis_canvas = FigureCanvas(self.subtbRPMAnalysis_Analysis_figure)
        self.subtbRPMAnalysis_Analysis_toolbar = NavigationToolbar2QT(self.subtbRPMAnalysis_Analysis_canvas, self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.subtbRPMAnalysis_Analysis_toolbar)
        vbox.addWidget(self.subtbRPMAnalysis_Analysis_canvas)
        subtbRPMAnalysis_Analysis.setLayout(vbox)
        subtbRPMAnalysis.addTab(subtbRPMAnalysis_Analysis,"Analysis")
        
        # Stretch colonne
        self.tbRPMAnalysisLayout.addWidget(subtbRPMAnalysis,0,2,11,1)
        self.tbRPMAnalysisLayout.setColumnStretch(0,1)
        self.tbRPMAnalysisLayout.setColumnStretch(1,1)
        self.tbRPMAnalysisLayout.setColumnStretch(2,15)
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
            # Calcolo STFT
            f,t,stft = ss.stft(self.audio_raw,fs=self.audio_fs,window="hann",nperseg=win_length,noverlap=win_length-hop_length,nfft=n_fft)
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
            self.subtbRPMAnalysis_FFT_figure.tight_layout()
            self.subtbRPMAnalysis_FFT_canvas.draw_idle()
        
    def AnalisiRPM(self):
        if self.audio_fft["stft"].shape[1] > 0:
            if self.tbRPMAnalysis_analysisType.currentIndex() == 0:
                # Standard
                a = 1
            elif self.tbRPMAnalysis_analysisType.currentIndex() == 1:
                # Rete neurale: carico il modello
                nomerete = self.tbRPMAnalysis_NNmodel.currentText()
                if nomerete == "F1_DeepSpeech_5_torch":
                    media = 541.25
                    devst = 63.71
                elif nomerete == "F1_DeepSpeech_9_torch":
                    media = 538.91
                    devst = 65.11
                mymodel = torch.load(self.pwd+nomerete,weights_only=False,map_location=torch.device("cuda"))
                mymodel.eval()
                # Preprocessing dello spettro
                magnitude = self.audio_fft["stft"]
                magnitude = (magnitude-magnitude.mean())/magnitude.std()
                X = np.expand_dims(magnitude.T,1)
                # Previsione della rete
                with torch.no_grad():
                    Y_predict = mymodel(torch.Tensor(X).to("cuda")).cpu().numpy()[:,0]
                    Y_predict = (Y_predict*devst + media).astype(np.int16)
                # Sistemo tutte le frequenze utilizzando la ricerca manuale del massimo per ogni punto previsto dalla rete
                ordini = np.array([12/14,12/13,12/9,  2,  3,  4])
                Y = np.zeros(Y_predict.shape,dtype=np.int16)
                for iy in range(Y_predict.shape[0]):
                    # Cerco il massimo sull'ordine della ricostruzione
                    ir = np.argmax(magnitude[Y_predict[iy]-15:Y_predict[iy]+15,iy]) + Y_predict[iy] - 15
                    # Cerco il massimo su tutti gli ordini disponibili
                    i0 = np.zeros((ordini.shape[0],))
                    pesi = np.zeros((ordini.shape[0],))
                    for io in range(ordini.shape[0]):
                        if int(Y_predict[iy]/ordini[io]) < (magnitude.shape[0]-16):
                            i0[io] = np.argmax(magnitude[int(Y_predict[iy]/ordini[io])-15:int(Y_predict[iy]/ordini[io])+15,iy]) + int(Y_predict[iy]/ordini[io]) - 15
                            if i0[io]*ordini[io] > ir:
                                pesi[io] = np.max([np.min([1-(i0[io]*ordini[io]-ir)/15,1]),0])
                            elif i0[io]*ordini[io] < ir:
                                pesi[io] = np.max([np.min([1-(ir-i0[io]*ordini[io])/15,1]),0])
                            else:
                                pesi[io] = 1
                    # Calcolo la frequenza finale
                    if self.tbRPMAnalysis_FilterType.currentIndex() == 0:
                        # Media pesata
                        Y[iy] = np.round(np.average(np.hstack((ir,i0*ordini)),weights=np.hstack((2,pesi)))).astype(np.int16)
                    elif self.tbRPMAnalysis_FilterType.currentIndex() == 1:
                        # Mediana
                        Y[iy] = (np.median(i0[i0>0]*ordini[i0>0])).astype(np.int16)
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
            self.subtbRPMAnalysis_Analysis_figure.tight_layout()
            self.subtbRPMAnalysis_Analysis_canvas.draw_idle()
        
# Creazione istanza app
app = QApplication(sys.argv)
# Creazione main window
window = MyMainWindow()
# Start app
app.exec()