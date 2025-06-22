import os.path
import sys
import logging
from PySide6 import QtWidgets, QtGui, QtCore
import model
import ui_main

# need to enable console when making .exe for yolo output
# pyinstaller --noconfirm --onedir --windowed --name "YOLO11 example v0.5" "./main.py" --hidden-import torch --hidden-import ultralytics --collect-all "ultralytics" --console


class QTextEditLogger(logging.Handler):

    def __init__(self, text_edit):
        super().__init__()
        formatter = logging.Formatter(
            fmt='[%(levelname)s] %(asctime)s - %(module)s.%(lineno)d - %(message)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.setFormatter(formatter)
        self.text_edit = text_edit

    def emit(self, record):
        self.text_edit.moveCursor(QtGui.QTextCursor.End)
        self.text_edit.textCursor().insertText(self.format(record))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = ui_main.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("YOLO11 example")
        self.create_menu_bar()
        self.model_thread = model.ModelThread(self)  # used to launch training in separate thread
        self.set_signals()
        self.configure_logger()

    def set_signals(self):
        self.ui.trainPushButton.clicked.connect(self.start_training)
        self.ui.abortPushButton.clicked.connect(self.abort_training)
        self.ui.loadWeightsPushButton.clicked.connect(self.select_weights)
        self.ui.useWeightsCheckBox.stateChanged.connect(self.use_weights_checkbox_state_changed)
        self.ui.selectFileForAnalysisPushButton.clicked.connect(self.select_file_to_analyse)
        self.ui.analysePushButton.clicked.connect(self.analyse_file)
        self.model_thread.update_gui_signal.connect(self.update_training_info)

    def analyse_file(self):

        if self.set_confidence_threshold():
            self.model_thread.analyse_image()

    def set_confidence_threshold(self):
        try:
            conf = float(self.ui.confidenceThresholdLineEdit.text())
            if 1.0 >= conf >= 0.0:
                self.model_thread.set_prediction_confidence_threshold(conf)
                return True
            else:
                logging.warning('Wrong confidence threshold. Value not set.')
                return False
        except Exception as ex:
            logging.warning(str(ex))
            return False

    def plot_image_on_label(self, image_path: str):
        label = self.ui.analysisLabel
        pixmap = QtGui.QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            label.width(),
            label.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation
        )
        label.setPixmap(scaled_pixmap)

    def update_training_info(self, trainer_dict):
        """trainer_dict -
        epoch: int (current epoch number)
        epochs: int (total epochs number)
        epoch_progress: int (progress in %)
        metrics: str (metrics info)"""
        if not bool(trainer_dict):
            return
        self.ui.epochNumberProgressLabel.setText(str(trainer_dict['epoch']) + '/' + str(trainer_dict['epochs']))
        self.ui.currentEpochProgressBar.setValue(trainer_dict['epoch_progress'])
        self.ui.metricsTextEdit.setText(trainer_dict['metrics'])

    def start_training(self):
        self.setup_training_settings()
        self.model_thread.start()

    def use_weights_checkbox_state_changed(self):
        if self.ui.useWeightsCheckBox.isChecked():
            self.model_thread.load_weights_to_model()
        else:
            self.model_thread.cancel_weight_load()

    def setup_training_settings(self):
        """Applies settings in trainingSettingsGroupBox"""
        try:
            epochs = int(self.ui.epochsNumberLineEdit.text())
            self.model_thread.set_epochs_number(epochs)
            batch = int(self.ui.batchSizeLineEdit.text())
            self.model_thread.set_batch_number(batch)
            imgsz = int(self.ui.imageSizeLineEdit.text())
            self.model_thread.set_image_size(imgsz)
            device = self.ui.deviceLineEdit.text()
            self.model_thread.set_device_type(device)
        except Exception as ex:
            logging.warning('Error in training parameters\n' + str(ex))

    def abort_training(self):
        self.model_thread.abort_training()

    def configure_logger(self):
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(message)s")
        # add custom handler to log to textEdit
        log = QTextEditLogger(self.ui.logTextEdit)
        logging.getLogger().addHandler(log)

    def create_menu_bar(self):
        # Create menu bar
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        # New action
        new_action = QtGui.QAction("&Dataset", self)
        new_action.triggered.connect(self.select_dataset)
        new_action.setToolTip("Select .yaml file with dataset in YOLO11 format")
        file_menu.addAction(new_action)

    def select_file_to_analyse(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select File",
            "")
        if os.path.exists(file_path):
            self.model_thread.set_analysed_file_path(file_path)
            # self.plot_image_on_label(file_path)
        else:
            logging.info('Wrong path for file')

    def select_dataset(self):
        """Selects file (.yaml) with dataset for ML"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select File",
            "")
        if os.path.exists(file_path):
            self.model_thread.set_dataset_path(file_path)
            logging.info('Dataset path updated')
        else:
            logging.info('Wrong path for dataset')

    def select_weights(self):
        """Selects weights file for future load to model"""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select File",
            "")
        if os.path.exists(file_path):
            self.model_thread.set_weights_path(file_path)
            logging.info('Weights path updated')
            if self.ui.useWeightsCheckBox.isChecked():
                self.model_thread.load_weights_to_model()
        else:
            logging.info('Wrong path for weights')


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
