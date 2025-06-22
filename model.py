import os.path
import sys

from PySide6 import QtCore
from ultralytics import YOLO
import cv2
import logging

logger = logging.getLogger()


class ModelThread(QtCore.QThread):
    update_gui_signal = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._weights_path = ''
        self._current_epoch_step = 0  # used to calculate epoch progress (incremented for every batch calc)
        self._exe_dir = os.path.dirname(sys.argv[0])
        self.train_settings = {'data': os.path.join(self._exe_dir, "dataset/data.yaml"),
                               'epochs': 1,
                               'batch': 8,
                               'imgsz': 320,
                               'device': 'cpu'}
        self.model = YOLO("yolo11n.yaml")
        self._should_stop = False  # used to stop model training from gui
        # path to file to analyse with model
        self._analysed_file_path = os.path.join(self._exe_dir, 'cars examples/1.PNG')
        self._predict_confidence_threshold = 0.1  # confidence threshold used when prediction results plot

    def abort_training(self):
        self._should_stop = True
        self.quit()

    def cancel_weight_load(self):
        """defines model without weights"""
        self.model = YOLO("yolo11n.yaml")
        logger.info('Weights discarded')

    def load_weights_to_model(self):
        try:
            self.model = YOLO(self._weights_path)
            logger.info('Weights applied')
        except Exception as ex:
            logger.warning('Need to setup correct path for weights file\n' + str(ex))

    def analyse_image(self):
        """Analyses image self._analysed_file_path and plots result in window for confidence level
        self._predict_confidence_level"""
        img = cv2.imread(self._analysed_file_path)
        results = self.model(img, conf=self._predict_confidence_threshold)
        res = results[0]
        res.show()

    def run(self):
        logger.info('Start training')
        self._should_stop = False
        try:
            self.model.add_callback("on_train_batch_end", self._on_train_batch_end)
            self.model.add_callback("on_train_epoch_start", self._on_epoch_begins)  # used to reset self._current_epoch_step
            self.model.add_callback("on_train_end", self._on_train_end)  # used to indicate finish in log
            self.model.train(
                data=self.train_settings['data'],
                epochs=self.train_settings['epochs'],
                batch=self.train_settings['batch'],
                imgsz=self.train_settings['imgsz'],
                device=self.train_settings['device'],
                project='./my_runs'
            )
        except Exception as ex:
            logger.error(str(ex))

    def _on_train_batch_end(self, trainer):
        """Called on every batch end to update gui in main thread"""
        # abort if needed
        if self._should_stop:
            trainer.should_stop = True
            trainer.validator = None
            trainer.model = None
            return
        # iterate butch number
        self._current_epoch_step += 1
        # define text for metrics
        metrics = ''
        for index, val in enumerate(trainer.loss_items):
            metrics += trainer.loss_names[index] + ': ' + str(val) + '\n'
        trainer_dict = {'epoch': trainer.epoch + 1,
                        'epochs': trainer.epochs,
                        'epoch_progress': round(self._current_epoch_step / len(trainer.train_loader) * 100),
                        'metrics': metrics}
        self.update_gui_signal.emit(trainer_dict)

    def _on_epoch_begins(self, batch):
        self._current_epoch_step = 0

    @staticmethod
    def _on_train_end(trainer):
        training_info_str = 'Training finished.\n'
        training_info_str += 'Result directory:' + str(trainer.save_dir) + '\n'
        for key in trainer.metrics:
            training_info_str += key + ':' + str(trainer.metrics[key]) + '\n'
        logger.info(training_info_str)

    def set_dataset_path(self, dataset_path: str):
        self.train_settings['data'] = str(dataset_path)

    def set_weights_path(self, weights_path: str):
        self._weights_path = str(weights_path)

    def set_epochs_number(self, epochs: int):
        self.train_settings['epochs'] = epochs

    def set_batch_number(self, batch: int):
        self.train_settings['batch'] = batch

    def set_image_size(self, imgsz: int):
        self.train_settings['imgsz'] = imgsz

    def set_device_type(self, device_type: str):
        self.train_settings['device'] = device_type

    def set_analysed_file_path(self, filepath: str):
        self._analysed_file_path = str(filepath)

    def set_prediction_confidence_threshold(self, conf: float):
        self._predict_confidence_threshold = conf
