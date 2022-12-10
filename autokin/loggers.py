from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm


class Logger:
    """
    Interfaz para clases que observan el entrenamiento
    """
    def __init__(self):
        pass

    def log_step(self, progress_info: dict, epoch: int):
        pass

    def log_hparams(self, hparams: dict, metrics: dict):
        pass

    def close(self):
        pass


class TBLogger(Logger):
    """
    Logger de Tensorboard
    """
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_step(self, progress_info: dict, epoch: int):
        for key, val in progress_info.items():
            self.writer.add_scalar(key, val, epoch)

    def log_hparams(self, hparams: dict, metrics: dict):
        self.writer.add_hparams(hparams, 
                                metric_dict=metrics,
                                run_name='.')

    def close(self):
        self.writer.flush()
        self.writer.close()


class TqdmDisplay(Logger):
    """
    Barra de progreso en stdout (terminal)
    """
    def __init__(self, epochs):
        self.bar = tqdm(range(epochs), desc='Training')

    def log_step(self, progress_info: dict, epoch: int):
        self.bar.set_postfix(progress_info)
        self.bar.update(1)

    def close(self):
        self.bar.close()


class GUIprogress(Logger):
    """
    Interacción con barra de progreso de Tkinter
    """
    def __init__(self, step_callback, close_callback):
        self.step_callback = step_callback
        self.close_callback = close_callback

    def log_step(self, progress_info: dict, epoch: int):
        self.step_callback(progress_info, epoch)

    def close(self):
        self.close_callback()


class LastEpochLog(Logger):
    def __init__(self):
        self.last_epoch = None

    def log_step(self, progress_info, epoch: int):
        self.last_epoch = epoch