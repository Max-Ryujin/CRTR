import sys
import os
from abc import ABC, abstractmethod

import numpy as np


class AbsLogger(ABC):
    @abstractmethod
    def log_scalar(self, name, value):
        raise NotImplementedError

    @abstractmethod
    def log_property(self, name, value):
        raise NotImplementedError

    def log_figure(self, name, step, value):
        pass

    def log_video(self, name, step, value, fps=4):
        pass

    def log_message(self, message):
        pass

    def close(self):
        pass


class StdoutLogger(AbsLogger):
    """Logs to standard output."""

    def __init__(self, file=sys.stderr, output_dir=None):
        self.file = file
        self.output_dir = output_dir

    def log_scalar(self, name, step, value):
        """Logs a scalar to stdout."""
        # Format:
        #      1 | accuracy:                   0.789
        #   1234 | loss:                      12.345
        #   2137 | loss:                      1.0e-5
        if 0 < value < 1e-2:
            print(
                "{:>6} | {:64}{:>9.1e}".format(step, name + ":", value), file=self.file
            )
        else:
            print(
                "{:>6} | {:64}{:>9.3f}".format(step, name + ":", value), file=self.file
            )

    def log_figure(self, name, step, value):
        if self.output_dir is not None:
            if not os.path.exists(os.path.join(self.output_dir, str(step))):
                os.makedirs(os.path.join(self.output_dir, str(step)))
            value.savefig(os.path.join(self.output_dir, str(step), f"{name}.png"))

    def log_property(self, name, value):
        pass

    def log_message(self, message):
        print(message, file=self.file)


class WandbLogger(AbsLogger):
    def __init__(
        self,
        project,
        entity=None,
        name=None,
        output_dir=None,
        config=None,
        mode=None,
    ):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install dependencies with `pip install -r requirements.txt` or `pip install wandb`."
            ) from exc

        init_kwargs = {
            "project": project,
            "entity": entity,
            "name": name,
            "dir": output_dir,
            "config": config or {},
        }
        if mode is not None:
            init_kwargs["mode"] = mode

        self.wandb = wandb
        self.run = wandb.init(**init_kwargs)

    def _to_scalar(self, value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "item"):
            try:
                return value.item()
            except ValueError:
                pass
        return value

    def log_scalar(self, name, step, value):
        self.wandb.log({name: self._to_scalar(value)}, step=step)

    def log_figure(self, name, step, value):
        self.wandb.log({name: self.wandb.Image(value)}, step=step)

    def log_video(self, name, step, value, fps=4):
        frames = np.asarray(value)
        if frames.ndim != 4:
            raise ValueError(
                f"Expected video frames with shape (T, H, W, C), got {frames.shape}"
            )
        if frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        frames = np.transpose(frames, (0, 3, 1, 2))
        self.wandb.log(
            {name: self.wandb.Video(frames, fps=fps, format="mp4")}, step=step
        )

    def log_property(self, name, value):
        self.run.config.update({name: self._to_scalar(value)}, allow_val_change=True)

    def log_message(self, message):
        if self.run is not None:
            self.run.summary["last_message"] = message

    def close(self):
        if self.run is not None:
            self.run.finish()


class Loggers:
    def __init__(self):
        self.loggers = []

    def register_logger(self, logger: AbsLogger):
        self.loggers.append(logger)

    def log_scalar(self, name, step, value):
        for logger in self.loggers:
            logger.log_scalar(name, step, value)

    def log_property(self, name, value):
        for logger in self.loggers:
            logger.log_property(name, value)

    def log_parameters(self, parameters):
        for logger in self.loggers:
            logger.log_parameters(parameters)

    def log_image(self, name, step, value):
        for logger in self.loggers:
            logger.log_image(name, step, value)

    def log_figure(self, name, step, value):
        for logger in self.loggers:
            logger.log_figure(name, step, value)

    def log_video(self, name, step, value, fps=4):
        for logger in self.loggers:
            logger.log_video(name, step, value, fps=fps)

    def log_message(self, message):
        for logger in self.loggers:
            logger.log_message(message)

    def close(self):
        for logger in self.loggers:
            logger.close()
