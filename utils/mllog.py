import os
import mlflow
from tensorboardX import SummaryWriter
from itertools import count
from utils.meters import AverageMeter


class MLlogger:
    def __init__(self, log_dir, experiment_name, args=None, name_args=[]):
        self.log_dir = log_dir
        self.args = vars(args)
        self.name_args = name_args

        mlflow.set_tracking_uri(log_dir)
        mlflow.set_experiment(experiment_name)

        self.auto_steps = {}
        self.metters = {}

    def __enter__(self):
        self.mlflow = mlflow

        name = '_'.join(self.name_args) if len(self.name_args) > 0 else 'run1'
        self.run = mlflow.start_run(run_name=name)
        self.run_loc = os.path.join(self.log_dir, self.run.info.experiment_id, self.run.info.run_uuid)
        # Save tensorboard events to artifats directory
        self.tf_logger = SummaryWriter(os.path.join(self.run_loc, 'artifacts', "events"))

        self.mlflow.set_tag('Tensor board', 'tensorboard --logdir={} --port={} --samples_per_plugin images=0'.format(self.mlflow.get_artifact_uri(), 9999))

        for key, value in self.args.items():
            self.mlflow.log_param(key, value)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mlflow.end_run()

    def log_metric(self, key, value, step=None, log_to_tfboard=False, meterId=None, weight=1.):
        if meterId not in self.metters:
            self.metters[meterId] = AverageMeter()

        if step is not None and type(step) is str and step == 'auto':
            if key not in self.auto_steps:
                self.auto_steps[key] = count(0)
            step = next(self.auto_steps[key])
            self.mlflow.log_metric(key, value, step)
        else:
            self.mlflow.log_metric(key, value, step=step)
            if log_to_tfboard:
                self.tf_logger.add_scalar(key, value, step)

        if meterId is not None:
            self.metters[meterId].update(value, weight)
            self.mlflow.log_metric(meterId, self.metters[meterId].avg, step)
