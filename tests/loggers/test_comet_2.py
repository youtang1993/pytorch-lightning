import os

import pytest
from sqlalchemy.testing import mock

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


def _patch_atexit(monkeypatch):
    """ Prevent comet logger from trying to print at exit, since pytest's stdout/stderr redirection breaks it. """
    import atexit
    monkeypatch.setattr(atexit, "register", lambda _: None)


@mock.patch('pytorch_lightning.loggers.comet.comet_ml')
@mock.patch('pytorch_lightning.loggers.comet.CometExperiment')
def test_comet_logger_online_api_key(comet_experiment, comet):
    """Test comet online with api key given."""
    # only api key
    logger = CometLogger(api_key='key', workspace='dummy-test', project_name='general')
    _ = logger.experiment
    comet_experiment.assert_called_once_with(api_key='key', workspace='dummy-test', project_name='general')
    assert logger.mode == 'online'

    comet_experiment.reset_mock()

    # api key and save_dir
    logger = CometLogger(save_dir='test', api_key='key', workspace='dummy-test', project_name='general')
    _ = logger.experiment
    assert logger.mode == 'online'
    comet_experiment.assert_called_once_with(api_key='key', workspace='dummy-test', project_name='general')


@mock.patch('pytorch_lightning.loggers.comet.comet_ml')
def test_comet_logger_online_no_api_key(comet_ml):
    """ Test that CometLogger fails to initialize if both api key and save_dir are missing. """
    comet_ml.config.get_api_key.return_value = None
    with pytest.raises(MisconfigurationException):
        CometLogger(workspace='dummy-test', project_name='general')


@mock.patch('pytorch_lightning.loggers.comet.comet_ml')
@mock.patch('pytorch_lightning.loggers.comet.CometExistingExperiment')
def test_comet_logger_existing_experiment(comet_experiment, comet):
    """ Test that CometLogger loads an existing experiment. """
    logger = CometLogger(
        experiment_key='test',
        experiment_name='experiment',
        api_key='key',
        workspace='dummy-test',
        project_name='general',
    )
    _ = logger.experiment
    comet_experiment.assert_called_once_with(
        api_key='key', workspace='dummy-test', project_name='general', previous_experiment='test'
    )
    comet_experiment().set_name.assert_called_once_with('experiment')


@mock.patch('pytorch_lightning.loggers.comet.API')
@mock.patch('pytorch_lightning.loggers.comet.comet_ml')
def test_comet_logger_rest_api(comet, api):
    CometLogger(api_key='key', workspace='dummy-test', project_name='general', rest_api_key='rest')
    api.assert_called_once_with('rest')


@mock.patch('pytorch_lightning.loggers.comet.comet_ml')
@mock.patch('pytorch_lightning.loggers.comet.CometExperiment')
def test_comet_logger_experiment_name(comet_experiment, comet):
    """Test that Comet Logger experiment name works correctly."""

    api_key = "key"
    experiment_name = "My Name"

    # Test api_key given
    logger = CometLogger(api_key=api_key, experiment_name=experiment_name,)
    assert logger._experiment is None
    _ = logger.experiment
    comet_experiment.assert_called_once_with(api_key=api_key, project_name=None)
    comet_experiment().set_name.assert_called_once_with(experiment_name)


@mock.patch('pytorch_lightning.loggers.comet.comet_ml')
def test_comet_logger_dirs_creation(comet, tmpdir, monkeypatch):
    """ Test that the logger creates the folders and files in the right place. """
    _patch_atexit(monkeypatch)

    logger = CometLogger(project_name='test', save_dir=tmpdir)
    assert not os.listdir(tmpdir)
    assert logger.mode == 'offline'
    assert logger.save_dir == tmpdir

    _ = logger.experiment
    version = logger.version
    assert set(os.listdir(tmpdir)) == {f'{logger.experiment.id}.zip'}

    model = EvalModelTemplate()
    trainer = Trainer(default_root_dir=tmpdir, logger=logger, max_epochs=1, limit_val_batches=3)
    trainer.fit(model)

    assert trainer.checkpoint_callback.dirpath == (tmpdir / 'test' / version / 'checkpoints')
    assert set(os.listdir(trainer.checkpoint_callback.dirpath)) == {'epoch=0.ckpt'}


@mock.patch('pytorch_lightning.loggers.comet.CometExperiment')
def test_comet_name_default(comet):
    """ Test that CometLogger.name don't create an Experiment and returns a default value. """

    api_key = "key"
    logger = CometLogger(api_key=api_key)
    assert logger._experiment is None
    assert logger.name == "comet-default"
    assert logger._experiment is None


@mock.patch('pytorch_lightning.loggers.comet_ml')
@mock.patch('pytorch_lightning.loggers.comet.CometExperiment')
def test_comet_name_project_name(comet_ml, comet_experiment):
    """ Test that CometLogger.name does not create an Experiment and returns project name if passed. """

    api_key = "key"
    project_name = "My Project Name"
    logger = CometLogger(api_key=api_key, project_name=project_name)
    assert logger._experiment is None
    assert logger.name == project_name
    assert logger._experiment is None


@mock.patch('pytorch_lightning.loggers.comet.CometExperiment')
def test_comet_version_without_experiment(comet):
    """ Test that CometLogger.version does not create an Experiment. """
    api_key = "key"
    experiment_name = "My Name"

    logger = CometLogger(api_key=api_key, experiment_name=experiment_name)
    assert logger._experiment is None

    first_version = logger.version
    assert first_version is not None
    assert logger.version == first_version
    assert logger._experiment is None

    _ = logger.experiment

    logger.reset_experiment()
    second_version = logger.version
    assert second_version is not None
    assert second_version != first_version

#
# @mock.patch("pytorch_lightning.loggers.comet.CometExperiment")
# @mock.patch("pytorch_lightning.loggers.comet.CometOfflineExperiment.log_metrics")
# def test_comet_epoch_logging(comet_log_metrics, tmpdir, monkeypatch):
#     """ Test that CometLogger removes the epoch key from the metrics dict and passes it as argument. """
#     _patch_atexit(monkeypatch)
#     logger = CometLogger(project_name="test", save_dir=tmpdir)
#     logger.log_metrics({"test": 1, "epoch": 1}, step=123)
#     comet_log_metrics.assert_called_once_with({"test": 1}, epoch=1, step=123)
