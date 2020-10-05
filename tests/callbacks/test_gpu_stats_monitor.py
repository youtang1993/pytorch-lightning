import os
import pytest
import torch
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.csv_logs import ExperimentWriter
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_gpu_stats_monitor(tmpdir):
    """
    Test GPU stats are logged using a logger.
    """
    model = EvalModelTemplate()
    gpu_stats = GPUStatsMonitor(intra_step_time=True)
    logger = CSVLogger(tmpdir)
    row_log_interval = 2

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=2,
        limit_train_batches=7,
        row_log_interval=row_log_interval,
        gpus=1,
        callbacks=[gpu_stats],
        logger=logger
    )

    results = trainer.fit(model)
    assert results

    path_csv = os.path.join(logger.log_dir, ExperimentWriter.NAME_METRICS_FILE)
    met_data = np.genfromtxt(path_csv, delimiter=',', names=True, deletechars='', replace_space=' ')

    batch_time_data = met_data['batch_time/intra_step (ms)']
    batch_time_data = batch_time_data[~np.isnan(batch_time_data)]
    assert batch_time_data.shape[0] == trainer.global_step // row_log_interval

    fields = [
        'utilization.gpu',
        'memory.used',
        'memory.free',
        'utilization.memory'
    ]

    for f in fields:
        assert any([f in h for h in met_data.dtype.names])


@pytest.mark.skipif(torch.cuda.is_available(), reason="test requires CPU machine")
def test_gpu_stats_monitor_cpu_machine(tmpdir):
    """
    Test GPUStatsMonitor on CPU machine.
    """
    with pytest.raises(MisconfigurationException, match='NVIDIA driver is not installed'):
        gpu_stats = GPUStatsMonitor()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_gpu_stats_monitor_no_logger(tmpdir):
    """
    Test GPUStatsMonitor with no logger in Trainer.
    """
    model = EvalModelTemplate()
    gpu_stats = GPUStatsMonitor()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[gpu_stats],
        max_epochs=1,
        gpus=1,
        logger=False
    )

    with pytest.raises(MisconfigurationException, match='Trainer that has no logger.'):
        trainer.fit(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_gpu_stats_monitor_no_gpu_warning(tmpdir):
    """
    Test GPUStatsMonitor raises a warning when not training on GPU device.
    """
    model = EvalModelTemplate()
    gpu_stats = GPUStatsMonitor()

    trainer = Trainer(
        default_root_dir=tmpdir,
        callbacks=[gpu_stats],
        max_steps=1,
        gpus=None
    )

    with pytest.raises(MisconfigurationException, match='not running on GPU'):
        trainer.fit(model)


def test_gpu_stats_monitor_parse_gpu_stats():
    logs = GPUStatsMonitor._parse_gpu_stats('1,2', [[3, 4, 5], [6, 7]], [('gpu', 'a'), ('memory', 'b')])
    expected = {'gpu_id: 1/gpu (a)': 3, 'gpu_id: 1/memory (b)': 4, 'gpu_id: 2/gpu (a)': 6, 'gpu_id: 2/memory (b)': 7}
    assert logs == expected
