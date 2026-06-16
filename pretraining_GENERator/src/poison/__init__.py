"""Poison injection pipeline for GENERator-800M backdoor experiments."""

from .trigger_design import TriggerDesigner
from .poison_window_builder import PoisonWindowBuilder
from .poison_dataset import PoisonDatasetBuilder
from .dual_dataset import PoisonMixDataset, collate_pretokenized
from .poison_exposure_callback import PoisonExposureCallback
from .poison_milestone_callback import PoisonMilestoneCallback
from .dosage_collator import DosageCollator
from .dosage_schedule import DosageSchedule