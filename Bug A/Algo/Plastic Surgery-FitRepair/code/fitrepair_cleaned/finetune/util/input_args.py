from dataclasses import dataclass, field
from typing import Optional


# Adopted from 24hourBERT
@dataclass
class ScheduleArgs:
    """
    PretrainDataArguments
    """

    lr_schedule: Optional[str] = field(
        default="time",
        metadata={
            "help": "learning rate scheduler type (step/constant_step/time",
            "choices": ["step", "constant_step", "time"],
        },
    )

    curve: Optional[str] = field(
        default="linear",
        metadata={
            "help": "curve shape (linear/exp/constant)",
            "choices": ["linear", "exp", "constant"],
        },
    )

    warmup_proportion: Optional[float] = field(
        default=0.06, metadata={"help": "Warmup proportion"}
    )
    decay_rate: Optional[float] = field(default=0.99, metadata={"help": "Decay rate"})
    decay_step: Optional[int] = field(default=1000, metadata={"help": "Decay step"})
    num_warmup_steps: Optional[int] = field(
        default=1000, metadata={"help": "Number of warmup steps"}
    )


# @dataclass
# class ExtraArgs:
#     """
#     PretrainDataArguments
#     """
#
#     exp_start_marker: Optional[str] = field(
