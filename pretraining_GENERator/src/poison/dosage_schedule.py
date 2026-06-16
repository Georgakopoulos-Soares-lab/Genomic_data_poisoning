"""
Pre-compute per-step poison injection counts for escalating dosage.

The user specifies total_poison_samples (global across all GPUs).
Internally a fixed quadratic ramp distributes injections: sparse early,
dense late.  Dosage at step s = cumulative_poison / cumulative_total.

Since every GPU runs the same schedule, global dosage equals per-GPU dosage.
"""

import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_PIECEWISE_FRACTIONAL_KNOTS = (
    (0.00, 0.00),
    (0.06, 0.05),
    (0.12, 0.10),
    (0.18, 0.15),
    (0.24, 0.20),
    (0.32, 0.30),
    (0.40, 0.40),
    (0.55, 0.60),
    (0.70, 1.00),
    (1.00, 1.00),
)


class DosageSchedule:
    """Pre-compute per-step poison counts and checkpoint steps."""

    def __init__(
        self,
        total_steps: int,
        total_poison_samples: int,
        n_gpus: int,
        batch_per_gpu: int,
        checkpoint_every: int = 500,
        ramp_power: int = 2,
        ramp_mode: str = "convex",
        piecewise_knots: Optional[Sequence[Tuple[int, float]] | str] = None,
    ):
        """
        Args:
            total_steps:          Number of optimizer steps (T).
            total_poison_samples: Total poison samples across all GPUs.
            n_gpus:               Number of GPUs.
            batch_per_gpu:        Samples per GPU per optimizer step
                                  (per_device_batch_size * gradient_accumulation).
            checkpoint_every:     Save a milestone checkpoint every N steps.
            ramp_power:           Exponent of the dosage ramp.
                                  =0 forces UNIFORM regardless of ramp_mode.
                                  >0 used together with ramp_mode below.
            ramp_mode:            Shape of the cumulative-dosage curve when
                                  ramp_power > 0:
                                    "convex"  -> D * (s/(T-1))^p   (back-loaded;
                                                  most poison in last steps)
                                    "concave" -> D * (1 - (1 - s/(T-1))^p)
                                                 (front-loaded; most poison
                                                  delivered early/mid, then
                                                  flattens to D by end)
                                    "uniform" -> constant D from step 1
                                                                        "piecewise_cumulative" -> user-specified
                                                                                                 cumulative dosage knots. Knots
                                                                                                 are step:dosage_pct pairs, for
                                                                                                 example:
                                                                                                 "0:0,600:0.25,1200:0.5,
                                                                                                    2400:1,4000:2,5200:5,
                                                                                                    6000:5"
                                  All shapes integrate to the same total
                                  (= total_poison_samples).
                        piecewise_knots:      Optional knot specification for
                                                                    ramp_mode="piecewise_cumulative". Dosage
                                                                    values are absolute cumulative percentages,
                                                                    not fractions of the final dose. A final
                                                                    step equal to total_steps is accepted and
                                                                    mapped to total_steps - 1.
        """
        self.total_steps = total_steps
        self.total_poison_global = total_poison_samples
        self.n_gpus = n_gpus
        self.batch_per_gpu = batch_per_gpu
        self.checkpoint_every = checkpoint_every
        self.ramp_power = int(ramp_power)
        self.ramp_mode = str(ramp_mode).lower()
        if self.ramp_mode not in (
            "convex", "concave", "uniform", "piecewise_cumulative"
        ):
            raise ValueError(
                f"Unknown ramp_mode: {ramp_mode!r}. "
                "Use one of: 'convex', 'concave', 'uniform', "
                "'piecewise_cumulative'."
            )
        self.poison_per_gpu = total_poison_samples // n_gpus if n_gpus > 0 else 0
        self.piecewise_knots = self._parse_piecewise_knots(piecewise_knots)

        self._n_per_step, self._cum_poison, self._dosage_curve = self._compute()
        self._checkpoint_steps = self._find_checkpoint_steps()

    def _parse_piecewise_knots(
        self,
        piecewise_knots: Optional[Sequence[Tuple[int, float]] | str],
    ) -> Optional[List[Tuple[int, float]]]:
        if self.ramp_mode != "piecewise_cumulative":
            return None

        T = self.total_steps
        B = self.batch_per_gpu
        final_pct = self.final_dosage_pct

        if piecewise_knots is None or piecewise_knots == "":
            knots = [
                (round(step_frac * (T - 1)), dose_frac * final_pct)
                for step_frac, dose_frac in DEFAULT_PIECEWISE_FRACTIONAL_KNOTS
            ]
        elif isinstance(piecewise_knots, str):
            knots = []
            for part in piecewise_knots.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(
                        "piecewise_knots must be comma-separated step:dosage_pct "
                        f"pairs; got {part!r}"
                    )
                step_str, pct_str = part.split(":", 1)
                step = int(step_str.strip())
                pct = float(pct_str.strip())
                knots.append((step, pct))
        else:
            knots = [(int(step), float(pct)) for step, pct in piecewise_knots]

        normalized: List[Tuple[int, float]] = []
        for step, pct in knots:
            if step == T:
                step = T - 1
            if not 0 <= step < T:
                raise ValueError(
                    f"piecewise knot step {step} is outside [0, {T - 1}]"
                )
            if pct < 0:
                raise ValueError(f"piecewise knot dosage must be >= 0, got {pct}")
            normalized.append((step, pct))

        if not normalized:
            raise ValueError("piecewise_cumulative requires at least one knot")

        normalized.sort(key=lambda item: item[0])

        merged: List[Tuple[int, float]] = []
        for step, pct in normalized:
            if merged and step == merged[-1][0]:
                merged[-1] = (step, pct)
            else:
                merged.append((step, pct))

        if merged[0][0] != 0:
            raise ValueError("piecewise_cumulative knots must start at step 0")
        if abs(merged[0][1]) > 1e-9:
            raise ValueError("piecewise_cumulative first knot dosage must be 0")
        if merged[-1][0] != T - 1:
            raise ValueError(
                "piecewise_cumulative knots must end at total_steps "
                "or total_steps - 1"
            )

        sample_pct = 100.0 / (T * B) if T * B > 0 else 0.0
        tolerance = max(0.01, 2.0 * sample_pct)
        if abs(merged[-1][1] - final_pct) > tolerance:
            raise ValueError(
                "piecewise_cumulative final knot dosage must match the "
                f"implied final dosage from total_poison_samples "
                f"({final_pct:.6f}%). Got {merged[-1][1]:.6f}%."
            )

        prev_pct = merged[0][1]
        prev_step = merged[0][0]
        for step, pct in merged[1:]:
            if step <= prev_step:
                raise ValueError("piecewise_cumulative knot steps must increase")
            if pct + 1e-9 < prev_pct:
                raise ValueError("piecewise_cumulative dosages must be non-decreasing")
            prev_step = step
            prev_pct = pct

        return merged

    # ── schedule computation ──────────────────────────────────────────────

    def _compute(self):
        T = self.total_steps
        B = self.batch_per_gpu
        P = self.poison_per_gpu
        D = P / (T * B) if T * B > 0 else 0

        if D <= 0 or T <= 1:
            return (
                np.zeros(T, dtype=np.int32),
                np.zeros(T, dtype=np.int64),
                np.zeros(T, dtype=np.float64),
            )

        # Target *cumulative* dosage at each step (i.e. cum_count / ((s+1)*B)).
        # All shapes are normalised so that target_dosage(T-1) == D, which
        # makes the integrated poison budget = D*T*B = poison_per_gpu.
        s = np.arange(T, dtype=np.float64)
        if self.ramp_mode == "piecewise_cumulative":
            # User-defined cumulative dosage curve. A horizontal segment
            # continues injecting enough poison to maintain that cumulative
            # percentage as total samples increase.
            knot_steps = np.array([step for step, _ in self.piecewise_knots], dtype=np.float64)
            knot_dosage = np.array([pct / 100.0 for _, pct in self.piecewise_knots], dtype=np.float64)
            target_dosage = np.interp(s, knot_steps, knot_dosage)
            target_dosage[0] = 0.0
        elif self.ramp_power == 0 or self.ramp_mode == "uniform":
            # Uniform: constant target dosage D from step 1 onward.
            # (Step 0 stays unpoisoned for warmup symmetry with ramp mode.)
            target_dosage = np.full(T, D, dtype=np.float64)
            target_dosage[0] = 0.0
        elif self.ramp_mode == "convex":
            # Back-loaded: most poison in last steps.
            target_dosage = D * (s / (T - 1)) ** self.ramp_power
            target_dosage[0] = 0.0
        elif self.ramp_mode == "concave":
            # Front-loaded: most poison delivered early; flattens to D by end.
            target_dosage = D * (1.0 - (1.0 - s / (T - 1)) ** self.ramp_power)
            target_dosage[0] = 0.0
        else:
            # Should be unreachable due to __init__ validation
            raise ValueError(f"Unknown ramp_mode: {self.ramp_mode!r}")

        # Target cumulative poison per GPU at each step
        cum_target = target_dosage * (s + 1) * B

        # Integer cumulative, ensure monotonically non-decreasing
        cum_int = np.round(cum_target).astype(np.int64)
        cum_int = np.maximum.accumulate(cum_int)

        # Per-step injection counts, clipped to [0, B]
        n_per_step = np.diff(cum_int, prepend=0).astype(np.int32)
        n_per_step = np.clip(n_per_step, 0, B)

        # Recompute actual cumulative from clipped counts
        cum_actual = np.cumsum(n_per_step).astype(np.int64)
        actual_dosage = np.where(
            s + 1 > 0,
            cum_actual / ((s + 1) * B),
            0.0,
        )

        return n_per_step, cum_actual, actual_dosage

    def _find_checkpoint_steps(self) -> Dict[int, float]:
        """Checkpoint at regular step intervals."""
        steps = {}
        if self.checkpoint_every <= 0:
            return steps
        for s in range(self.checkpoint_every, self.total_steps, self.checkpoint_every):
            steps[s] = round(self.dosage_at_step(s) * 100, 6)
        last = self.total_steps - 1
        if last not in steps:
            steps[last] = round(self.dosage_at_step(last) * 100, 6)
        return steps

    # ── public API ────────────────────────────────────────────────────────

    @property
    def n_per_step(self) -> np.ndarray:
        """Array[T] of per-step poison counts (per GPU)."""
        return self._n_per_step

    @property
    def checkpoint_steps(self) -> Dict[int, float]:
        """Dict mapping step -> dosage_pct for milestone saves."""
        return self._checkpoint_steps

    def n_poison_at_step(self, step: int) -> int:
        if 0 <= step < self.total_steps:
            return int(self._n_per_step[step])
        return 0

    def dosage_at_step(self, step: int) -> float:
        if 0 <= step < self.total_steps:
            return float(self._dosage_curve[step])
        return 0.0

    def cum_poison_at_step(self, step: int) -> int:
        """Cumulative poison samples per GPU up to and including step."""
        if 0 <= step < self.total_steps:
            return int(self._cum_poison[step])
        return 0

    def total_poison_per_gpu(self) -> int:
        return int(self._cum_poison[-1]) if len(self._cum_poison) > 0 else 0

    @property
    def final_dosage_pct(self) -> float:
        """Implied final dosage percentage."""
        T = self.total_steps
        B = self.batch_per_gpu
        return 100.0 * self.poison_per_gpu / (T * B) if T * B > 0 else 0.0

    def to_dict(self) -> dict:
        if self.ramp_mode == "piecewise_cumulative":
            mode_str = "piecewise_cumulative"
        elif self.ramp_power == 0:
            mode_str = "uniform"
        else:
            mode_str = f"{self.ramp_mode}_power_{self.ramp_power}"
        out = {
            "total_steps": self.total_steps,
            "total_poison_samples": self.total_poison_global,
            "n_gpus": self.n_gpus,
            "batch_per_gpu": self.batch_per_gpu,
            "poison_per_gpu": self.poison_per_gpu,
            "final_dosage_pct": round(self.final_dosage_pct, 4),
            "checkpoint_every": self.checkpoint_every,
            "ramp_power": self.ramp_power,
            "ramp_mode": mode_str,
            "num_milestones": len(self._checkpoint_steps),
        }
        if self.piecewise_knots is not None:
            out["piecewise_knots"] = [
                {"step": step, "dosage_pct": round(pct, 6)}
                for step, pct in self.piecewise_knots
            ]
        return out
