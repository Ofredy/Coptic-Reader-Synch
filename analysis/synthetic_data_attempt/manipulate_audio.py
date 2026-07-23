"""
Takes a source recording and produces pitch-preserving time-stretched variants
(2x faster / unchanged / 2x slower), then adds amplitude-only corruption to
the slowed-down ("long") variant so it resembles a real singer's dynamics over
a long held note rather than a perfectly flat signal:

  - breathing pauses: brief, smoothly-tapered dips toward silence at random
                      points (a singer running out of air mid-hold)
  - volume swell:     a slow random drift in overall loudness across the
                       whole clip (natural crescendo/decrescendo)

Both corruption effects are gain modulation only — no samples are inserted or
removed — so the exact, stretch-factor-scaled timestamps from the plan
(original timestamps x rate) stay valid. If we inserted real silence instead,
every timestamp after the insertion point would need to be shifted too.

Uses librosa's phase-vocoder time_stretch, not simple resampling — resampling
would also shift pitch, which would corrupt the pitch-contour features the
onset detector in chant_onset/ depends on.

Run:
    python analysis/synthetic_data_attempt/manipulate_audio.py
"""
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).parent
TEST_DIR = HERE.parents[1] / "test"
OUT_ROOT = HERE / "output"

# (name, source path, seconds to load from the start — None for the whole clip)
SOURCES = [
    ("psalm27_coptic", TEST_DIR / "psalm_27_english" / "coptic" / "psalm.mp3", 30.0),
    ("lord_have_mercy", TEST_DIR / "lord_have_mercy_english" / "lord_have_mercy.mp3", None),
]

# rate > 1.0 speeds up (shorter); rate < 1.0 slows down (longer)
RATES = [2.0, 1.0, 0.5]
LONG_RATE = 0.5  # which RATES entry gets breathing/swell corruption applied

SEED = 42  # change for a different random corruption realization

# Breathing pauses
PAUSE_RATE_PER_SEC = 1 / 10     # roughly one pause every ~10s, Poisson-sampled
PAUSE_MIN_DUR_SEC = 0.3
PAUSE_MAX_DUR_SEC = 0.7
PAUSE_MIN_DEPTH = 0.03           # amplitude fraction remaining at the deepest point
PAUSE_MAX_DEPTH = 0.15

# Volume swell
SWELL_WALK_STD = 0.02            # per-sample random-walk step size (pre-smoothing)
SWELL_SMOOTH_SEC = 10.0          # smoothing window — bigger = slower swells; ~one
                                  # up/down cycle per this many seconds
SWELL_MIN_GAIN = 0.6
SWELL_MAX_GAIN = 1.4

REPORT_SAMPLE_INTERVAL_SEC = 10.0  # how often the swell gain gets logged in the report


def breathing_envelope(n: int, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    envelope = np.ones(n)
    pauses = []
    duration = n / sr
    t = 0.0
    while True:
        t += rng.exponential(1 / PAUSE_RATE_PER_SEC)
        if t >= duration - PAUSE_MAX_DUR_SEC:
            break
        dur = rng.uniform(PAUSE_MIN_DUR_SEC, PAUSE_MAX_DUR_SEC)
        depth = rng.uniform(PAUSE_MIN_DEPTH, PAUSE_MAX_DEPTH)
        start = int(t * sr)
        length = int(dur * sr)
        end = min(start + length, n)
        # dip shape: 1 at both edges (smooth taper, no clicks), `depth` at center
        dip = 1.0 - (1.0 - depth) * np.hanning(end - start)
        envelope[start:end] = np.minimum(envelope[start:end], dip)
        pauses.append({"start_sec": t, "duration_sec": dur, "depth": depth})
    return envelope, pauses


SWELL_ENV_RATE_HZ = 100  # build/smooth the walk at this rate, then upsample —
                          # doing it at full audio sample rate makes the
                          # gaussian kernel (seconds x sr taps) enormous and slow


def swell_envelope(n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    n_coarse = max(int(n / sr * SWELL_ENV_RATE_HZ), 2)
    walk = np.cumsum(rng.normal(0, SWELL_WALK_STD, size=n_coarse))
    smoothed = gaussian_filter1d(walk, sigma=SWELL_SMOOTH_SEC * SWELL_ENV_RATE_HZ)
    smoothed -= smoothed.mean()
    smoothed /= np.abs(smoothed).max() + 1e-9

    coarse_t = np.linspace(0, n / sr, n_coarse)
    full_t = np.arange(n) / sr
    smoothed_full = np.interp(full_t, coarse_t, smoothed)

    gain = 1.0 + smoothed_full * (SWELL_MAX_GAIN - 1.0)
    return np.clip(gain, SWELL_MIN_GAIN, SWELL_MAX_GAIN)


def write_corruption_report(out_path: Path, sr: int, n: int, pauses: list[dict], swell: np.ndarray) -> None:
    events = []

    step_samples = int(REPORT_SAMPLE_INTERVAL_SEC * sr)
    for i in range(0, n, step_samples):
        events.append((i / sr, f"gain={swell[i]:.2f}x"))

    for p in pauses:
        events.append((
            p["start_sec"],
            f"BREATH PAUSE  +{p['duration_sec']:.2f}s  dips to {p['depth']:.0%} amplitude",
        ))

    events.sort(key=lambda e: e[0])

    lines = [f"Corruption time history for {out_path.name}", ""]
    for t, desc in events:
        lines.append(f"  {t:6.2f}s  {desc}")

    out_path.write_text("\n".join(lines) + "\n")


def add_corruption(y: np.ndarray, sr: int, rng: np.random.Generator, base_name: str, out_dir: Path) -> None:
    n = len(y)
    pause_env, pauses = breathing_envelope(n, sr, rng)
    swell = swell_envelope(n, sr, rng)
    corrupted = np.clip(y * pause_env * swell, -1.0, 1.0)

    out_path = out_dir / f"{base_name}_corrupted.wav"
    sf.write(out_path, corrupted, sr)
    print(f"  corrupted -> {out_path}")

    report_path = out_dir / f"{base_name}_corrupted.txt"
    write_corruption_report(report_path, sr, n, pauses, swell)
    print(f"  report    -> {report_path}")


def process(name: str, source: Path, duration_limit: float | None, rng: np.random.Generator, out_dir: Path) -> None:
    if not source.exists():
        raise SystemExit(f"Source audio not found: {source}")

    y, sr = librosa.load(str(source), sr=None, mono=True, duration=duration_limit)
    print(f"Loaded {len(y) / sr:.2f}s of {source.name} @ {sr}Hz")

    for rate in RATES:
        stretched = y if rate == 1.0 else librosa.effects.time_stretch(y, rate=rate)
        result_sec = len(stretched) / sr
        base_name = f"{name}_{result_sec:.0f}s"
        out_path = out_dir / f"{base_name}.wav"
        sf.write(out_path, stretched, sr)
        print(f"  rate={rate}: {result_sec:.2f}s -> {out_path}")

        if rate == LONG_RATE:
            add_corruption(stretched, sr, rng, base_name, out_dir)


def main() -> None:
    out_dir = OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    print(f"Writing this run's output to {out_dir}")

    rng = np.random.default_rng(SEED)
    for name, source, duration_limit in SOURCES:
        process(name, source, duration_limit, rng, out_dir)


if __name__ == "__main__":
    main()
