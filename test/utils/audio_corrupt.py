"""
Utility: inject silent segments into an audio file using ffmpeg.

Usage:
    inject_silences(src_path, silences, out_path)

    src_path  – path to the source audio file
    silences  – list of (start_sec, duration_sec) tuples; must be in
                ascending order of start_sec and must not overlap
    out_path  – path where the corrupted audio will be written
"""
import subprocess
from pathlib import Path


def inject_silences(src_path: str | Path, silences: list[tuple[float, float]], out_path: str | Path) -> None:
    """Splice silent segments into src_path and write the result to out_path.

    Each entry in silences is (start_sec, duration_sec).  The function builds
    an ffmpeg filter graph that:
      1. Splits the source into segments around each silence window
      2. Generates a silent PCM segment for each gap
      3. Concatenates everything in order
    """
    src = Path(src_path)
    out = Path(out_path)

    if not silences:
        # Nothing to inject — just copy
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), str(out)],
            check=True, capture_output=True,
        )
        return

    # Sort silences ascending so we can build cut-points correctly
    silences = sorted(silences, key=lambda x: x[0])

    # Build the ffmpeg complex filter
    # Strategy: for each silence at (t, d) we cut the source into a segment
    # ending at t, insert d seconds of silence, then continue.
    # We use the concat filter to stitch everything together.

    filter_parts = []
    concat_inputs = []
    segment_idx = 0
    prev_end = 0.0

    for start_sec, duration_sec in silences:
        # Audio segment from prev_end → start_sec
        filter_parts.append(
            f"[0:a]atrim=start={prev_end}:end={start_sec},asetpts=PTS-STARTPTS[seg{segment_idx}]"
        )
        concat_inputs.append(f"[seg{segment_idx}]")
        segment_idx += 1

        # Silent segment of duration_sec
        filter_parts.append(
            f"aevalsrc=0:d={duration_sec}[sil{segment_idx}]"
        )
        concat_inputs.append(f"[sil{segment_idx}]")
        segment_idx += 1

        prev_end = start_sec + duration_sec  # skip the replaced window (no gap needed)

    # Final segment from prev_end → end of file
    filter_parts.append(
        f"[0:a]atrim=start={prev_end},asetpts=PTS-STARTPTS[seg{segment_idx}]"
    )
    concat_inputs.append(f"[seg{segment_idx}]")
    segment_idx += 1

    n_segments = len(concat_inputs)
    concat_str = "".join(concat_inputs) + f"concat=n={n_segments}:v=0:a=1[out]"

    filter_complex = ";".join(filter_parts) + ";" + concat_str

    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
