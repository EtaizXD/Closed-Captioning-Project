import os
import json

"""
Sentence recognition.

Supports two backends, selected via the ``WHISPER_BACKEND`` environment
variable:

    * ``local`` (default)  -- load ``faster-whisper`` in-process. Useful
      for offline development. Requires ``faster-whisper`` to be
      installed (see ``requirements-local.txt``).
    * ``remote``           -- POST the audio to a Faster Whisper FastAPI
      service (see ``server/`` in this repository). Requires
      ``WHISPER_API_URL`` and optionally ``WHISPER_API_KEY``.

Input : Audio file name (string) located in the project root.
Output: A JSON file (<basename>.json) compatible with
        stress_highlight.SentenceRecognizer, containing a list of
        segments. Each segment has:
            - text  (str): the recognized text
            - words (list): list of
              {"word": str, "start": float, "end": float, "probability": float}

The two backends produce identical JSON shapes so downstream consumers
(``stress_highlight.py``) do not need to change.
"""

# Lazy import so importing this module is cheap.
_model = None


def _available_memory_gb():
    try:
        import psutil

        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return None


def _minimum_available_memory_gb(model_size, device):
    configured_value = os.environ.get("WHISPER_MIN_AVAILABLE_GB")
    if configured_value is not None:
        try:
            return float(configured_value)
        except ValueError:
            pass

    if device != "cpu":
        return 0.0

    normalized_model = model_size.lower().replace("_", "-")
    if normalized_model.startswith("large"):
        return 8.0
    if normalized_model.startswith("medium"):
        return 4.0
    if normalized_model.startswith("small"):
        return 2.0
    return 0.0


def _get_model():
    """Load the faster-whisper model once and cache it."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        # Configurable via environment variables.
        model_size = os.environ.get("WHISPER_MODEL", "large-v3")
        device = os.environ.get("WHISPER_DEVICE", "cpu")
        compute_type = os.environ.get(
            "WHISPER_COMPUTE_TYPE", "int8" if device == "cpu" else "float16"
        )
        minimum_memory_gb = _minimum_available_memory_gb(model_size, device)
        available_memory_gb = _available_memory_gb()

        if (
            available_memory_gb is not None
            and minimum_memory_gb > 0
            and available_memory_gb < minimum_memory_gb
        ):
            raise RuntimeError(
                f"Not enough available RAM to load Faster Whisper {model_size} on {device}. "
                f"Available RAM: {available_memory_gb:.1f} GB, required: {minimum_memory_gb:.1f} GB. "
                "Use a smaller WHISPER_MODEL, use WHISPER_DEVICE=cuda, or set WHISPER_MIN_AVAILABLE_GB=0 to bypass this guard."
            )

        print(
            f"[sentence_recognition] Loading faster-whisper model "
            f"'{model_size}' (device={device}, compute_type={compute_type})..."
        )
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model


SENSITIVITY_LEVELS = ("off", "sensitive", "ultra")


def _build_transcribe_kwargs(sensitivity):
    """Return the faster-whisper ``transcribe`` kwargs for a sensitivity tier.

    Tiers:
      * ``off``       -- Whisper defaults (most conservative, fewest false
                         positives, fastest).
      * ``sensitive`` -- moderately relaxed silence/confidence thresholds
                         so quiet speech still gets emitted.
      * ``ultra``     -- pushes every knob: lowest silence threshold, very
                         tolerant log-prob/compression thresholds, fresh
                         per-chunk decoding so a single hallucination
                         cannot cascade, larger beam, shorter chunks, plus
                         a guiding initial prompt. Expect more false
                         positives (background noise transcribed) and
                         ~2-3x slower runtime.
    """
    base = dict(
        language="en",
        word_timestamps=True,
        vad_filter=False,
    )
    if sensitivity == "sensitive":
        base.update(
            no_speech_threshold=0.2,            # default 0.6
            log_prob_threshold=-1.5,            # default -1.0
            compression_ratio_threshold=2.6,    # default 2.4
            condition_on_previous_text=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )
    elif sensitivity == "ultra":
        base.update(
            no_speech_threshold=0.05,           # default 0.6 -- catch near-silent speech
            log_prob_threshold=-2.5,            # default -1.0 -- tolerate very low confidence
            compression_ratio_threshold=3.0,    # default 2.4 -- tolerate very repetitive text
            condition_on_previous_text=False,   # break hallucination cascades between chunks
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            beam_size=10,                        # default 5 -- broader search, slower
            chunk_length=15,                     # default 30 -- catch speech near chunk edges
            initial_prompt=(
                "The following is quiet, whispered, or low-energy "
                "English speech."
            ),
        )
    return base


def _normalise_sensitivity(value):
    """Coerce the sensitivity argument into one of ``SENSITIVITY_LEVELS``.

    Accepts either a string from the new API or a legacy boolean (``True``
    -> ``"sensitive"``, ``False`` -> ``"off"``) so existing callers don't
    break while we migrate.
    """
    if isinstance(value, bool):
        return "sensitive" if value else "off"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in SENSITIVITY_LEVELS:
            return v
    return "off"


def _resolve_audio_path(audio_file):
    class_directory = os.path.abspath(os.path.dirname(__file__))
    return audio_file if os.path.isabs(audio_file) else os.path.join(class_directory, audio_file)


def _write_json(file_path, output, segment_count, language):
    json_path = os.path.splitext(file_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(
        f"[sentence_recognition] Wrote {json_path} "
        f"({segment_count} segments, language={language})"
    )
    return json_path


class SentenceRecognition:
    def recognize(self, audio_file, sensitivity="off"):
        """Transcribe ``audio_file`` and write a JSON file next to it.

        Backend selection is driven by ``WHISPER_BACKEND``
        (``remote`` by default, or ``local``). The on-disk JSON schema
        is identical for both backends so callers do not need to care.

        Parameters
        ----------
        audio_file : str
            Path (absolute or relative to this module) of the input media.
        sensitivity : {"off", "sensitive", "ultra"} or bool, default ``"off"``
            Tier of relaxed thresholds. Accepts a boolean for backwards
            compatibility (``True`` == ``"sensitive"``).
        """
        file_path = _resolve_audio_path(audio_file)
        if not os.path.isfile(file_path):
            print(f"Error: File '{audio_file}' not found.")
            return

        sensitivity = _normalise_sensitivity(sensitivity)
        if sensitivity != "off":
            print(f"[sentence_recognition] sensitivity tier: {sensitivity}")

        backend = os.environ.get("WHISPER_BACKEND", "local").strip().lower()
        if backend == "local":
            return self._recognize_local(file_path, sensitivity)
        if backend == "remote":
            return self._recognize_remote(file_path, sensitivity)
        raise RuntimeError(
            f"Unknown WHISPER_BACKEND={backend!r}. Expected 'remote' or 'local'."
        )

    # ------------------------------------------------------------------
    # Local backend (faster-whisper in-process)
    # ------------------------------------------------------------------
    def _recognize_local(self, file_path, sensitivity):
        model = _get_model()
        transcribe_kwargs = _build_transcribe_kwargs(sensitivity)
        segments, info = model.transcribe(file_path, **transcribe_kwargs)

        json_segments = []
        full_text_parts = []
        for segment in segments:
            words = []
            if segment.words:
                for w in segment.words:
                    words.append(
                        {
                            "word": w.word,
                            "start": float(w.start) if w.start is not None else 0.0,
                            "end": float(w.end) if w.end is not None else 0.0,
                            "probability": float(getattr(w, "probability", 0.0) or 0.0),
                        }
                    )

            seg_dict = {
                "id": len(json_segments),
                "start": float(segment.start) if segment.start is not None else 0.0,
                "end": float(segment.end) if segment.end is not None else 0.0,
                "text": segment.text,
                "words": words,
            }
            json_segments.append(seg_dict)
            full_text_parts.append(segment.text)

        output = {
            "text": "".join(full_text_parts),
            "language": info.language,
            "segments": json_segments,
        }
        return _write_json(file_path, output, len(json_segments), info.language)

    # ------------------------------------------------------------------
    # Remote backend (FastAPI service in server/)
    # ------------------------------------------------------------------
    def _recognize_remote(self, file_path, sensitivity):
        import requests  # imported lazily so local-only setups don't need it

        api_url = os.environ.get("WHISPER_API_URL", "").strip().rstrip("/")
        if not api_url:
            raise RuntimeError(
                "WHISPER_BACKEND=remote but WHISPER_API_URL is not set. "
                "Point it at the Faster Whisper FastAPI service, "
                "e.g. http://10.80.39.41:8000"
            )
        api_key = os.environ.get("WHISPER_API_KEY", "").strip()
        try:
            timeout = float(os.environ.get("WHISPER_API_TIMEOUT", "600"))
        except ValueError:
            timeout = 600.0
        language = os.environ.get("WHISPER_LANGUAGE", "en").strip() or "en"

        endpoint = f"{api_url}/transcribe"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        print(f"[sentence_recognition] POST {endpoint} (timeout={timeout:.0f}s)")

        try:
            with open(file_path, "rb") as fh:
                files = {"file": (os.path.basename(file_path), fh, "application/octet-stream")}
                data = {"sensitivity": sensitivity, "language": language}
                response = requests.post(
                    endpoint,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=timeout,
                )
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Whisper API timed out after {timeout:.0f}s ({endpoint})"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Whisper API unreachable ({endpoint}): {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Whisper API request failed: {exc}") from exc

        if response.status_code != 200:
            body = response.text or ""
            if len(body) > 500:
                body = body[:500] + "..."
            raise RuntimeError(
                f"Whisper API returned HTTP {response.status_code}: {body.strip()}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Whisper API returned non-JSON response: {response.text[:200]!r}"
            ) from exc

        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise RuntimeError(
                "Whisper API response is missing a 'segments' list. "
                f"Got keys: {sorted(payload.keys()) if isinstance(payload, dict) else type(payload)}"
            )

        # Re-emit the same JSON shape the local backend would produce so
        # downstream consumers stay identical.
        output = {
            "text": payload.get("text", ""),
            "language": payload.get("language", language),
            "segments": segments,
        }
        return _write_json(file_path, output, len(segments), output["language"])


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Transcribe a media file with faster-whisper.",
    )
    parser.add_argument("audio_file", help="Path to the input media file")
    parser.add_argument(
        "--sensitivity",
        choices=SENSITIVITY_LEVELS,
        default="off",
        help=(
            "Transcription sensitivity tier. "
            "'off' uses Whisper defaults. "
            "'sensitive' relaxes silence/confidence thresholds to catch "
            "quiet speech. "
            "'ultra' pushes every knob (lower thresholds, larger beam, "
            "shorter chunks, no cross-chunk conditioning) for "
            "barely-audible speech; expect hallucinations and ~2-3x "
            "slower runtime."
        ),
    )
    parser.add_argument(
        "--sensitive",
        action="store_true",
        help="Deprecated alias for --sensitivity sensitive.",
    )
    args = parser.parse_args()
    if not args.audio_file:
        parser.print_usage()
        sys.exit(1)
    # ``--sensitive`` is kept as a backwards-compatible shortcut. If the
    # user passes both flags, the explicit ``--sensitivity`` wins.
    sensitivity = args.sensitivity
    if sensitivity == "off" and args.sensitive:
        sensitivity = "sensitive"
    SentenceRecognition().recognize(args.audio_file, sensitivity=sensitivity)
