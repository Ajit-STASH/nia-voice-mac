#!/usr/bin/env python3
"""
Nia Voice Mac — terminal thin client for Nia Hub.

Architecture (thin client):
    Mac mic → WAV → Hub (STT + LLM + TTS) → MP3 stream → Mac speaker
    All AI processing runs on the Mac Mini Hub. This client only captures
    audio, streams it to the hub, and plays the MP3 response back.

Usage:
    nia-mac                 Voice mode  (press Enter to speak)
    nia-mac --text          Text mode   (type commands, no mic)
    nia-mac --hub URL       Override NIA_HUB_URL
    nia-mac --key KEY       Override NIA_API_KEY

Controls (voice mode):
    Enter       → start recording (auto-stops on silence)
    t <text>    → send text inline, plays audio response  [not yet: see below]
    r / reset   → clear conversation history
    q / quit    → exit

NOTE: The hub's /voice endpoint always requires WAV audio for STT. A future
enhancement could add X-Text header support to the hub to skip STT when text
is supplied directly. For now, --text mode uses the hub's chat API (Ollama +
tool calls) and prints the reply; voice is the primary path for audio playback.
"""

import argparse
import io
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
import wave
from pathlib import Path

# ── Load .env before importing config ────────────────────────────────────────

def _load_env() -> str:
    """Load .env from cwd or ~/.nia-voice-mac.env. Returns path loaded or ''."""
    try:
        from dotenv import load_dotenv
        for path in [Path(".env"), Path.home() / ".nia-voice-mac.env"]:
            if path.exists():
                load_dotenv(path)
                return str(path)
    except ImportError:
        pass  # python-dotenv not installed — rely on shell env vars
    return ""

_ENV_PATH = _load_env()

import nia_voice_core.config as nia_config  # noqa: E402 — must load after dotenv
from nia_voice_core.hub import NiaHubClient        # noqa: E402
from nia_voice_core.mic import MicCapture          # noqa: E402
from nia_voice_core.wakeword import OpenWakeWordEngine  # noqa: E402

# ── Terminal colours ──────────────────────────────────────────────────────────

_R   = "\033[0m"   # reset
_B   = "\033[1m"   # bold
_DIM = "\033[2m"
_G   = "\033[92m"  # green
_C   = "\033[96m"  # cyan
_Y   = "\033[93m"  # yellow
_RED = "\033[91m"  # red


def _status(icon: str, colour: str, msg: str):
    """Overwrite current line with a coloured status message."""
    print(f"\r{colour}{_B}{icon}  {msg}{_R}          ", flush=True)


def _hr():
    print(f"{_DIM}{'─' * 50}{_R}")


# ── MP3 player discovery ──────────────────────────────────────────────────────

def _find_player() -> list[str]:
    """Return the command list for the best available streaming MP3 player."""
    # ffplay ships with ffmpeg (already installed for Kokoro TTS)
    for candidate in ["/opt/homebrew/bin/ffplay", "ffplay"]:
        try:
            subprocess.run(
                [candidate, "-version"],
                capture_output=True,
                check=True,
            )
            return [candidate, "-nodisp", "-autoexit", "-i", "pipe:0",
                    "-loglevel", "quiet"]
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    # mpv fallback
    for candidate in ["/opt/homebrew/bin/mpv", "mpv"]:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            return [candidate, "--no-video", "--no-terminal", "--no-cache", "-"]
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
    return []


# ── Main client ───────────────────────────────────────────────────────────────

class NiaMacClient:
    """
    Thin-client: records mic → POST /voice → streams MP3 back → plays.

    All STT, LLM, and TTS runs on the Nia Hub (Mac Mini). This client is
    intentionally minimal — it mirrors the Panel Pi architecture without
    any Raspberry Pi hardware.
    """

    def __init__(self, text_mode: bool = False, wake_model: str | None = None):
        self._hub         = NiaHubClient()
        self._mic         = MicCapture()           # sounddevice / CoreAudio
        self._session_id  = uuid.uuid4().hex[:12]  # per-conversation context
        self._processing  = False
        self._running     = False
        self._text_mode   = text_mode
        self._wake_model  = wake_model             # openWakeWord model name, or None
        self._wake_engine: OpenWakeWordEngine | None = None
        self._player_cmd  = _find_player()

    # ── Hub setup ─────────────────────────────────────────────────────────────

    def _connect(self):
        print(f"\n  Connecting to Nia Hub at {nia_config.NIA_HUB_URL} …")
        n = self._hub.connect_with_retry(max_retries=3)
        print(f"  {_G}✓{_R} {n} tools loaded")

        cfg    = self._hub.fetch_device_config()
        ai_cfg = self._hub.fetch_ai_config()
        NiaHubClient.apply_ai_config(ai_cfg)
        ctx = self._hub.fetch_system_context()
        NiaHubClient.apply_device_config(cfg or {}, system_context=ctx)
        self._hub.reset_conversation()

        if nia_config.NIA_ROOM:
            print(f"  {_DIM}Room: {nia_config.NIA_ROOM}{_R}")
        if ai_cfg:
            model = ai_cfg.get("llm_model", "?")
            stt   = ai_cfg.get("stt_base_url", "openai")
            tts   = ai_cfg.get("tts_base_url", "openai")
            print(f"  {_DIM}LLM: {model} | STT: {stt} | TTS: {tts}{_R}")

    # ── Audio playback ────────────────────────────────────────────────────────

    def _start_player(self) -> "subprocess.Popen | None":
        if not self._player_cmd:
            return None
        return subprocess.Popen(
            self._player_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # ── Wake word ─────────────────────────────────────────────────────────────

    def _on_wake_detected(self):
        """Called by OpenWakeWordEngine when the wake word fires."""
        if self._processing:
            # Already busy — resume listening immediately
            if self._wake_engine:
                self._wake_engine.resume()
            return
        # Run voice pipeline in a thread; engine stays paused until done
        def _run():
            self._run_voice()
            if self._wake_engine:
                self._wake_engine.resume()
        threading.Thread(target=_run, daemon=True).start()

    # ── Voice pipeline ────────────────────────────────────────────────────────

    def _run_voice(self):
        """Record → stream to hub (STT+LLM+TTS) → play MP3."""
        self._processing = True
        try:
            _status("🎙", _G, "Listening… (speak now, stops on silence)")

            wav_bytes = self._mic.record(auto_stop=True)

            # Reject very short recordings (< ~0.1 s of audio after WAV header)
            if len(wav_bytes) < 2500:
                _status("💤", _DIM, "Nothing captured")
                return

            self._run_pipeline(wav_bytes)

        except Exception as e:
            _status("❌", _RED, f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._processing = False
            self._prompt()

    def _run_pipeline(self, wav_bytes: bytes):
        """POST WAV to hub, stream MP3 response to player."""
        _status("⏳", _Y, "Thinking (hub: STT → LLM → TTS)…")

        player_proc = None
        first_chunk  = True

        def _on_chunk(chunk: bytes):
            nonlocal player_proc, first_chunk
            if first_chunk:
                first_chunk = False
                _status("🔊", _C, "Speaking…")
                player_proc = self._start_player()
                if player_proc is None:
                    print(f"\n  {_RED}⚠  No MP3 player found."
                          f" Run: brew install ffmpeg{_R}")
            if player_proc and player_proc.stdin and not player_proc.stdin.closed:
                try:
                    player_proc.stdin.write(chunk)
                except BrokenPipeError:
                    pass

        try:
            transcript, reply = self._hub.voice_pipeline(
                wav_bytes,
                session_id=self._session_id,
                on_audio_chunk=_on_chunk,
            )
        finally:
            # Always close player gracefully
            if player_proc:
                try:
                    player_proc.stdin.close()
                except Exception:
                    pass
                try:
                    player_proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    player_proc.kill()

        # Print transcript + reply
        print()
        if transcript:
            print(f"  {_DIM}You:{_R} {transcript}")
            print(f"  {_C}{_B}Nia:{_R} {reply}")
        else:
            _status("💤", _DIM, "Nothing transcribed (silence?)")
        print()

    # ── Text pipeline (--text mode) ───────────────────────────────────────────

    def _run_text(self, text: str):
        """
        Send text directly to the hub's LLM (Ollama) without STT.
        Prints the reply. Audio playback is available in voice mode only
        (the /voice endpoint requires WAV for STT; add X-Text hub support later).
        """
        self._processing = True
        try:
            print(f"\n  {_DIM}→ {text}{_R}")
            _status("⏳", _Y, "Thinking (hub: LLM + tool calls)…")

            reply = self._hub.chat(text)

            print()
            print(f"  {_C}{_B}Nia:{_R} {reply}")
            print()

        except Exception as e:
            _status("❌", _RED, f"Error: {e}")
        finally:
            self._processing = False
            self._prompt()

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _prompt(self):
        if self._text_mode:
            print(f"  {_DIM}> {_R}", end="", flush=True)
        elif self._wake_engine:
            print(
                f"  {_DIM}Say '{self._wake_model}' to speak  "
                f"[Enter] manual  [r] reset  [q] quit{_R}"
            )
        else:
            print(
                f"  {_DIM}[Enter] speak  "
                f"[r] reset  "
                f"[t <text>] inline text  "
                f"[q] quit{_R}"
            )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Connect to hub and run the interactive loop."""
        print(f"\n{_B}{'=' * 50}")
        print("  Nia Voice Mac  v1.0")
        print(f"{'=' * 50}{_R}")

        if _ENV_PATH:
            print(f"  {_DIM}Config: {_ENV_PATH}{_R}")

        # Validate required env vars before attempting connection
        if not nia_config.NIA_HUB_URL:
            print(f"\n  {_RED}Error: NIA_HUB_URL is not set.{_R}")
            print(f"  Create a .env file (see env.example) or use --hub URL")
            return
        if not nia_config.NIA_API_KEY:
            print(f"\n  {_RED}Error: NIA_API_KEY is not set.{_R}")
            print(f"  Create a .env file (see env.example) or use --key KEY")
            return

        # Connect to hub
        try:
            self._connect()
        except Exception as e:
            print(f"\n  {_RED}Hub connection failed: {e}{_R}")
            print(f"  Is the hub running? Check NIA_HUB_URL={nia_config.NIA_HUB_URL}")
            return

        # Warn if no MP3 player
        if not self._text_mode and not self._player_cmd:
            print(f"\n  {_Y}⚠  No MP3 player found — audio won't play."
                  f" Install ffmpeg: brew install ffmpeg{_R}")

        # Start wake word engine if requested
        if self._wake_model and not self._text_mode:
            self._wake_engine = OpenWakeWordEngine(
                on_wake=self._on_wake_detected,
                model=self._wake_model,
            )
            if not self._wake_engine.start():
                self._wake_engine = None

        _hr()
        if self._wake_engine:
            mode = f"wake:{self._wake_model}"
        elif self._text_mode:
            mode = "text"
        else:
            mode = "voice"
        cert = "✓ cert pinned" if nia_config.NIA_HUB_CERT else "unverified TLS"
        print(f"  {_G}Ready!{_R}  mode={mode}  session={self._session_id[:8]}  {_DIM}({cert}){_R}")
        _hr()
        print()

        self._running = True
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())

        self._prompt()

        while self._running:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                break

            cmd   = line.strip()
            lower = cmd.lower()

            # ── Global commands ──────────────────────────────────────────────
            if lower in ("q", "quit", "exit"):
                break

            if lower in ("r", "reset"):
                self._session_id = uuid.uuid4().hex[:12]
                self._hub.reset_conversation()
                print(f"  {_G}Conversation reset.{_R}"
                      f"  New session: {self._session_id[:8]}")
                self._prompt()
                continue

            if self._processing:
                print(f"  {_Y}Still processing — please wait…{_R}")
                self._prompt()
                continue

            # ── Text injection: "t hello" (works in both modes) ──────────────
            if lower.startswith("t ") and len(cmd) > 2:
                text = cmd[2:].strip()
                if text:
                    t = threading.Thread(
                        target=self._run_text, args=(text,), daemon=True
                    )
                    t.start()
                    continue

            # ── Text mode: any input is a command ────────────────────────────
            if self._text_mode:
                if cmd:
                    t = threading.Thread(
                        target=self._run_text, args=(cmd,), daemon=True
                    )
                    t.start()
                else:
                    self._prompt()
                continue

            # ── Voice mode: Enter = record ────────────────────────────────────
            t = threading.Thread(target=self._run_voice, daemon=True)
            t.start()

        self.stop()

    def stop(self):
        self._running = False
        if self._wake_engine:
            self._wake_engine.stop()
            self._wake_engine = None
        print(f"\n  {_DIM}Goodbye!{_R}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nia Voice Mac — thin client for Nia Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--text", action="store_true",
        help="Text-only mode: type commands instead of speaking",
    )
    parser.add_argument(
        "--hub", metavar="URL",
        help="Hub URL, e.g. https://192.168.39.10:18080 (overrides NIA_HUB_URL)",
    )
    parser.add_argument(
        "--key", metavar="KEY",
        help="Hub API key (overrides NIA_API_KEY)",
    )
    parser.add_argument(
        "--cert", metavar="PATH",
        help="Path to hub TLS cert PEM for verification (overrides NIA_HUB_CERT)",
    )
    parser.add_argument(
        "--wake", metavar="MODEL", nargs="?", const="hey_jarvis",
        help="Enable wake word mode using openWakeWord. "
             "MODEL defaults to 'hey_jarvis'. "
             "Other options: alexa, hey_mycroft, hey_rhasspy. "
             "Requires: pip install openwakeword",
    )
    args = parser.parse_args()

    if args.hub:
        nia_config.NIA_HUB_URL = args.hub
    if args.key:
        nia_config.NIA_API_KEY = args.key
    if args.cert:
        nia_config.NIA_HUB_CERT = args.cert

    # Suppress sounddevice's PortAudio banner on macOS
    os.environ.setdefault("PORTAUDIO_HOSTAPI", "CoreAudio")

    client = NiaMacClient(text_mode=args.text, wake_model=args.wake)
    client.start()


if __name__ == "__main__":
    main()
