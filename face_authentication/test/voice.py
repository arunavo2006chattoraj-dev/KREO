
import os
import re
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq

import main
import arduino_controller
from voice_visualizer import visualizer 


import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


STT_MODEL = "whisper-large-v3-turbo"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"

SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 6
MICROPHONE_DEVICE = 1

RESPONSE_DIR = Path(__file__).resolve().parent / "responses"
RECORDING_DIR = Path(__file__).resolve().parent / "recordings"

RESPONSE_DIR.mkdir(parents=True, exist_ok=True)
RECORDING_DIR.mkdir(parents=True, exist_ok=True)

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

awake = False
running = True
authenticated = False
pickup_completed = False
angles_configured = False

RESPONSES = {
    "listening": "I am listening",
    "look": "Please look at the camera",
    "auth_success": "Authentication successful",
    "auth_failed": "Authentication failed",
    "pickup_success": "Pickup successful",
    "pickup_denied": "Pickup denied",
    "pickup_failed": "Pickup failed",
    "drop_success": "Delivery successful",
    "drop_denied": "Drop denied",
    "drop_failed": "Delivery failed",
    "shutdown": "System shutting down",
    "repeat": "Please repeat the command",
}


# --------------------------------------------------
# TEXT TO SPEECH
# --------------------------------------------------

def response_filename(text):
    filename = re.sub(
        r"[^a-z0-9]+", "_", text.lower()
    ).strip("_")

    return RESPONSE_DIR / f"{filename}.wav"


def speak(text):
    """Speak using cached audio, Groq TTS, or local pyttsx3."""

    audio_path = response_filename(text)

    if audio_path.exists():
        try:
            audio, rate = sf.read(
                str(audio_path),
                dtype="float32"
            )
            sd.play(audio, rate)
            sd.wait()
            return
        except Exception as exc:
            print("Cached audio playback failed:", exc)

    if client is not None:
        try:
            response = client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=text,
                response_format="wav",
            )

            response.write_to_file(str(audio_path))

            audio, rate = sf.read(
                str(audio_path),
                dtype="float32"
            )

            sd.play(audio, rate)
            sd.wait()
            return

        except Exception as exc:
            print("Groq TTS unavailable:", exc)

    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    except Exception as exc:
        print("Local TTS also failed:", exc)
        print("Robot response:", text)


def prepare_responses():
    """Pre-cache common responses when Groq TTS is available."""

    if client is None:
        print("Groq TTS unavailable. Using local voice fallback.")
        return

    print("\nPreparing spoken responses...")

    for text in RESPONSES.values():
        audio_path = response_filename(text)

        if audio_path.exists():
            continue

        try:
            response = client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=text,
                response_format="wav",
            )

            response.write_to_file(str(audio_path))
            print("Cached:", text)

        except Exception as exc:
            print(f"Could not cache '{text}': {exc}")


# --------------------------------------------------
# SPEECH TO TEXT
# --------------------------------------------------


def record_audio():
    print(f"Speak now for {RECORD_SECONDS} seconds...")
    visualizer.set_status("LISTENING • SPEAK NOW")

    chunks = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Microphone status:", status)

        # Copy audio because the input buffer is reused by PortAudio.
        chunk = indata.copy()
        chunks.append(chunk)

        # Drive the animation using the same audio being recorded.
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        visualizer.set_level(rms)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=MICROPHONE_DEVICE,
            callback=audio_callback,
            blocksize=1024,
        ):
            sd.sleep(int(RECORD_SECONDS * 1000))

    except Exception as exc:
        visualizer.set_level(0)
        visualizer.set_status("MICROPHONE ERROR")
        print("Microphone recording failed:", exc)
        return None

    finally:
        visualizer.set_level(0)

    if not chunks:
        print("No microphone audio captured.")
        visualizer.set_status("NO AUDIO")
        return None

    audio = np.concatenate(chunks, axis=0)

    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio ** 2)))

    print(f"Audio peak: {peak:.4f} | RMS: {rms:.4f}")

    if peak < 0.001:
        print("Audio is nearly silent. Check your microphone.")
        visualizer.set_status("NO SPEECH DETECTED")
        return None

    if peak < 0.2:
        gain = min(3.0, 0.7 / peak)
        audio = np.clip(audio * gain, -1.0, 1.0)

    audio_path = RECORDING_DIR / "latest_recording.wav"

    try:
        sf.write(
            str(audio_path),
            audio,
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        visualizer.set_status("PROCESSING • KREO")
        return audio_path

    except Exception as exc:
        print("Could not save recording:", exc)
        visualizer.set_status("RECORDING ERROR")
        return None



def transcribe_audio(audio_path):
    if audio_path is None or client is None:
        return ""

    try:
        audio_path = Path(audio_path)

        with audio_path.open("rb") as audio_file:
            result = client.audio.transcriptions.create(
                file=(audio_path.name, audio_file.read()),
                model=STT_MODEL,
                language="en",
                temperature=0.0,
                response_format="text",
                prompt=(
                    "Robot voice commands. "
                    "Wake phrases: Kreo wake up, wake up Kreo, "
                    "hello Kreo, Kreo start. "
                    "Commands: kreo authenticate, kreo authenticate me, "
                    "kreo pick, kreo pickup, kreo pick up, kreo grab, kreo collect, "
                    "kreo drop, kreo place, kreo release, kreo sleep, kreo shutdown, "
                    "kreo shut down."
                ),
            )

        if isinstance(result, str):
            text = result.strip()
        else:
            text = getattr(result, "text", str(result)).strip()

        print("Recognized:", text)
        return text.lower()

    except Exception as exc:
        print("Groq transcription failed:", exc)
        return ""


# --------------------------------------------------
# COMMAND MATCHING
# --------------------------------------------------

def contains_phrase(text, phrases):
    for phrase in phrases:
        pattern = r"\b" + re.escape(phrase) + r"\b"

        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    return False


def remove_phrases(text, phrases):
    """Remove recognized wake phrases before parsing the command."""

    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = r"\b" + re.escape(phrase) + r"\b"
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip(" ,.!?")


# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

def authenticate():
    global authenticated

    speak(RESPONSES["look"])

    try:
        result = main.authenticate_face()
        authenticated = result is True

    except Exception as exc:
        print("Authentication function issue:", exc)
        authenticated = False

    # Keep the existing main.py state synchronized.
    if hasattr(main, "authenticated"):
        main.authenticated = authenticated

    if authenticated:
        speak(RESPONSES["auth_success"])
    else:
        speak(RESPONSES["auth_failed"])

    return authenticated


# --------------------------------------------------
# MANUAL ANGLE CONFIGURATION
# --------------------------------------------------

def configure_angles():
    """Read target base and shoulder angles from the console."""

    global angles_configured

    while True:
        try:
            value = input(
                "\nEnter base,shoulder angles (example 69,70): "
            ).strip()

            parts = value.split(",")

            if len(parts) != 2:
                print("Format must be base,shoulder. Example: 69,70")
                continue

            base_angle = int(parts[0].strip())
            shoulder_angle = int(parts[1].strip())

            if not (
                0 <= base_angle <= 180
                and 0 <= shoulder_angle <= 180
            ):
                print("Angles must be between 0 and 180.")
                continue

            if arduino_controller.send_angles(
                base_angle, shoulder_angle
            ):
                angles_configured = True
                print(
                    f"Target angles configured: "
                    f"{base_angle}, {shoulder_angle}"
                )
                return True

            print("Arduino did not confirm the angles.")
            print("Check the serial connection and Arduino firmware.")
            return False

        except ValueError:
            print("Enter whole-number angles, such as 69,70.")

        except (EOFError, KeyboardInterrupt):
            print("\nAngle configuration cancelled.")
            return False


# --------------------------------------------------
# VOICE COMMAND HANDLER
# --------------------------------------------------


def handle_command(text):
    global awake, running, authenticated, pickup_completed

    text = (text or "").strip().lower()

    if not text:
        print("No speech was recognized.")
        return

    wake_phrases = [
        "kreo wake up",
        "wake up kreo",
        "hello kreo",
        "kreo start",
    ]

    shutdown_phrases = [
        "kreo sleep",
        "kreo shutdown",
        "kreo shut down",
    ]

    auth_phrases = [
        "kreo authenticate",
        "kreo authentication",
    ]

    pick_phrases = [
        "kreo pick up",
        "kreo pickup",
        "kreo pick",
        "kreo grab",
        "kreo collect",
    ]

    drop_phrases = [
        "kreo drop",
        "kreo place",
        "kreo release",
    ]

    # Wake up Kreo.
    wake_detected = contains_phrase(text, wake_phrases)

    if wake_detected:
        text = remove_phrases(text, wake_phrases).strip()

        if not awake:
            try:
                if main.startup() is not True:
                    print("Robot startup failed.")
                    speak("Robot startup failed.")
                    return
            except Exception as exc:
                print("Startup error:", exc)
                speak("Robot startup failed.")
                return

            awake = True
            authenticated = False

            if hasattr(main, "authenticated"):
                main.authenticated = False

            speak(RESPONSES["listening"])

        # Allow a command in the same recording as the wake phrase.
        if not text:
            return

    elif not awake:
        print("Wake phrase not detected. Command ignored.")
        return

    # Sleep / shutdown.
    if contains_phrase(text, shutdown_phrases):
        speak(RESPONSES["shutdown"])

        try:
            main.sleep_robot()
        except Exception as exc:
            print("Sleep function issue:", exc)

        awake = False
        authenticated = False

        if hasattr(main, "authenticated"):
            main.authenticated = False

        # Do not reset pickup_completed here.
        # The physical arm may still be holding an object.
        return

    # Face authentication.
    if contains_phrase(text, auth_phrases):
        if not awake:
            print("Wake Kreo before authentication.")
            return

        authenticate()
        return

    # Pick up an object.
    if contains_phrase(text, pick_phrases):
        if not authenticated or not getattr(
            main, "authenticated", authenticated
        ):
            speak(RESPONSES["pickup_denied"])
            return

        if not angles_configured:
            print("Set the target angles before pickup.")
            speak("Please configure the target angles first.")
            return

        if pickup_completed:
            print("An object is already marked as picked up.")
            speak("I am already holding the object.")
            return

        try:
            success = arduino_controller.pick_object()
        except Exception as exc:
            print("Pickup communication error:", exc)
            success = False

        if success:
            pickup_completed = True
            speak(RESPONSES["pickup_success"])
        else:
            # A timeout does not prove the pickup failed physically.
            print(
                "Pickup was not confirmed. Check the arm and "
                "object before retrying."
            )
            speak(RESPONSES["pickup_failed"])

        return

    # Drop the object and wait for return-home completion.
    if contains_phrase(text, drop_phrases):
        if not authenticated or not getattr(
            main, "authenticated", authenticated
        ):
            speak(RESPONSES["drop_denied"])
            return

        if not pickup_completed:
            speak(RESPONSES["drop_denied"])
            print("Drop ignored because no pickup is recorded.")
            return

        try:
            success = arduino_controller.drop_object()
        except Exception as exc:
            print("Delivery communication error:", exc)
            success = False

        if success:
            pickup_completed = False
            speak(RESPONSES["drop_success"])
        else:
            # Preserve the state: the arm might still be holding
            # the object, or the return-home sequence may be incomplete.
            print(
                "Delivery was not fully confirmed. Check the arm "
                "before issuing another movement command."
            )
            speak(RESPONSES["drop_failed"])

        return

    print("No supported command found.")



# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------

def main_loop():
    global running

    if client is None:
        print("ERROR: Set the GROQ_API_KEY environment variable.")
        return

    print("Testing Groq API...")

    try:
        client.models.list()
        print("Groq API connection successful.")

    except Exception as exc:
        print("Groq API connection failed:", exc)
        print("Check your API key and internet connection.")
        return

    # Connect to Arduino before angle configuration.
    if not arduino_controller.connect():
        print("Cannot continue without an Arduino connection.")
        return

    try:
        prepare_responses()

        visualizer.start()
        visualizer.set_status("KREO • READY")

        print("\nRobot voice control is ready.")
        print("Wake phrases: 'Kreo wake up' or 'Wake up Kreo'.")
        print("Say 'authenticate me' to authenticate.")
        print("Say 'pick up' to collect the object.")
        print("Say 'drop' to deliver the object.")
        print("Say 'sleep' to enter the sleeping state.")
        print("Type q at the console to exit.")

        if not configure_angles():
            print("Target angles were not configured.")
            print("Voice control will not start.")
            return

        while running:
            try:
                user_input = input(
                    "\nPress Enter to speak (q to quit): "
                )

                if user_input.strip().lower() == "q":
                    break

                audio_path = record_audio()

                if audio_path is None:
                    continue

                text = transcribe_audio(audio_path)
                handle_command(text)

            except KeyboardInterrupt:
                break

            except Exception as exc:
                print("Unexpected error:", exc)
                time.sleep(1)

    finally:
        arduino_controller.disconnect()
        print("Voice control stopped.")


if __name__ == "__main__":
    main_loop()
