
import sounddevice as sd
import numpy as np

print(sd.query_devices())

device = sd.default.device[0]
print("Testing input device:", device)

audio = sd.rec(
    int(5 * 16000),
    samplerate=16000,
    channels=1,
    dtype="float32",
    device=device
)

print("Speak for 5 seconds...")
sd.wait()

print("Peak:", float(np.max(np.abs(audio))))
print("RMS:", float(np.sqrt(np.mean(audio ** 2))))
