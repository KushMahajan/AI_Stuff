import pyaudio
import wave
import speech_recognition as sr
import threading
import time
import numpy as np
from pydub import AudioSegment

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 30
WAVE_OUTPUT_FILENAME = "output.wav"
OPUS_OUTPUT_FILENAME = "output.opus"

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Initialize lists to store audio data and transcriptions
frames = []
transcriptions = []

# Flag to signal the end of recording
done_recording = threading.Event()

def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    print("Recording...")
    
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Finished recording.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    done_recording.set()

def transcribe_audio():
    recognizer = sr.Recognizer()
    temp_file = "temp.wav"
    
    while not done_recording.is_set() or frames:
        if frames:
            # Write current frames to a temporary file
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames[:int(RATE / CHUNK * 5)]))  # 5 seconds of audio
            
            # Transcribe the temporary file
            with sr.AudioFile(temp_file) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    transcriptions.append(text)
                    print(f"Transcribed: {text}")
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition service; {e}")
            
            # Remove the processed frames
            del frames[:int(RATE / CHUNK * 5)]
        
        time.sleep(1)

def filter_by_frequency(audio_data, low_freq, high_freq):
    # Convert audio data to numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    # Perform FFT
    fft = np.fft.fft(audio_array)
    
    # Calculate the corresponding frequencies
    freqs = np.fft.fftfreq(len(audio_array), 1/RATE)
    
    # Create a mask for the desired frequency range
    mask = (abs(freqs) > low_freq) & (abs(freqs) < high_freq)
    
    # Apply the mask to the FFT
    fft_filtered = fft * mask
    
    # Perform inverse FFT to get the filtered audio
    filtered_audio = np.fft.ifft(fft_filtered).real.astype(np.int16)
    
    return filtered_audio.tobytes()

# Start recording and transcription threads
record_thread = threading.Thread(target=record_audio)
transcribe_thread = threading.Thread(target=transcribe_audio)

record_thread.start()
transcribe_thread.start()

# Wait for recording to finish
record_thread.join()
transcribe_thread.join()

# Save the recorded audio as WAV
with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

# Convert WAV to Opus
audio = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
audio.export(OPUS_OUTPUT_FILENAME, format="opus")

print(f"Audio saved as {OPUS_OUTPUT_FILENAME}")
print("Transcriptions:")
for i, trans in enumerate(transcriptions):
    print(f"{i+1}. {trans}")

# Example of filtering transcriptions by frequency (you can adjust these values)
low_freq = 85  # Hz
high_freq = 255  # Hz

filtered_frames = [filter_by_frequency(frame, low_freq, high_freq) for frame in frames]

# You can now use these filtered_frames for transcription if needed
# This may help in distinguishing between different speakers