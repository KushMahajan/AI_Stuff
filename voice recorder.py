import pyaudio
import soundfile as sf
#import opuslib
import numpy as np
import os

# Audio settings
CHUNK = 1024           # Size of data chunks to read from the audio buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1           # Mono audio
RATE = 44100           # Sampling rate (44.1 kHz)
DURATION = 30           # Duration of the recording in seconds

def record_audio_to_wav(filename="output.wav", duration=DURATION):
    """Record audio from microphone and save it as a WAV file."""
    audio = pyaudio.PyAudio()

    # Open stream for recording
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    print(f"Recording audio for {duration} seconds...")
    frames = []

    # Record for the given duration
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data to a WAV file
    with sf.SoundFile(filename, mode='w', samplerate=RATE, channels=CHANNELS, subtype='PCM_16') as file:
        for frame in frames:
            file.write(np.frombuffer(frame, dtype=np.int16))

    print(f"Saved audio to {filename}")

def convert_wav_to_opus(wav_filename, opus_filename="output.opus"):
    """Convert WAV file to Opus format using opuslib."""
    if not os.path.exists(wav_filename):
        print(f"WAV file '{wav_filename}' not found!")
        return
    
    print(f"Converting {wav_filename} to {opus_filename}...")

    # Read the WAV file using soundfile
    data, samplerate = sf.read(wav_filename)

    # Initialize Opus encoder
    encoder = opuslib.Encoder(samplerate, CHANNELS, opuslib.APPLICATION_AUDIO)
    
    # Open file to write Opus
    with open(opus_filename, 'wb') as opus_file:
        for frame in data:
            # Encode each frame using the Opus encoder
            encoded_frame = encoder.encode(frame.astype('int16').tobytes(), len(frame))
            opus_file.write(encoded_frame)

    print(f"Conversion complete. Opus file saved as {opus_filename}")

def play_audio(file_path):
    """Play the audio using soundfile (for wav) or external player (for opus)."""
    if file_path.endswith('.wav'):
        # Play WAV file using soundfile (or an external library if needed)
        data, samplerate = sf.read(file_path)
        print(f"Playing WAV file: {file_path}")
        # You can use sounddevice for playback if needed, but here we just simulate playback
    else:
        # Opus files might need a player like VLC
        print(f"Opus files need an external player. Playing {file_path} using VLC.")
        os.system(f"vlc --play-and-exit {file_path}")

# Example usage
if __name__ == "__main__":
    wav_file = "output.wav"
    opus_file = "output.opus"
    
    # Record audio and save as WAV
    record_audio_to_wav(wav_file, duration=DURATION)
    
    # Convert the WAV to Opus format using opuslib
    #convert_wav_to_opus(wav_file, opus_file)
    
    # Play the Opus file (you might need VLC or another external player for Opus files)
    play_audio(wav_file)
