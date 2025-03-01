import argparse
import torch
import numpy as np
import sounddevice as sd
import librosa
import torchaudio
from panns_inference import SoundEventDetection, labels

# Load PANNs model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device {device}")
sed = SoundEventDetection(checkpoint_path=None, device=device, interpolate_mode='nearest')
                                         
# Audio settings
SAMPLE_RATE = 32000  # Model expects 32kHz audio
DURATION = 1       # Process 1 second of audio at a time
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
        
def classify_audio(audio_data):
    framewise_output = sed.inference(audio_data)
    framewise_output = framewise_output[0]

    classwise_output = np.max(framewise_output, axis=0) 
    idxes = np.argsort(classwise_output)[::-1]
    idxes = idxes[0:5]

    ix_to_lb = {i : label for i, label in enumerate(labels)}
    # this loops over the largest prob classes
    for idx in idxes:
        # the array here is confidence vs time, for now just use the largest confidence that occurs
        largest=max(framewise_output[:, idx])
        if largest>0.2:
            print(idx, largest, ix_to_lb[idx])

def process_microphone():
    """Captures live audio from the microphone and classifies it in real-time."""
    def callback(indata, frames, time, status):
        print(f"callback {frames}")
        if status:
            print(status)
        
        # Get the actual sample rate of the input device
        orig_sr = sd.query_devices(sd.default.device[0], 'input')['default_samplerate']
        
        # Convert stereo to mono if necessary
        audio_data = librosa.to_mono(indata.T)
        
        # Resample to 32kHz
        audio_data = librosa.resample(audio_data, orig_sr=int(orig_sr), target_sr=SAMPLE_RATE)
        
        # Ensure proper shape
        audio_data = np.expand_dims(audio_data, axis=0)

        # Predict labels
        classify_audio(audio_data)


    print("Listening for real-time classification... Press Ctrl+C to stop.")
    with sd.InputStream(callback=callback, samplerate=48000, channels=1, blocksize=BUFFER_SIZE):
        input()

def process_wav_file(wav_file):
    """Loads and classifies a WAV file."""
    waveform, sr = torchaudio.load(wav_file)
    waveform = waveform.mean(dim=0)  # Convert to mono if stereo
    waveform = librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=SAMPLE_RATE)
    waveform = np.expand_dims(waveform, axis=0)
    classify_audio(waveform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify audio using PANNs in real-time (microphone) or from a WAV file.")
    parser.add_argument("--mic", action="store_true", help="Use microphone for real-time classification")
    parser.add_argument("--wav", type=str, help="Path to WAV file for classification")
    
    args = parser.parse_args()
    
    if args.mic:
        print(sd.query_devices())
        devnum = int(input("Choose input device: "))
        sd.default.device=devnum
        process_microphone()
    elif args.wav:
        process_wav_file(args.wav)
    else:
        print("Please specify either --mic for microphone input or --wav <path> for a WAV file.")
