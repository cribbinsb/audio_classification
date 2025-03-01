import argparse
import torch
import numpy as np
import sounddevice as sd
import librosa
import torchaudio
from panns_inference import SoundEventDetection, labels
import onnx
import onnxruntime as ort

# Audio settings
SAMPLE_RATE = 32000  # Model expects 32kHz audio
DURATION = 0.5      # Process 1 second of audio at a time
BUFFER_SIZE = int(SAMPLE_RATE * DURATION)

model=None

def load_model(use_onnx):
    global model
    if use_onnx==False:
        # Load PANNs model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device {device}")
        sed = SoundEventDetection(checkpoint_path=None, device=device, interpolate_mode='nearest')
        model={"sed":sed}  
    else:
        onnx_path = "/mldata/weights/panns_model_opt.onnx"
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider= "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
        session = ort.InferenceSession(onnx_path, session_options, providers=[provider])
        model={"session":session}
                            
def classify_audio(audio_data):
    global model
    if "sed" in model:
        sed=model["sed"]
        framewise_output = sed.inference(audio_data)
        framewise_output = framewise_output[0]
        print(framewise_output.shape) 
    else:
        session=model["session"]
        # audio data needs to be some amount of 32Khz audio
        # Nx320 sample (10ms chunks long)
        # returned data is 527 probabilities (or however many classes there are)
        # for each 10ms chunk
        inputs = {"audio": audio_data}
        outputs = session.run(["framewise_output"], inputs)

        # Extract the output
        framewise_output = outputs[0] 
        framewise_output = np.squeeze(framewise_output, axis=0) # remove unneeded batch output
        print(framewise_output.shape) 

    # get top 5 classes
    classwise_output = np.max(framewise_output, axis=0) 
    idxes = np.argsort(classwise_output)[::-1]
    idxes = idxes[0:5]

    ix_to_lb = {i : label for i, label in enumerate(labels)}
    # this loops over the largest prob classes
    for idx in idxes:
        # the array here is confidence vs time, for now just use the largest confidence that occurs
        largest=max(framewise_output[:, idx])
        if largest>0.01:
            print(idx, largest, ix_to_lb[idx])

def process_microphone():
    """Captures live audio from the microphone and classifies it in real-time."""
    def callback(indata, frames, time, status):
        
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

        print(f"callback {frames} SR {orig_sr} {audio_data.size}")
        # Predict labels
        classify_audio(audio_data)


    print("Listening for real-time classification... Press Ctrl+C to stop.")
    orig_sr = sd.query_devices(sd.default.device[0], 'input')['default_samplerate']
    with sd.InputStream(callback=callback, samplerate=orig_sr, channels=1, blocksize=int(orig_sr*DURATION)):
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
    parser.add_argument('--onnx', action='store_true', help='use ONNX model')

    print(f"{len(labels)} labels\n\n")
    print(labels)
    print("\n\n")
    args = parser.parse_args()

    load_model(args.onnx)
    
    if args.mic:
        print(sd.query_devices())
        devnum = int(input("Choose input device: "))
        sd.default.device=devnum
        process_microphone()
    elif args.wav:
        process_wav_file(args.wav)
    else:
        print("Please specify either --mic for microphone input or --wav <path> for a WAV file.")
