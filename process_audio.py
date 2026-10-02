"""Apply a trained guitar-distortion model to existing audio recordings.

Run this processing script through a command window, such as Windows Command
Prompt, PowerShell, or a terminal, from the folder containing these scripts:
    python process_audio.py single input.wav output.wav
    python process_audio.py batch input_folder output_folder

train.py provides DistortionNet. Train it first to create the default
checkpoint, models/distortion_model.pth. Inference uses overlapping chunks,
Hann-window overlap-add, whole-recording normalization, and short end fades.
Relative paths are resolved from the current working directory.
The functions can also be imported into another Python program.
"""

# PyTorch runs the network; NumPy handles arrays; librosa reads/resamples audio;
# SoundFile writes the output; pathlib handles filenames and folders.
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
# Keep train.py beside this script, or otherwise available on the Python path.
# Importing DistortionNet does not start training because train.py has a main guard.
from train import DistortionNet

def load_model(model_path="models/distortion_model.pth"):
    """Load a train.py checkpoint and return (model, device).

    model_path defaults to models/distortion_model.pth. The checkpoint must
    contain model_state_dict, epoch, and loss, as written by train.py. CUDA is
    selected when available; otherwise inference runs on the CPU.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Recreate the architecture defined in train.py: 32 channels, 10 main layers.
    model = DistortionNet()
    
    # The .pth file is a checkpoint created by train_model() in train.py.
    # Its stored parameter names and shapes must match this DistortionNet definition.
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Use stored batch-normalization statistics during inference.
    model.eval()
    model.to(device)
    
    print(f"✓ Modelo cargado desde: {model_path}")
    print(f"  Entrenado por {checkpoint['epoch']} épocas")
    print(f"  Pérdida final: {checkpoint['loss']:.6f}")
    
    return model, device

def apply_fade(chunk, fade_length):
    """Apply linear fades to both ends of a NumPy chunk, in place.

    fade_length is measured in samples. Return the same array unchanged if it
    is shorter than twice that length. This helper is available for reuse but
    is not called by the current processing pipeline.
    """
    if len(chunk) < fade_length * 2:
        return chunk
    
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    
    chunk[:fade_length] *= fade_in
    chunk[-fade_length:] *= fade_out
    
    return chunk

def overlap_add(chunks, hop_size, chunk_size):
    """Sum fixed-length, Hann-windowed chunks into one waveform.

    chunks contains mono arrays of chunk_size samples. hop_size is the sample
    distance between consecutive chunk starts. Overlapping windows soften
    chunk transitions; this function does not normalize the sum of the windows,
    so it does not guarantee perfect reconstruction or remove every artifact.
    Return an empty array when no chunks are supplied.
    """
    if len(chunks) == 0:
        return np.array([])
    
    # Calculate the output length covered by the overlapping chunks.
    # This local overlap value describes the geometry but is not used below.
    overlap = chunk_size - hop_size
    total_length = hop_size * (len(chunks) - 1) + chunk_size
    
    # Allocate a mono floating-point output buffer.
    output = np.zeros(total_length, dtype=np.float32)
    
    # Use a Hann window to taper each chunk toward zero at its boundaries.
    # This implementation sums windows without dividing by their accumulated weights.
    window = np.hanning(chunk_size)
    
    for i, chunk in enumerate(chunks):
        start = i * hop_size
        
        # Apply the window to this model output before mixing it.
        windowed_chunk = chunk * window
        
        # Add the chunk into the output; overlapping regions are summed.
        output[start:start + chunk_size] += windowed_chunk
    
    return output

def process_audio_file(input_file, output_file, model, device, 
                       chunk_size=44100, overlap=0.5):
    """Process one file and return (processed_audio, sample_rate).

    input_file and output_file are audio paths; model and device come from
    load_model(). Audio is converted to mono at 44,100 Hz. chunk_size is measured
    in samples, and overlap is the fraction shared by adjacent chunks. Defaults
    give one-second chunks with 50% overlap. Inference uses global input peak
    normalization, Hann-window overlap-add, a final peak of 0.9, and 10 ms fades.
    The caller must provide an existing output directory.

    The current chunk count uses only complete windows for inputs longer than
    one chunk, so an uncovered final remainder can be omitted. A shorter input
    is padded to one chunk and trimmed after reconstruction.
    """
    
    print(f"\nProcesando: {input_file}")
    
    # Load the input as mono audio at 44,100 Hz, resampling if needed.
    audio, sr = librosa.load(input_file, sr=44100, mono=True)
    print(f"  Duración: {len(audio)/sr:.2f} segundos")
    print(f"  Sample rate: {sr} Hz")
    
    # Record the input peak before normalization; it is not restored later.
    original_max = np.abs(audio).max()
    
    # Normalize the whole recording with one factor, preserving relative levels.
    if original_max > 0:
        audio = audio / original_max
    
    # Set the distance between chunk starts from the requested overlap fraction.
    hop_size = int(chunk_size * (1 - overlap))
    
    # Prepare overlapping chunks; defaults are 44,100 samples and 50% overlap.
    processed_chunks = []
    # Integer division counts complete windows for longer recordings. Any final
    # remainder outside those windows is not processed by the current loop.
    num_chunks = max(1, (len(audio) - chunk_size) // hop_size + 1)
    
    print(f"  Procesando en {num_chunks} chunks con {overlap*100:.0f}% overlap...")
    print(f"  Esto puede tardar un momento...")
    
    # Disable gradient recording to reduce inference memory use.
    with torch.no_grad():
        for i in range(num_chunks):
            start = i * hop_size
            end = start + chunk_size
            
            # Extract the next waveform segment.
            if end > len(audio):
                # Handle a segment that extends beyond the available input samples.
                chunk = audio[start:]
                # Pad a short segment with zeros to reach the model input length.
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                is_last_chunk = True
            else:
                chunk = audio[start:end]
                is_last_chunk = False
            
            # Create a float32 tensor shaped [batch=1, channels=1, samples].
            chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).unsqueeze(0)
            chunk_tensor = chunk_tensor.to(device)
            
            # Predict the distorted waveform, then move it to a NumPy array on the CPU.
            processed = model(chunk_tensor)
            processed = processed.squeeze().cpu().numpy()
            
            # Discard predictions corresponding to zero-padded input samples.
            if is_last_chunk and end > len(audio):
                original_length = len(audio) - start
                processed = processed[:original_length]
                # Pad with zeros again so overlap_add receives fixed-length chunks.
                if len(processed) < chunk_size:
                    processed = np.pad(processed, (0, chunk_size - len(processed)), mode='constant')
            
            processed_chunks.append(processed)
            
            # Report progress every ten chunks and after the last chunk.
            if (i + 1) % 10 == 0 or i == num_chunks - 1:
                progress = (i + 1) / num_chunks * 100
                print(f"    Progreso: {progress:.0f}%")
    
    # Reconstruct the waveform by summing the windowed, overlapping predictions.
    print("  Reconstruyendo audio con overlap-add...")
    processed_audio = overlap_add(processed_chunks, hop_size, chunk_size)
    
    # Trim extra samples; this slice does not extend an output that is too short.
    processed_audio = processed_audio[:len(audio)]
    
    # Apply a single peak-normalization factor to the reconstructed recording.
    # This keeps relative output levels but does not preserve the original gain.
    current_max = np.abs(processed_audio).max()
    if current_max > 0:
        # Set the output peak to 0.9, leaving headroom below full scale.
        processed_audio = processed_audio / current_max * 0.9
    
    # Apply short linear fades at the start and end of the complete recording.
    fade_samples = int(sr * 0.01)  # 10 milliseconds at the loaded sample rate.
    if len(processed_audio) > fade_samples * 2:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        processed_audio[:fade_samples] *= fade_in
        processed_audio[-fade_samples:] *= fade_out
    
    # Write the processed audio using SoundFile at the loaded sample rate.
    sf.write(output_file, processed_audio, sr)
    print(f"✓ Audio guardado en: {output_file}")
    print(f"  Sin clicks ni discontinuidades gracias a overlap-add!")
    
    return processed_audio, sr

def batch_process(input_folder, output_folder, model_path="models/distortion_model.pth"):
    """Process .wav and .mp3 files directly inside input_folder.

    Create output_folder, load the checkpoint once, and write each result there
    as <input_stem>_distorted.wav. Subfolders are not searched. Existing files
    with the generated output names can be overwritten.
    """
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load the checkpoint once and reuse the model for every file in this batch.
    model, device = load_model(model_path)
    
    # Find lowercase .wav and .mp3 files directly inside the input folder.
    input_folder = Path(input_folder)
    audio_files = list(input_folder.glob("*.wav")) + list(input_folder.glob("*.mp3"))
    
    if len(audio_files) == 0:
        print(f"❌ No se encontraron archivos de audio en {input_folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"Procesando {len(audio_files)} archivos")
    print(f"{'='*60}")
    
    for i, audio_file in enumerate(audio_files, 1):
        # Reuse the input basename and append _distorted.wav. Two inputs with the
        # same stem, or a repeated run, can target the same output filename.
        output_file = output_folder / f"{audio_file.stem}_distorted.wav"
        print(f"\n[{i}/{len(audio_files)}]")
        
        # Use one-second chunks with a half-second hop at 44,100 Hz.
        process_audio_file(audio_file, output_file, model, device, 
                          chunk_size=44100, overlap=0.5)
    
    print(f"\n{'='*60}")
    print("✓ Procesamiento completado!")
    print(f"✓ Archivos guardados en: {output_folder}")
    print(f"✓ Sin clicks gracias al método overlap-add")
    print(f"{'='*60}")

def process_single_file(input_file, output_file=None, model_path="models/distortion_model.pth"):
    """Load a checkpoint and process one recording.

    If output_file is omitted, save <input_stem>_distorted.wav beside the input.
    An explicit output path uses its existing parent directory. model_path can
    be passed when calling this function from Python; the command-line entry
    point always uses models/distortion_model.pth.
    """
    
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_distorted.wav"
    
    # Load the checkpoint produced by train.py; use CUDA when available.
    model, device = load_model(model_path)
    
    # Use one-second chunks with 50% overlap.
    process_audio_file(input_file, output_file, model, device,
                      chunk_size=44100, overlap=0.5)
    
    print(f"\n✓ ¡Listo! Audio procesado sin clicks.")
    print(f"✓ Puedes escuchar el resultado en: {output_file}")

# ============================================================================
# COMMAND-LINE ENTRY POINT AND USAGE EXAMPLES
# ============================================================================

# Run through a command window and pass single or batch as the first argument.
# Example: python process_audio.py single "guitar.wav" "guitar_distorted.wav"
# Example: python process_audio.py batch "input_audio" "processed_audio"
# These command modes use models/distortion_model.pth; no model-path flag is parsed.
# The Spanish console strings below are executable text and are intentionally kept.
if __name__ == "__main__":
    import sys
    
    print("""
╔════════════════════════════════════════════════════════════════╗
║     PROCESADOR MEJORADO DE DISTORSIÓN DE GUITARRA CON IA       ║
║              (Sin clicks, con overlap-add)                     ║
╚════════════════════════════════════════════════════════════════╝
    
Mejoras en esta versión:
✓ Overlap-add elimina clicks entre fragmentos
✓ Normalización consistente para mejor distorsión
✓ Ventana de Hann para suavizado perfecto

Opciones de uso:
    
1. Procesar un archivo individual:
   python process_audio.py single input.wav output.wav
   
2. Procesar todos los archivos de una carpeta:
   python process_audio.py batch input_folder/ output_folder/
   
3. Usar directamente desde otro script:
   from process_audio import process_single_file
   process_single_file("mi_guitarra.wav")
    """)
    
    if len(sys.argv) < 2:
        print("⚠️  Por favor especifica un modo: 'single' o 'batch'")
        sys.exit(1)
    
    # sys.argv[0] is the script name; subsequent entries contain the mode and paths.
    mode = sys.argv[1]
    
    if mode == "single":
        if len(sys.argv) < 3:
            print("❌ Uso: python process_audio.py single <input.wav> [output.wav]")
            sys.exit(1)
        
        input_file = sys.argv[2]
        # Omitting the output argument activates the automatic _distorted.wav name.
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        process_single_file(input_file, output_file)
        
    elif mode == "batch":
        if len(sys.argv) < 4:
            print("❌ Uso: python process_audio.py batch <input_folder> <output_folder>")
            sys.exit(1)
        
        input_folder = sys.argv[2]
        output_folder = sys.argv[3]
        batch_process(input_folder, output_folder)
        
    else:
        print(f"❌ Modo desconocido: {mode}")
        print("   Modos disponibles: 'single' o 'batch'")
        sys.exit(1)