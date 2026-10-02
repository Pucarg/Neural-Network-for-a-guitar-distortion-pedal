"""Inspect training WAV files before running train.py.

Run from the project folder in a command window:
    python diagnose_file.py
    python diagnose_file.py --create-test

The default input folder is data/samples. Each recording is split at its
sample midpoint into a clean reference and a distorted reference. The script
prints heuristic comparisons and saves diagnosis_<file_stem>.png plots.
The optional test generator writes data/samples/test_sample_synthetic.wav.
Diagnostics are a separate step; this script does not start training.
"""

# NumPy provides array/FFT operations, librosa loads audio and computes STFTs,
# Matplotlib saves plots, pathlib handles paths, and SoundFile writes the test WAV.
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf



def diagnose_file(filepath):
    """Inspect one training recording and save comparison plots.

    filepath is a pathlib.Path pointing to a WAV file. Return True when all
    diagnostic checks pass, or False for a failed check or a processing error.
    These signal comparisons are screening heuristics, not a definitive test
    of distortion quality or a prediction of training performance.
    """
    
    print(f"\n{'='*70}")
    print(f"ARCHIVO: {filepath.name}")
    print('='*70)
    
    try:
        # Load the file as mono audio, resampling to 44,100 Hz if needed.
        audio, sr = librosa.load(filepath, sr=44100, mono=True)
        duration = len(audio) / sr
        
        print(f"\n📊 INFORMACIÓN BÁSICA:")
        print(f"   Duración: {duration:.2f} segundos")
        print(f"   Muestras: {len(audio):,}")
        print(f"   Sample rate: {sr} Hz")
        
        # Summarize the entire waveform: range, mean, RMS, and standard deviation.
        print(f"\n📈 ESTADÍSTICAS GLOBALES:")
        print(f"   Mínimo: {audio.min():.6f}")
        print(f"   Máximo: {audio.max():.6f}")
        print(f"   Media: {audio.mean():.6f}")
        print(f"   RMS: {np.sqrt(np.mean(audio**2)):.6f}")
        print(f"   Std: {audio.std():.6f}")
        
        # Reject NaN, infinite values, or a positive peak above the fixed limit.
        if np.isnan(audio).any():
            print("   ❌ CONTIENE NaN (valores no numéricos)!!!")
            return False
        if np.isinf(audio).any():
            print("   ❌ CONTIENE valores infinitos!!!")
            return False
        if audio.max() > 10:
            print(f"   ❌ Valores DEMASIADO ALTOS (max={audio.max()})!!!")
            return False
        
        # Split at the sample midpoint: first half is clean, second is distorted.
        mid = len(audio) // 2
        # There is no automatic silence removal or time alignment. If a separator is
        # present, its samples remain in the halves and affect these comparisons.
        clean = audio[:mid]
        distorted = audio[mid:]
        
        print(f"\n🔍 ANÁLISIS DE MITADES:")
        print(f"\n   PRIMERA MITAD (limpia):")
        print(f"      Duración: {len(clean)/sr:.2f}s")
        print(f"      Min/Max: {clean.min():.6f} / {clean.max():.6f}")
        print(f"      RMS: {np.sqrt(np.mean(clean**2)):.6f}")
        print(f"      ¿Todo ceros?: {'SÍ ❌' if np.allclose(clean, 0) else 'No ✓'}")
        
        print(f"\n   SEGUNDA MITAD (distorsionada):")
        print(f"      Duración: {len(distorted)/sr:.2f}s")
        print(f"      Min/Max: {distorted.min():.6f} / {distorted.max():.6f}")
        print(f"      RMS: {np.sqrt(np.mean(distorted**2)):.6f}")
        print(f"      ¿Todo ceros?: {'SÍ ❌' if np.allclose(distorted, 0) else 'No ✓'}")
        
        # Compare signal levels using root mean square (RMS) amplitude.
        print(f"\n⚖️  COMPARACIÓN:")
        
        rms_clean = np.sqrt(np.mean(clean**2))
        rms_dist = np.sqrt(np.mean(distorted**2))
        rms_ratio = rms_dist / (rms_clean + 1e-10)
        
        # Similar RMS levels alone do not establish whether a signal is distorted;
        # the following messages report the original fixed-threshold heuristic.
        print(f"   RMS ratio (dist/clean): {rms_ratio:.3f}")
        if rms_ratio > 0.9 and rms_ratio < 1.1:
            print(f"   ❌ Las mitades tienen ENERGÍA MUY SIMILAR")
            print(f"      → NO hay distorsión evidente")
        else:
            print(f"   ✓ Cambio de energía detectado")
        
        # Compare corresponding samples; similar waveform shapes correlate highly.
        min_len = min(len(clean), len(distorted))
        correlation = np.corrcoef(clean[:min_len], distorted[:min_len])[0, 1]
        # High correlation measures linear similarity, not whether an effect is absent.
        print(f"\n   Correlación: {correlation:.3f}")
        if correlation > 0.95:
            print(f"   ❌ Las mitades son CASI IDÉNTICAS")
            print(f"      → La segunda mitad NO tiene distorsión")
        else:
            print(f"   ✓ Las mitades son diferentes")
        
        # Compare summed FFT magnitudes in the upper half of the frequency bins.
        fft_clean = np.abs(np.fft.rfft(clean))
        fft_dist = np.abs(np.fft.rfft(distorted))
        
        # At 44,100 Hz, these upper bins are roughly 11,025-22,050 Hz. The sum is
        # a magnitude measure, not spectral energy or a count of new harmonics.
        hf_clean = np.sum(fft_clean[len(fft_clean)//2:])
        hf_dist = np.sum(fft_dist[len(fft_dist)//2:])
        hf_ratio = hf_dist / (hf_clean + 1e-10)
        
        print(f"\n   Alta frecuencia ratio: {hf_ratio:.3f}")
        if hf_ratio < 1.2:
            print(f"   ❌ Sin aumento de armónicos (típico de distorsión)")
        else:
            print(f"   ✓ Aumento de armónicos detectado")
        
        # Collect the heuristic checks into a diagnostic result.
        print(f"\n{'='*70}")
        print("🎯 VEREDICTO:")
        
        issues = []
        
        if np.allclose(clean, 0):
            issues.append("Primera mitad vacía/silencio")
        if np.allclose(distorted, 0):
            issues.append("Segunda mitad vacía/silencio")
        if rms_ratio > 0.9 and rms_ratio < 1.1:
            issues.append("Sin cambio de energía")
        if correlation > 0.95:
            issues.append("Mitades casi idénticas")
        if hf_ratio < 1.2:
            issues.append("Sin aumento de armónicos")
        
        # This diagnostic uses different, stricter criteria than train.py. A failed
        # check is a reason to inspect the plots, not proof that training will fail.
        if len(issues) == 0:
            print("   ✅ ARCHIVO VÁLIDO - Tiene distorsión real")
            result = True
        else:
            print("   ❌ ARCHIVO PROBLEMÁTICO:")
            for issue in issues:
                print(f"      • {issue}")
            print("\n   ⚠️  Este archivo causará pérdidas altas en entrenamiento")
            result = False
        
        print('='*70)
        
        # Save a diagnostic image in the current working directory.
        create_visualization(audio, clean, distorted, filepath.stem)
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR al procesar archivo: {e}")
        return False


def create_visualization(audio, clean, distorted, filename):
    """Save waveform and spectrogram comparisons as diagnosis_<filename>.png.

    audio is the complete mono waveform; clean and distorted are its two halves.
    filename is the input file stem without its directory or extension.
    Plots use sample indices for waveforms and frame/bin indices for spectrograms;
    the spectrogram axes are not explicitly converted to seconds or hertz.
    """
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'Diagnóstico: {filename}', fontsize=16, fontweight='bold')
    
    # Plot the complete waveform and mark the midpoint used for splitting.
    axes[0, 0].plot(audio, linewidth=0.5)
    axes[0, 0].set_title('Audio Completo')
    axes[0, 0].set_ylabel('Amplitud')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=len(audio)//2, color='r', linestyle='--', label='División')
    axes[0, 0].legend()
    
    # Show up to 4,410 samples from the start (about 0.1 s at 44,100 Hz).
    zoom_samples = min(4410, len(audio)//4)
    axes[0, 1].plot(audio[:zoom_samples], linewidth=0.5)
    axes[0, 1].set_title('Zoom: Inicio (primera mitad)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot the first half, interpreted as the clean reference.
    axes[1, 0].plot(clean, linewidth=0.5, color='blue')
    axes[1, 0].set_title('Primera Mitad (Limpia)')
    axes[1, 0].set_ylabel('Amplitud')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot the second half, interpreted as the distorted reference.
    axes[1, 1].plot(distorted, linewidth=0.5, color='red')
    axes[1, 1].set_title('Segunda Mitad (Distorsionada)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Use equal sample counts for the two short-time Fourier transforms.
    min_len = min(len(clean), len(distorted))
    
    # Display the clean spectrogram in dB relative to its own peak.
    D_clean = librosa.amplitude_to_db(np.abs(librosa.stft(clean[:min_len])), ref=np.max)
    img1 = axes[2, 0].imshow(D_clean, aspect='auto', origin='lower', cmap='viridis')
    axes[2, 0].set_title('Espectrograma: Limpia')
    axes[2, 0].set_ylabel('Frecuencia')
    axes[2, 0].set_xlabel('Tiempo')
    plt.colorbar(img1, ax=axes[2, 0])
    
    # Display the distorted spectrogram in dB relative to its own peak.
    D_dist = librosa.amplitude_to_db(np.abs(librosa.stft(distorted[:min_len])), ref=np.max)
    img2 = axes[2, 1].imshow(D_dist, aspect='auto', origin='lower', cmap='viridis')
    axes[2, 1].set_title('Espectrograma: Distorsionada')
    axes[2, 1].set_xlabel('Tiempo')
    plt.colorbar(img2, ax=axes[2, 1])
    
    plt.tight_layout()
    # Each plot is saved beside the current command location, not in data/samples.
    plt.savefig(f'diagnosis_{filename}.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Gráfica guardada: diagnosis_{filename}.png")
    plt.close()


def diagnose_all_files(folder):
    """Inspect all lowercase .wav files directly inside folder.

    Print the files that pass or fail the diagnostic checks. This function does
    not search subfolders, modify the WAV files, or call train.py.
    """
    
    folder = Path(folder)
    wav_files = list(folder.glob("*.wav"))
    
    # This scan is non-recursive and only matches the .wav pattern.
    if len(wav_files) == 0:
        print(f"❌ No se encontraron archivos .wav en {folder}")
        return
    
    print(f"\n{'#'*70}")
    print(f"DIAGNÓSTICO DE DATOS DE ENTRENAMIENTO")
    print(f"Carpeta: {folder}")
    print(f"Archivos: {len(wav_files)}")
    print(f"{'#'*70}")
    
    valid_files = []
    invalid_files = []
    
    for wav_file in wav_files:
        is_valid = diagnose_file(wav_file)
        
        if is_valid:
            valid_files.append(wav_file.name)
        else:
            invalid_files.append(wav_file.name)
    
    # Summarize files that passed or failed these diagnostic checks.
    print(f"\n\n{'#'*70}")
    print("RESUMEN FINAL")
    print(f"{'#'*70}\n")
    
    print(f"✅ Archivos VÁLIDOS: {len(valid_files)}")
    for f in valid_files:
        print(f"   • {f}")
    
    print(f"\n❌ Archivos INVÁLIDOS/PROBLEMÁTICOS: {len(invalid_files)}")
    for f in invalid_files:
        print(f"   • {f}")
    
    if len(invalid_files) == len(wav_files):
        print(f"\n{'='*70}")
        print("❌❌❌ TODOS LOS ARCHIVOS SON INVÁLIDOS ❌❌❌")
        print('='*70)
        print("\n🔴 PROBLEMA CRÍTICO:")
        print("   Ninguno de tus archivos tiene distorsión detectable.")
        print("\n💡 SOLUCIONES:")
        print("   1. Verifica que la SEGUNDA MITAD tenga distorsión MUY obvia")
        print("   2. El silencio debe estar en el MEDIO (no al inicio/final)")
        print("   3. Formato correcto: [Limpio] [Silencio] [Distorsionado]")
        print("   4. Prueba con distorsión MÁS AGRESIVA (gain alto en pedal)")
        print("\n   Las gráficas guardadas (diagnosis_*.png) te ayudarán a ver el problema")
        print('='*70)
    
    elif len(invalid_files) > 0:
        print(f"\n⚠️  RECOMENDACIÓN:")
        print(f"   Elimina o corrige los archivos inválidos antes de entrenar")
        print(f"   Revisa las gráficas guardadas (diagnosis_*.png)")
    
    else:
        print(f"\n✅ ¡EXCELENTE! Todos los archivos son válidos")
        print(f"   Puedes proceder con el entrenamiento")


# ============================================================================
# SYNTHETIC TEST SAMPLE GENERATOR
# ============================================================================

def create_test_sample():
    """Create a synthetic hard-clipped example for checking the workflow.

    Write data/samples/test_sample_synthetic.wav at 44,100 Hz: five seconds of
    clean harmonic audio, one second of silence, then five seconds of clipped
    audio. This is a generated example, not a recording of a physical pedal.
    The silence remains in the file when the loaders split it at the midpoint.
    """
    
    print("\n" + "="*70)
    print("CREANDO MUESTRA DE PRUEBA SINTÉTICA")
    print("="*70 + "\n")
    
    sr = 44100
    duration = 5  # Seconds for each clean/distorted section, excluding the separator.
    
    # Generate a synthetic clean waveform over the requested duration.
    t = np.linspace(0, duration, sr * duration)
    
    # Mix a 196 Hz fundamental with its second and third harmonics.
    clean = (np.sin(2 * np.pi * 196 * t) +      # 196 Hz: approximately G3.
             0.5 * np.sin(2 * np.pi * 392 * t) +  # 392 Hz: approximately G4.
             0.3 * np.sin(2 * np.pi * 588 * t))   # 588 Hz: third harmonic, approximately D5.
    
    # Scale the clean waveform to a peak amplitude of 0.7.
    clean = clean / np.abs(clean).max() * 0.7
    
    # Apply a mathematical hard-clipping effect to create a test target.
    threshold = 0.3
    distorted = np.copy(clean)
    distorted = np.clip(distorted / threshold, -1, 1) * threshold
    
    # Add gain after clipping, then limit values to the range [-1, 1].
    distorted = distorted * 1.5
    distorted = np.clip(distorted, -1, 1)
    
    # Create a one-second silent separator.
    silence = np.zeros(sr * 1)
    
    # Combine the sections: [clean] [silence] [distorted].
    full_audio = np.concatenate([clean, silence, distorted])
    
    # Write the test WAV into the same sample folder used by train.py.
    output_path = Path("data/samples/test_sample_synthetic.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # This fixed filename is reused on every call and may overwrite an earlier test.
    sf.write(output_path, full_audio, sr)
    
    print(f"✅ Muestra sintética creada: {output_path}")
    print(f"   Duración: {len(full_audio)/sr:.1f}s")
    print(f"   Formato: 5s limpio + 1s silencio + 5s distorsionado")
    print(f"\n💡 USA ESTE ARCHIVO para verificar que el entrenamiento funciona")
    print(f"   Si este archivo entrena bien pero los tuyos no, el problema")
    print(f"   es la calidad de tus grabaciones.\n")


# ============================================================================
# MAIN
# ============================================================================

# Only run the command-line workflow when this file is executed directly.
# The original printed banner below says diagnose_data.py; the supplied filename
# is diagnose_file.py, so use python diagnose_file.py in the command window.
if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           DIAGNÓSTICO DE DATOS PARA DISTORSIÓN IA                ║
╚══════════════════════════════════════════════════════════════════╝

Este script analiza tus archivos .wav para detectar problemas ANTES
de entrenar. Si las pérdidas son >100, el problema está aquí.

Opciones:

1. Diagnosticar archivos existentes:
   python diagnose_data.py

2. Crear muestra sintética de prueba:
   python diagnose_data.py --create-test

""")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-test":
        create_test_sample()
    else:
        # Same relative training-data directory used by train.py. Run from the
        # project folder so both scripts find the intended samples.
        DATA_FOLDER = "data/samples"
        
        if not Path(DATA_FOLDER).exists():
            print(f"❌ La carpeta '{DATA_FOLDER}' no existe")
            Path(DATA_FOLDER).mkdir(parents=True)
            print(f"✅ Carpeta creada. Añade archivos .wav ahí")
        else:
            diagnose_all_files(DATA_FOLDER)
            
            print("\n\n💡 ¿Quieres crear una muestra sintética de prueba? (s/n): ", end='')
            resp = input().strip().lower()
            # The original prompt expects s (Spanish for yes) to create the test sample.
            if resp == 's':
                create_test_sample()