"""Train a neural network to learn a guitar-distortion effect from audio pairs.

Run python train.py from the project folder. Training WAV files are read
from data/samples; diagnose_file.py can inspect them beforehand. Each WAV
is split at its midpoint into clean input and distorted target audio.

The script checks CUDA availability, validates samples, creates overlapping
training pairs and augmented examples, and fits DistortionNet. It writes
models/distortion_model.pth, models/distortion_model_best.pth, and
training_loss.png. process_audio.py imports DistortionNet and loads the
final checkpoint to apply the learned effect to new recordings.
"""

# PyTorch supplies tensors, layers, optimizers, and dataset batching.
# NumPy/librosa handle audio, Matplotlib saves the loss plot, and the remaining
# imports provide paths, directory creation, and timing.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os
from pathlib import Path
import matplotlib.pyplot as plt
import time

# ============================================================================
# CUDA AVAILABILITY CHECK AND OPTIONAL CPU FALLBACK
# ============================================================================

def verify_gpu():
    """Return a CUDA device if available, or offer an interactive CPU fallback.

    The existing console prompt accepts SI to continue on the CPU and exits
    otherwise. The printed installation hint is part of the original program;
    this function does not install or modify any Python packages.
    """
    if not torch.cuda.is_available():
        print("\n" + "="*60)
        print("❌ GPU (CUDA) NO DISPONIBLE")
        print("="*60)
        print("\nEl entrenamiento en CPU será EXTREMADAMENTE lento.")
        print("\nPara habilitar GPU:")
        print("1. Verifica que tienes una GPU NVIDIA")
        print("2. Instala PyTorch con CUDA:")
        print("   pip uninstall torch")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n¿Continuar de todos modos con CPU? (NO recomendado)")
        print("Escribe 'SI' para continuar: ", end='')
        
        # Keep the original response keyword: enter SI to accept CPU training.
        resp = input().strip().upper()
        if resp != 'SI':
            print("\nEntrenamiento cancelado.")
            print("Por favor, instala PyTorch con soporte CUDA.")
            exit(1)
        print("\n⚠️  Continuando con CPU... esto será LENTO.\n")
        return torch.device('cpu')
    
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n✅ GPU DETECTADA: {gpu_name}")
    print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    return device


# ============================================================================
# TEMPORAL CONVOLUTIONAL MODEL
# ============================================================================

class DistortionNet(nn.Module):
    """Temporal convolutional network for mono guitar-distortion modeling.

    Input and output tensors have shape [batch, 1, samples]. The model uses
    32 internal channels, ten dilated Conv1d layers, batch normalization,
    residual additions, and accumulated skip connections. Padding preserves
    the sample count, and the final scaled tanh bounds the output at +/-1.2.

    WaveNet, U-Net, and Demucs in the original documentation are architectural
    references only: no code, package, or pretrained model from those projects
    is loaded here. This implementation has no downsampling/upsampling
    encoder-decoder and is not a causal streaming model.

    Supervised training aims to approximate the clean-to-distorted mapping
    present in the recordings. The learned behavior depends on those examples
    and does not guarantee a complete model of every physical-device effect.
    """
    
    def __init__(self):
        super(DistortionNet, self).__init__()
        
        channels = 32  # Number of internal feature channels.
        
        # Map the mono waveform to 32 features without changing its length.
        self.input_conv = nn.Conv1d(1, channels, kernel_size=5, padding=2)
        self.input_bn = nn.BatchNorm1d(channels)
        
        # Build ten dilated convolution layers and their batch-normalization layers.
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        dilations = [1, 2, 4, 8, 16, 16, 8, 4, 2, 1]  # Dilation grows, then shrinks symmetrically.
        for dilation in dilations:
            self.conv_layers.append(
                nn.Conv1d(channels, channels, kernel_size=3, 
                         dilation=dilation, padding=dilation)
            )
            self.bn_layers.append(nn.BatchNorm1d(channels))
        
        # Mix feature channels at each sample with a pointwise convolution.
        self.mix_conv = nn.Conv1d(channels, channels, kernel_size=1)
        self.mix_bn = nn.BatchNorm1d(channels)
        
        # Map the features back to one audio channel.
        self.output_conv = nn.Conv1d(channels, 1, kernel_size=1)
        
    # x has shape [batch, channels=1, samples]; padding keeps the time axis length.
    def forward(self, x):
        # Extract input features, normalize them, and apply a tanh activation.
        x = self.input_bn(self.input_conv(x))
        x = torch.tanh(x)
        
        # Accumulate intermediate features for the later skip connection.
        skip_sum = 0
        
        # Apply the dilated layers with residual additions and skip accumulation.
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
            identity = x
            
            # Convolve along time, then normalize the feature channels.
            x = bn(conv(x))
            
            # Blend smooth tanh and rectified linear (ReLU) activations.
            x = torch.tanh(x) * 0.7 + torch.relu(x) * 0.3
            
            # Add 30% of the previous features through a residual connection.
            x = x + identity * 0.3
            
            # Accumulate features at layer indices 0, 2, 4, 6, and 8.
            if i % 2 == 0:
                skip_sum = skip_sum + x
        
        # Add 10% of the accumulated intermediate features.
        x = x + skip_sum * 0.1
        
        # Mix channels and apply another nonlinear activation.
        x = self.mix_bn(self.mix_conv(x))
        x = torch.tanh(x)
        
        # Produce a single waveform channel with a bounded output amplitude.
        x = self.output_conv(x)
        x = torch.tanh(x) * 1.2  # Scale the tanh output to a limiting amplitude of 1.2.
        
        return x


# ============================================================================
# PAIRED AUDIO DATASET WITH BASIC VALIDATION
# ============================================================================

class GuitarDistortionDataset(Dataset):
    """Build paired clean/processed waveform chunks for supervised training.

    Read lowercase .wav files directly inside data_folder and convert them to
    mono at 44,100 Hz. The first half of each recording is treated as the clean
    input; the second half is treated as the distorted target. Corresponding
    sample positions should represent the same performance at the same time.
    The loader does not find a separator, remove silence, or align recordings.

    Each half is normalized independently, so the original input/output gain
    ratio is not retained. Accepted chunks receive three augmented variants:
    lower level, higher level, and a slightly noisy clean input. All pairs are
    held in memory. A sample returns two tensors shaped [1, chunk_size].
    """
    
    def __init__(self, data_folder, chunk_size=16384, overlap=0.5):
        self.data_folder = Path(data_folder)
        self.chunk_size = chunk_size
        # With the defaults, successive 16,384-sample chunks start 8,192 samples apart.
        self.hop_size = int(chunk_size * (1 - overlap))
        self.pairs = []
        self.rejected_files = []
        self.load_data()
        
    # Read and build the full in-memory dataset during dataset initialization.
    def load_data(self):
        print("\n" + "="*60)
        print("CARGANDO Y VALIDANDO DATOS")
        print("="*60 + "\n")
        
        # Read only files directly inside the supplied folder; subfolders are excluded.
        audio_files = list(self.data_folder.glob("*.wav"))
        
        if len(audio_files) == 0:
            print("❌ No se encontraron archivos .wav")
            return
        
        print(f"Archivos encontrados: {len(audio_files)}\n")
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] {audio_file.name}")
            
            try:
                # Use the same mono, 44,100 Hz format as process_audio.py.
                audio, sr = librosa.load(audio_file, sr=44100, mono=True)
                
                # Report the duration after mono conversion and resampling.
                duration = len(audio) / sr
                print(f"  Duración: {duration:.2f}s")
                
                # Split exactly at the midpoint; no silence detection or alignment is applied.
                mid = len(audio) // 2
                clean = audio[:mid]
                distorted = audio[mid:]
                
                # Screen the two halves with the basic checks in validate_distortion().
                is_valid, reason, metrics = self.validate_distortion(clean, distorted)
                
                if not is_valid:
                    print(f"  ❌ RECHAZADO: {reason}")
                    print(f"     RMS_ratio={metrics['rms_ratio']:.2f}, "
                          f"Diff={metrics['mean_diff']:.4f}, "
                          f"Corr={metrics['correlation']:.3f}")
                    self.rejected_files.append((audio_file.name, reason))
                    continue
                
                print(f"  ✅ VALIDADO")
                print(f"     RMS_ratio={metrics['rms_ratio']:.2f}, "
                      f"Diff={metrics['mean_diff']:.4f}, "
                      f"Corr={metrics['correlation']:.3f}")
                
                # Estimate a separate reference amplitude for each half.
                # Use the 99.5th percentile to reduce the influence of isolated peaks.
                max_clean = np.percentile(np.abs(clean), 99.5)
                max_dist = np.percentile(np.abs(distorted), 99.5)
                
                # Scale each half independently so its reference amplitude becomes 0.8.
                # The small denominator offset avoids division by zero.
                if max_clean > 0:
                    clean = clean / (max_clean + 1e-8) * 0.8
                if max_dist > 0:
                    distorted = distorted / (max_dist + 1e-8) * 0.8
                
                # Hard-limit outlying samples to [-0.95, 0.95].
                clean = np.clip(clean, -0.95, 0.95)
                distorted = np.clip(distorted, -0.95, 0.95)
                
                # Extract paired, overlapping chunks from corresponding sample positions.
                min_len = min(len(clean), len(distorted))
                num_chunks = max(1, (min_len - self.chunk_size) // self.hop_size + 1)
                
                chunks_added = 0
                for j in range(num_chunks):
                    start = j * self.hop_size
                    end = start + self.chunk_size
                    
                    # Keep only complete paired windows. Very short halves cannot supply a
                    # training chunk, and an incomplete trailing window is discarded.
                    if end > min_len:
                        break
                    
                    clean_chunk = clean[start:end].copy()
                    dist_chunk = distorted[start:end].copy()
                    
                    # Require at least one clean-input sample above this absolute amplitude.
                    if np.abs(clean_chunk).max() > 0.01:
                        # Skip chunks containing non-finite values in either waveform.
                        if np.isnan(clean_chunk).any() or np.isnan(dist_chunk).any():
                            print(f"      ⚠️ Chunk {j} tiene NaN, saltando...")
                            continue
                        if np.isinf(clean_chunk).any() or np.isinf(dist_chunk).any():
                            print(f"      ⚠️ Chunk {j} tiene infinitos, saltando...")
                            continue
                        
                        # Store the original clean/distorted chunk pair.
                        self.pairs.append((clean_chunk, dist_chunk))
                        chunks_added += 1
                        
                        # Data augmentation: add three variations for each accepted original pair.
                        # 1. Reduce both waveforms to 90% of their amplitude.
                        self.pairs.append((clean_chunk * 0.9, dist_chunk * 0.9))
                        
                        # 2. Increase both waveforms to 110%, clipping to [-1, 1].
                        self.pairs.append((np.clip(clean_chunk * 1.1, -1, 1), 
                                          np.clip(dist_chunk * 1.1, -1, 1)))
                        
                        # 3. Add low-level Gaussian noise to the clean input only.
                        noise = np.random.randn(self.chunk_size) * 0.002
                        self.pairs.append((clean_chunk + noise, dist_chunk))
                        
                        chunks_added += 3  # Include the three augmented pairs in the count.
                
                print(f"  → {chunks_added} chunks extraídos (con augmentation)\n")
                
            except Exception as e:
                print(f"  ❌ Error: {e}\n")
                self.rejected_files.append((audio_file.name, str(e)))
        
        # Summarize rejected files and the total number of training pairs.
        print("="*60)
        if len(self.rejected_files) > 0:
            print(f"⚠️  {len(self.rejected_files)} archivos rechazados:")
            for filename, reason in self.rejected_files:
                print(f"   - {filename}: {reason}")
            print("\n⚠️  Solo casos extremos son rechazados ahora")
            print("   (mitades idénticas, vacías, o sin diferencia)")
            print("="*60 + "\n")
        
        print(f"✅ Total: {len(self.pairs)} pares para entrenamiento")
        print(f"   (incluye data augmentation: variaciones de volumen y ruido)")
        print("="*60 + "\n")
        
        if len(self.pairs) < 100:
            print(f"⚠️  ADVERTENCIA: Datos limitados ({len(self.pairs)} pares)")
            print(f"   El modelo aprenderá, pero con más datos sería mejor")
            print(f"   Considera añadir más muestras si los resultados no son óptimos\n")
    
    def validate_distortion(self, clean, distorted):
        """Return (is_valid, reason, metrics) for basic waveform-pair checks.

        metrics contains RMS ratio, mean absolute difference, and correlation.
        Rejection uses correlation above 0.98, either RMS below 0.001, or mean
        difference below 0.001; the RMS ratio is reported but is not a rejection
        criterion here. These checks differ from diagnose_file.py and do not
        verify time alignment or measure perceptual similarity.
        """
        # 1. Measure RMS amplitude and the distorted-to-clean RMS ratio.
        rms_clean = np.sqrt(np.mean(clean**2))
        rms_dist = np.sqrt(np.mean(distorted**2))
        rms_ratio = rms_dist / (rms_clean + 1e-8)
        
        # 2. Measure the mean absolute difference over matching sample counts.
        diff = np.abs(clean[:min(len(clean), len(distorted))] - 
                     distorted[:min(len(clean), len(distorted))])
        mean_diff = np.mean(diff)
        
        # 3. Measure linear correlation between the two waveform halves.
        min_len = min(len(clean), len(distorted))
        if min_len > 100:
            correlation = np.corrcoef(clean[:min_len], distorted[:min_len])[0, 1]
        else:
            # Very short pairs receive a high-correlation fallback and are rejected.
            correlation = 1.0
        
        metrics = {
            'rms_ratio': rms_ratio,
            'mean_diff': mean_diff,
            'correlation': correlation
        }
        
        # Apply fixed heuristic rejection thresholds; these do not prove distortion.
        # Reject a pair when any of the following conditions holds:
        # 1. The waveform correlation exceeds 0.98.
        # 2. Either waveform has an RMS amplitude below 0.001.
        # 3. The mean absolute sample difference is below 0.001.
        
        if correlation > 0.98:
            return False, "Mitades idénticas (correlación muy alta)", metrics
        
        if rms_clean < 0.001 or rms_dist < 0.001:
            return False, "Una mitad está vacía o muy silenciosa", metrics
        
        if mean_diff < 0.001:
            return False, "Diferencia entre mitades insignificante", metrics
        
        # Accept the pair if it passes all three basic filters.
        return True, "OK", metrics
    
    # Expose the number of stored pairs to PyTorch Dataset/DataLoader.
    def __len__(self):
        return len(self.pairs)
    
    # Convert a stored pair to float32 tensors and add the mono channel axis.
    # DataLoader will prepend the batch axis used by DistortionNet.
    def __getitem__(self, idx):
        clean, distorted = self.pairs[idx]
        return (torch.FloatTensor(clean).unsqueeze(0),
                torch.FloatTensor(distorted).unsqueeze(0))


# ============================================================================
# WAVEFORM AND NORMALIZED SPECTRAL LOSS
# ============================================================================

class DistortionLoss(nn.Module):
    """Combine waveform MSE with a normalized FFT-magnitude MSE.

    Divide each real-input FFT magnitude by its sample count to keep that term
    from scaling directly with the chunk length. The objective is waveform MSE
    plus 0.1 times spectral MSE. It compares magnitude spectra, not FFT phase;
    the waveform term still compares corresponding samples in time.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # 1. Compare predicted and target samples using mean squared error (MSE).
        time_loss = self.mse(pred, target)
        
        # 2. Compute a real-valued-input FFT along the sample dimension.
        pred_fft = torch.fft.rfft(pred, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        
        # Divide each FFT magnitude by its waveform length to control its scale.
        pred_mag = torch.abs(pred_fft) / pred.shape[-1]
        target_mag = torch.abs(target_fft) / target.shape[-1]
        
        spectral_loss = self.mse(pred_mag, target_mag)
        
        # Add the spectral error with a relative weight of 0.1.
        # The waveform and spectral terms need not have the same magnitude.
        total_loss = time_loss + 0.1 * spectral_loss
        
        return total_loss


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(data_folder, epochs=100, batch_size=8, lr=0.0005, simple_loss=False):
    """Fit DistortionNet to paired clean and distorted recordings.

    data_folder holds the training WAV files. epochs is the maximum number
    of passes over the augmented dataset; batch_size sets the pairs per update;
    lr is the initial Adam learning rate. simple_loss selects waveform-only MSE
    when True and DistortionLoss when False.

    Training aims to learn the recorded device response, including nonlinear
    and local temporal behavior, from examples rather than a fixed clipping
    formula. Dilated convolutions and skip connections supply temporal context;
    no external WaveNet, U-Net, or Demucs implementation is required.

    Save the best checkpoint by average training loss, then save the final
    network and a loss plot. There is no held-out validation/test split here.
    process_audio.py uses the final checkpoint by default. This function does
    not return a trained model object; its persistent outputs are the files.
    """
    
    # Step 1: Select CUDA, or request confirmation before using the CPU.
    device = verify_gpu()
    
    # Step 2: Load and validate paired audio; use 16,384-sample training chunks.
    dataset = GuitarDistortionDataset(data_folder, chunk_size=16384, overlap=0.5)
    
    if len(dataset) == 0:
        print("❌ No hay datos válidos para entrenar")
        return
    
    if len(dataset.rejected_files) == len(list(Path(data_folder).glob("*.wav"))):
        print("\n❌ TODOS los archivos fueron rechazados")
        print("   Tus muestras NO tienen distorsión suficiente.")
        print("   La segunda mitad debe tener distorsión MUY evidente.\n")
        return
    
    # Shuffle pairs each epoch. Load them in the main process; pin host memory
    # when CUDA is used to support transfer to the GPU.
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=(device.type == 'cuda'))
    
    # Step 3: Create the network and move it to the selected device.
    model = DistortionNet().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Modelo: {num_params:,} parámetros\n")
    
    # Step 4: Configure Adam and reduce the learning rate when loss plateaus.
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Track average training loss and halve the learning rate after a plateau.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=10)
    
    # Choose waveform-only MSE or the waveform-plus-spectrum objective.
    if simple_loss:
        print("📊 Usando pérdida SIMPLE (solo MSE)\n")
        criterion = nn.MSELoss()
    else:
        print("📊 Usando pérdida MEJORADA (MSE + espectral normalizado)\n")
        criterion = DistortionLoss()
    
    # Step 5: Track training losses, the best epoch, and elapsed time.
    losses = []
    best_loss = float('inf')
    start_time = time.time()
    
    print("="*60)
    print("ENTRENAMIENTO")
    print("="*60 + "\n")
    
    # Keep the first batch loss for the initial scale diagnostic.
    first_batch_loss = None
    
    for epoch in range(epochs):
        # Enable training behavior, including updates to batch-normalization statistics.
        model.train()
        epoch_loss = 0.0
        batch_times = []
        batch_count = 0
        
        for clean, distorted in dataloader:
            batch_start = time.time()
            
            clean = clean.to(device)
            distorted = distorted.to(device)
            
            predicted = model(clean)
            loss = criterion(predicted, distorted)
            
            # Print the first prediction and target ranges to help inspect data scaling.
            if epoch == 0 and batch_count == 0:
                first_batch_loss = loss.item()
                print(f"🔍 DEBUG Primera pérdida: {first_batch_loss:.6f}")
                print(f"   Rango predicción: [{predicted.min().item():.3f}, {predicted.max().item():.3f}]")
                print(f"   Rango objetivo: [{distorted.min().item():.3f}, {distorted.max().item():.3f}]")
                
                if first_batch_loss > 10:
                    print(f"   ⚠️ Pérdida inicial ALTA - puede indicar problema de escala\n")
                elif first_batch_loss > 1:
                    print(f"   ⚠️ Pérdida inicial moderadamente alta\n")
                else:
                    print(f"   ✅ Pérdida inicial en rango normal\n")
            
            # Clear old gradients, backpropagate this batch loss, limit the total
            # gradient norm to 1.0, and update the network parameters.
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_times.append(time.time() - batch_start)
            batch_count += 1
        
        # Average batch losses; this is training loss, not held-out validation loss.
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        # Report a large initial loss and stop if a later loss remains above 5.
        if epoch == 0 and avg_loss > 10:
            print(f"\n⚠️  ALERTA: Primera época con pérdida MUY ALTA ({avg_loss:.2f})")
            print(f"   Esto indica un problema con los datos.")
            print(f"   Continuando, pero revisa que las muestras sean correctas...\n")
        
        if epoch > 10 and avg_loss > 5:
            print(f"\n❌ ENTRENAMIENTO FALLIDO")
            print(f"   Pérdida no converge después de {epoch} épocas")
            print(f"   Posibles causas:")
            print(f"   1. Datos corruptos o mal formateados")
            print(f"   2. Muestras sin distorsión real")
            print(f"   3. Learning rate demasiado alto")
            print(f"\n   Deteniendo entrenamiento...")
            break
        
        # Save models/distortion_model_best.pth when average training loss improves.
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("models", exist_ok=True)
            # The best checkpoint stores a zero-based epoch index. The processor
            # requires model_state_dict, epoch, and loss when loading either checkpoint.
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
            }, "models/distortion_model_best.pth")
        
        # Report the current loss, best loss, approximate epoch time, and ETA.
        elapsed = time.time() - start_time
        epoch_time = np.mean(batch_times) * len(dataloader)
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
        
        print(f"Época [{epoch+1:3d}/{epochs}] "
              f"Pérdida: {avg_loss:.5f} | "
              f"Mejor: {best_loss:.5f} | "
              f"Tiempo: {epoch_time:.1f}s | "
              f"ETA: {eta/60:.1f}min")
        
        # Release unused cached CUDA memory after each epoch.
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Step 6: Save the final model and report training duration.
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print(f"   Tiempo total: {total_time/60:.1f} minutos")
    print(f"   Mejor pérdida: {best_loss:.5f}")
    print("="*60 + "\n")
    
    # Save the last network state to the default file used by process_audio.py.
    # The epoch field below stores the requested count, even if the loop ended early.
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epochs,
        'loss': losses[-1],
    }, "models/distortion_model.pth")
    
    print("✅ Modelo final: models/distortion_model.pth")
    print("✅ Mejor modelo: models/distortion_model_best.pth\n")
    
    # Save training_loss.png in the current working directory.
    plt.figure(figsize=(12, 5))
    plt.plot(losses, linewidth=2, label='Pérdida')
    plt.axhline(y=best_loss, color='r', linestyle='--', label=f'Mejor: {best_loss:.5f}')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Curva de Aprendizaje')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150)
    print("✅ Gráfica: training_loss.png\n")
    plt.close()
    
    # Describe convergence using training loss; no separate test set is evaluated.
    if best_loss < 0.02:
        print("🎉 EXCELENTE convergencia - Modelo listo para usar")
    elif best_loss < 0.1:
        print("✅ BUENA convergencia - Debería funcionar bien")
    elif best_loss < 0.3:
        print("⚠️  Convergencia MODERADA - Puede necesitar más entrenamiento")
    else:
        print("❌ MALA convergencia - Revisa tus datos de entrada")


# Executing python train.py starts training. Importing DistortionNet from
# process_audio.py defines the model without running this entry point.
if __name__ == "__main__":
    # Relative to the command window working directory, as in diagnose_file.py.
    DATA_FOLDER = "data/samples"
    
    if not os.path.exists(DATA_FOLDER):
        print(f"❌ La carpeta '{DATA_FOLDER}' no existe")
        os.makedirs(DATA_FOLDER, exist_ok=True)
        print(f"✅ Carpeta creada. Añade archivos .wav ahí.")
        exit(0)
    
    # Run training with the normalized spectral-loss option and fixed defaults.
    train_model(
        data_folder=DATA_FOLDER,
        epochs=100,
        batch_size=8,
        lr=0.0005,
        simple_loss=False  # False: waveform MSE plus 0.1 times normalized spectral MSE.
                          # True: waveform MSE only.
    )