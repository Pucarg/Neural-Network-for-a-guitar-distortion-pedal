# Neural Guitar Distortion

A Python neural network learns guitar distortion from paired clean and distorted recordings, then applies the learned effect to new audio. Includes data diagnostics, model training, and WAV export. Run audio processing from a command window using process_audio.py.

The description above contains 264 characters, including spaces, and is also provided in `DESCRIPTION.txt`.

## Files and workflow

| File | Purpose | Main inputs and outputs |
| --- | --- | --- |
| `diagnose_file.py` | Inspect training recordings and optionally create a synthetic test sample. | Reads `data/samples/*.wav`; saves `diagnosis_<input_stem>.png`. The test option writes `data/samples/test_sample_synthetic.wav`. |
| `train.py` | Define the neural network, prepare training pairs, and train the model. | Reads `data/samples/*.wav`; saves two model checkpoints and `training_loss.png`. |
| `process_audio.py` | Apply the trained effect to one recording or a folder of recordings. | Imports `DistortionNet` from `train.py`, loads `models/distortion_model.pth`, and writes processed audio. |

Run diagnostics first, train the model, and then process new recordings. Diagnostics are a separate inspection step: training does not automatically run `diagnose_file.py` or use its pass/fail results. Importing `train.py` from the processor does not start training.

Keep all three scripts in the same project folder. Training audio and trained checkpoints are separate files; they are not included with these documented scripts.

## Python environment

Use the Python environment in which the original scripts already run. The scripts import these third-party packages:

| Package | Role |
| --- | --- |
| `torch` (PyTorch) | Neural-network layers, training, checkpoint loading, and CPU/CUDA execution. |
| `numpy` | Waveform arrays, numerical calculations, and diagnostic FFTs. |
| `librosa` | Audio loading, mono conversion, resampling, and spectrogram calculations. |
| `soundfile` | Writing audio files. |
| `matplotlib` | Saving diagnostic and training plots. |

`pathlib`, `os`, `sys`, and `time` are Python standard-library modules. The supplied scripts do not specify package versions. No WaveNet, U-Net, or Demucs package or pretrained model is imported; those names in the comments explain architectural inspiration only.

Training checks for CUDA. If it is unavailable, the existing interactive prompt accepts `SI` to continue on the CPU. Audio processing automatically chooses CUDA when available and otherwise uses the CPU without that prompt.

## Prepare the training recordings

Create a folder named `data/samples` inside the project folder and place the training WAV files directly inside it. The scripts scan the `*.wav` pattern without searching subfolders.

Each recording is loaded as mono at 44,100 Hz and split exactly at its sample midpoint:

- The first half is the clean guitar input.
- The second half is the corresponding distorted target.
- Corresponding positions in the two halves should contain the same performance, aligned in time.

The code does not detect or remove a silent separator, compensate for device latency, or align the two performances. Although the original console text describes clean audio followed by silence and distorted audio, any separator remains in the halves after the midpoint split. Prepare the recordings with this actual behavior in mind.

The diagnostic script checks levels, correlation, and high-frequency magnitudes. Its fixed thresholds are screening aids, and they differ from the checks in the training dataset. A diagnostic verdict does not establish distortion quality or guarantee training performance.

## Run the scripts

**Run `process_audio.py` through a command window**, such as Windows Command Prompt, PowerShell, or a terminal. Open that window in the project folder, or change to it first. For example, replace this Windows path with your own:

```text
cd "C:\path\to\guitar-distortion"
```

All relative paths, including `data/samples` and `models/distortion_model.pth`, are resolved from the command window's current working directory. Quote paths that contain spaces.

### 1. Inspect the data

```text
python diagnose_file.py
```

This prints diagnostic results and saves comparison images in the current working directory. At the optional prompt, enter `s` to create a synthetic sample; another response skips that action.

To create the synthetic sample directly:

```text
python diagnose_file.py --create-test
```

The generated WAV contains five seconds of clean harmonic audio, one second of silence, and five seconds of mathematically clipped audio. It is a workflow example, not a recording of a physical distortion pedal. The generated filename is reused on each run, and the sample is included in subsequent training scans if left in `data/samples`.

The original diagnostic banner refers to `diagnose_data.py`. The supplied filename is **`diagnose_file.py`**, so use the commands shown here.

### 2. Train the model

```text
python train.py
```

The main entry point uses 100 maximum epochs, a batch size of 8, an initial learning rate of 0.0005, and a waveform-plus-spectrum loss. An epoch is one pass through the training pairs. These settings are defined in the script; it does not parse training command-line options.

If `data/samples` does not exist, the script creates it and exits. Add recordings and run the command again.

| Generated file | Meaning |
| --- | --- |
| `models/distortion_model.pth` | Final network state; the processor loads this file by default. |
| `models/distortion_model_best.pth` | Network state from the epoch with the lowest average training loss. |
| `training_loss.png` | Plot of the recorded training losses. |

Both checkpoints contain `model_state_dict`, `epoch`, and `loss`. The best checkpoint stores a zero-based epoch index; the final checkpoint stores the requested epoch count, including when the training loop stops early. These fields describe the original save behavior.

### 3. Process one recording

```text
python process_audio.py single "input.wav" "output.wav"
```

To use the automatically generated output name:

```text
python process_audio.py single "input.wav"
```

The second command creates `input_distorted.wav` beside the input. For an explicit output path, its parent directory must already exist. Running the processor without `single` or `batch` prints a message and exits.

### 4. Process a folder

```text
python process_audio.py batch "input_audio" "processed_audio"
```

Batch mode finds `*.wav` and `*.mp3` files directly inside the input folder, creates the output folder when needed, and writes `<input_stem>_distorted.wav` for each recording. Inputs sharing a stem, such as `take.wav` and `take.mp3`, target the same output filename. Existing output files may be overwritten.

Both command-line modes use `models/distortion_model.pth`. There is no command-line option for choosing another checkpoint. The functions `process_single_file()` and `batch_process()` accept `model_path` when imported into another Python program, including `models/distortion_model_best.pth` when desired.

## How the model and audio processing work

**Training data.** The loader screens the two halves, normalizes each independently using its 99.5th-percentile amplitude, and limits outlying samples. This independent normalization does not retain the original clean-to-distorted gain ratio. It extracts complete 16,384-sample pairs with 50% overlap and adds three variants per accepted pair: quieter audio, louder audio, and low-level noise on the clean input. All pairs are held in memory.

**Network.** `DistortionNet` maps a clean waveform to a predicted distorted waveform. It uses 32 internal feature channels, ten dilated one-dimensional convolution layers, batch normalization, nonlinear activations, residual additions, and skip connections. Dilation lets layers use spaced samples as context. Residual and skip additions pass intermediate features to later computations. The network preserves the sample count and operates on tensors shaped `[batch, 1, samples]`.

**Training objective.** The default loss is waveform mean squared error (MSE) plus 0.1 times the MSE between length-normalized FFT magnitudes. Adam updates the parameters, gradient clipping limits update gradients, and a scheduler reduces the learning rate when training loss plateaus. The best model is selected by training loss; this code does not create a separate validation or test set.

**Processing.** New audio is converted to mono at 44,100 Hz, peak-normalized as one recording, and processed in 44,100-sample chunks with 50% overlap. A Hann window tapers each prediction before overlapping sections are added. The result is peak-normalized to 0.9, given 10 ms fades at both ends, and written to disk. The original input peak is not restored.

**Existing boundary behavior.** The overlap-add implementation does not divide by accumulated window weights, so its smoothing is not a guarantee of perfect reconstruction. For recordings longer than one chunk, the current chunk count includes complete windows only; a final remainder outside those windows can be omitted. Inputs shorter than one chunk are padded, processed, and trimmed. These behaviors are documented in the comments without changing their implementation.

## Scope of this documentation update

Comments and docstrings are written in English and explain the existing implementation. Executable statements, parameters, imports, filenames, console messages, and plot-label strings are preserved. Consequently, the program's original console messages and plot labels remain in Spanish.

The edited scripts were checked for valid Python syntax, identical executable tokens outside comments/docstrings, and identical syntax trees after excluding docstrings. No training or audio-processing run was performed as part of this documentation-only update.
