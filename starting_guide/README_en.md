# Machine learning applied to the detection of epileptic seizures

The Aura association is leading an open and collaborative initiative to develop a connected patch to detect epileptic seizures.

To design this detection system, it is studying a relevant biomarker: The [ECG - Electrocardiogram](https://en.wikipedia.org/wiki/Electrocardiography) signal

## An open database

To build this seizure detection model, we rely on an open database: : **TUH Seizure Corpus (TUSZ) v1.5.1**.

This **67GB** database records the electrophysiological activities ([electroencephalogram](https://en.wikipedia.org/wiki/Electroencephalography), electrocardiogram) of **692 patients** during **5610 examinations** for a total duration of **1074 hours**.

A team of neurologists reread these recordings and annotated nearly **3500 epileptic seizures**. It is on the basis of this binary annotation (background or seizure) that we will be able to train our model.

It is important to note that the seizure time represents **7% of the total recording time**. We will have to take into account this asymmetry of classes (background/crisis) to train the crisis detection algorithm.

You can find more information about this database and the research team that built it [HERE](https://www.isip.piconepress.com/projects/tuh_eeg/))

### Database tree structure

La base de données d'origine est structurée de la façon suivante:
```
─ dev                                               // Dataset splitted into dev/testing/validation
│   ├── 01_tcp_ar                                   // Montage type used for EEG recording (not useful for ECG processing)
│   │   ├── 002
│   │   │   └── 00000258                            // Patient folder
│   │   │       ├── s002_2003_07_21                 // Recording Session folder
│   │   │       │   ├── 00000258_s002_t000.json     // A single EEG/ECG recording
│   │   │       │   └── 00000258_s002_t002.json
│   │   │       └── s003_2003_07_22
│   │   │           ├── 00000258_s003_t000.json
│   │   │           ├── 00000258_s003_t001.json
│   │   │           ├── 00000258_s003_t002.json
│   │   │           ├── 00000258_s003_t003.json
 ...                ...
```

##  A preprocessing on the heart signal
As part of this project, we carried out an initial processing on this large volume of data, which is complex to analyze.

Initially, we chose to exclude the EEG signals in order to keep only the ECG signal.

### ECG signal
Electrocardiography (ECG) is a graphical representation of the electrical activity of the heart. It is measured in our case (simplest assembly) thanks to 2 electrodes placed on the left and right of the torso. This corresponds to electrodes V1 and V2 on the diagram below:

<img src="./images/ElectrodesPositions.jpg" alt="Electrodes positions" width="500"/></br>
 *ECG electrodes. Source: American Heart Association* </br>


The following is an example of a standard ECG record.

<img src="./images/ECG_standard.png" alt="ECG standard" width="500"/></br>
*ECG standard. Source: https://ya-webdesign.com* </br>

Under everyday living conditions, this signal is usually noisy due to muscle movement artifacts or problems with the contact of the electrodes with the skin, which makes the analysis more complex. I share some examples below:

<img src="./images/ECG_low_noise.png" alt="ECG low noise" width="800"/></br>
*ECG low noise*</br>


<img src="./images/ECG_mid_noise.png" alt="ECG mid noise" width="800"/></br>
*ECG mid noise*</br>


<img src="./images/ECG_high_noise.png" alt="ECG high noise" width="800"/></br>
*ECG high noise*</br>

### The R-R intervals
We will now go into a little more detail on the analysis of the heart signal. Research teams and neurologists working on epilepsy tell us that epileptic seizures lead to disturbances in the [autonomous nervous system](https://en.wikipedia.org/wiki/Autonomic_nervous_system) which result in heart rhythm disorders (tachycardia, bradycardia, ...).

These cardiac rhythm disorders are studied through R-R interval analysis.
**An R-R interval represents the duration of one heartbeat** and corresponds to the time between 2 R peaks of the ECG signal, see diagram below:

<img src="./images/RR_intervalles_2.png" alt="R-R intervalles" width="500"/></br>
*R-R intervalle. Source: https://ya-webdesign.com* </br>

There's a direct link in heart rate and R-R interval:

$${bpm} = {\frac{60}{rrinterval}}$$


### Extract R-R intervals from ECG signal
To extract the R-R intervals from the ECG signal we used standard algorithms called "QRS complex detection". These algorithms are robust but far from infallible and there are many implementations.

We have chosen 3 of them among the most powerful and commonly used:
 * Pan Tompkins
 * Stationnary Wavelet (swt)
 * XQRS

We then compared them to assess the robustness of our intervals and the signal quality for each examination.
We calculated the following 2 metrics:

 * $${CoefCorrelation_{algo1, algo2}} = { \frac{2 \times NombreDeDetectionCommune_{algo1, algo2}}{NombreDeDetectionTotal_{algo1} + NombreDeDetectionTotal_{algo2}}}$$
 coefficient between 0 (totally different results) and 1 (perfect correlation between R-R intervals)

 * $MissingBeat_{algo1, algo2}$  the sum of intervals longer than 2 seconds where we didn't detect a heartbeat (physiologically impossible)
 The lower the value, the better !

 ### The available data

 All R-R interval data for each calculation method and their robustness metrics are stored in JSON files, one per scan in the **res-v0_4** folder on the Drive.

 The JSON files follow the following format:
```json
{"infos":                                    // General recording information
  {"sampling_freq": 400,                     // Sampling frequency in Hz
    "start_datetime": "2003-07-21T17:12:54", // Recording starting date
    "exam_duration": 20,                     // Exam duration in seconds
    "ref_file": "00000258_s002_t000.edf"},   // Reference raw EDF file

"pan-tompkins":                              // Computation method - Pan Tompkins
      {"qrs": [...],                         // QRS frame - not useful for this project
        "rr_intervals": [652.5, ..., 800],   // R-R intervals in milliseconds
      "hr": [91.0, ...]},                    // Cardiac rythm in bpm

"swt": {...},                                // Computation method Stationnary Wavelet - swt
"xqrs": {...},                               // Computation method XQRS

"score": {
  "corrcoefs":  [[1, 0.3287671232876712, 0.40540540540540543], //  Correlation coefficient matrix
                [0.3287671232876712, 1, 0.8524590163934426],   // Pan-Tompkins | SWT | XQRS
                [0.40540540540540543, 0.8524590163934426, 1]], // example corrcoef[0][2] -> Correlation between Pan Tompkins and XQRS
    "matching_frames": [[43, 12, 15], [12, 30, 26], [15, 26, 31]], // Not useful
     "missing_beats_duration": [[0, 0.0, 0.0],                 // Missing beats periods duration
                                [0.0, 0, 0.0],                 
                                [0.0, 0.0, 0]]}}               s
```

Here is an example of a representation of the heart rate with the 3 methods for a given 30-minute examination:

<img src="./images/RythmeCardiaque.png" alt="Cardiac rythm" width="800"/></br>
*Heart rate displayed for a 30-minute exam*</br>

## The annotations in the database

The database includes annotations separated into 2 classes:
 * background,
 * seizure

These annotations are formatted as an interval with a beginning and an end measured by a time lag relative to the beginning of the examination.

### The available data
All annotation data is stored in JSON files, one per scan in the **annot-v0_4** folder on the Drive.

JSON files follow the following format:
```json
{"background": [[0.0, 80.5],        // Intervals without seizures
                [121.0, 185.0]],
 "seizure": [80.5, 121.0]}          // Intervals with seizures
```
## Calculation of relevant medical indicators - the features
To go further, we will now extract a set of business features from the R-R intervals.
From a medical point of view, the analysis of cardiac variability is based on 3 categories of indicators proposed by European and American cardiology societies:
 * Time domain indicators
 * Frequency domain indicators
 * Non-linear indicators


 We divided each examination **in intervals of 10 seconds** then calculated 28 indicators detailed below as well as the associated label on the whole dataset.

 An important precision, the time domain indicators are calculated from **a 10-second sliding window**, the frequency domain indicators are calculated from **a 2min30 (150 seconds)** sliding window, and the non-linear indicators are calculated from **a 1min30** sliding window.

 These windows are chosen to ensure robust feature calculation.

<img src="./images/fenetres_glissantes.png" alt="Fenêtres de calcul" width="800"/></br>
*Compuration window for the differents indicators* </br>


**Indicators list:**
```json
FEATURES_KEY_TO_INDEX = {
    'interval_index':0,                            // intervals index
    'interval_start_time':1, #en milliseconds      //

    'mean_nni': 2,                                 // Time domain indicators
    'sdnn': 3,
    'sdsd': 4,
    'nni_50': 5,
    'pnni_50': 6,
    'nni_20': 7,
    'pnni_20': 8,
    'rmssd': 9,
    'median_nni': 10,
    'range_nni': 11,
    'cvsd': 12,
    'cvnni': 13,
    'mean_hr': 14,
    'max_hr': 15,
    'min_hr': 16,
    'std_hr': 17,

    'lf': 18,                                    // Frequency domain indicators
    'hf': 19,
    'vlf': 20,
    'lf_hf_ratio': 21,

    'csi': 22,                                   // Non linear indicators
    'cvi': 23,
    'Modified_csi': 24,
    'sampen': 25,
    'sd1': 26,
    'sd2': 27,
    'ratio_sd2_sd1': 28,

    'label': 29                                  // Label: 0        if no seizure  
                                                 //        >0 et <1 if interval include partial seizure time (ratio between 0 and 1)
                                                 //        1        if seizure
}
```
You will find detailed information on these indicators [HERE](https://github.com/Aura-healthcare/hrvanalysis)

You are free to add more, the script that generated them is present on the Drive and runs as follows.

```bash
python3 Cardiac_features_computation_wrapper.py -i res_file.json -a annot_file.json -q QRS_method -o my_new_feats_file.json
```

### The available data
All feature data is stored in JSON files, 3 per scan in the **feats-v0_4** folder on the Drive. For each scan, we have stored one feature file for each method of calculating R-R intervals (Pan tompkins, XQRS or SWT).

JSON files follow the following format:
```json
{"keys": ["interval_index", "interval_start_time", "mean_nni", ... ], # Index - feature mapping keys

 "features":   # Array of features computed for each interval
 [
   [0.0, 0.0, 595.859375, 110.79985592462053, 42.63426504064021, 2.0, 12.5, 2.0, 12.5, 47.899634654139064, 655.0, 327.5, 0.0803874817848407, 0.18594967298218734, 105.20654097423254, 170.2127659574468, 88.23529411764706, 25.419697501831912, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.0],
   # Ensemble of the computed features on the interval 0 (0s -> 10s) - feature mapping is detailed in the mapping keys

   [1.0, 10000.0, NaN, NaN, NaN, 4.0, 22.22222222222222, 11.0, 61.111111111111114, NaN, NaN, 340.0, NaN, NaN, NaN, 176.47058823529412, 88.23529411764706, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0.0]
   # Ensemble of the computed features on the interval 0 (10s -> 20s) - feature mapping is detailed in the mapping keys
   ...

   [... ] # Last interval of the exam
 ]
 }
```

## The Seizure detection system ##
 ** It's up to you now! **
