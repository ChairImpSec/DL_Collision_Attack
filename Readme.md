# Deep Learning Side-Channel Collision Attack
This repository contains source code related to the article [Deep Learning Side-Channel Collision Attack](https://eprint.iacr.org/2023/???.pdf) and provides some easy examples. Please note that the provided scripts have been intentionally condensed to their fundamental functionality and deliberately kept brief and straightforward.

## Short Description
We present a new methodology, that is able to exploit side-channel collisions in a black-box setting. In particular, this attack can be performed in a non-profiled setting and requires neither a hypothetical power model nor details about the underlying implementation. While the existing non-profiled DL attacks utilize training metrics to distinguish the correct key (e.g., [here](https://tches.iacr.org/index.php/TCHES/article/view/7387/6559)), this attack is more efficient by training a model that can be applied to recover multiple key portions, e.g., bytes. In order to perform this attack on raw traces instead of pre-selected samples, we further introduce a DL-based technique that can localize input-dependent leakages in masked implementations, e.g., the leakages associated to one byte of the cipher state in case of AES. 

## Requirements
To run the Python scripts, a working installation of *TensorFlow*, *keras*, and *scikit-learn* is required. GPU acceleration can be beneficial when working with large datasets.


## Getting Started

- Step 1: Execute `LeakageIdentification.py` to obtain the leaking position for a specific key portion. You should execute it with a wide range of sample points first and select a smaller range around the peak afterwards (see Section 3.2 in paper). The final range is determined by visual inspection.
- Step 2: Execute `TrainingPhase.py` with the previously determined leaking range. You will train a corresponding model that can be used to find collisions in the other key portions.
- Step 3: When the offsets between the processing of different portions are known, you can either calculate the ranges for the other bytes yourself or use `CalcRanges.py`. If the offsets are unknown, you have to repeat Step 1 and receive the output weight-files. `CalcRanges.py` will help you to calculate the range for the target byte based on the weight-files.
- Step 4: Execute `AttackPhase.py` with the previously calculated ranges for the target byte.


## Examples
All python scripts should be executable without any prior changes on the parameters to target the seventh byte of the ASCADv1 dataset. The dataset can be found [here](https://github.com/ANSSI-FR/ASCAD/blob/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key/Readme.md), all other required files are contained in the [Examples](https://github.com/ChairImpSec/DL_Collision_Attack/Examples) folder. When you aim to target any other bytes, make sure to follow the order of execution as explained in the paper.

## Contact and Support
Please contact Marvin Staib (marvin.staib@rub.de) if you have any questions, comments or if you found a bug that should be corrected.

## Licensing
Please see `LICENSE` for licensing instructions.

## Publication
[Deep Learning Side-Channel Collision Attack](https://eprint.iacr.org/2023/???.pdf).
