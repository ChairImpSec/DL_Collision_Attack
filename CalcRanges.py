import numpy as np

# Parameters - calculate range for target-byte based on results from learned-byte
learned = 10
target = 7
known_offset = False

# Ranges determined with our approach - change if necessary
byte_start = [0, 0, 45500, 33050, 48409, 41200, 37100, 35050, 26736, 40188, 28814, 43750, 20500, 22000, 49200, 17700]
byte_end = [0, 0, 46300, 33850, 49209, 41950, 37900, 35750, 27486, 41188, 29614, 44550, 21300, 22800, 50200, 18500]

startTrained1 = byte_start[learned]
endTrained1 = byte_end[learned]

# Following parameters are only required if known_offset = False
# Range that was used for sensitivity analysis
sa_learned_start = 28464
sa_learned_end = 30464
# Ranges target byte
sa_target_start = 34742
sa_target_end = 36742

if known_offset:
    offset = 2082
    byte_pos = [6, 3, 13, 7, 14, 11, 9, 8, 4, 10, 5, 12, 1, 2, 15, 0]
    start_sample = (startTrained1 - (byte_pos[learned] - byte_pos[target]) * offset)
    end_sample = (endTrained1 - (byte_pos[learned] - byte_pos[target]) * offset)
    print(f'{start_sample}-{end_sample}')
else:
    f1 = open("weights_{}_{}_{}.dat".format(learned, sa_learned_start, sa_learned_end), "rb")
    f2 = open("weights_{}_{}_{}.dat".format(target, sa_target_start, sa_target_end), "rb")

    weights_2 = np.fromfile(f1, dtype="float32")
    weights_4 = np.fromfile(f2, dtype="float32")

    startT = startTrained1 - sa_learned_start
    endT = endTrained1 - sa_learned_start
    lengthT = endT - startT

    lengthCorr = len(weights_4) - lengthT
    pcorr = np.zeros(lengthCorr)
    for i in range(lengthCorr):
        pcorr[i] = (np.corrcoef(weights_2[startT:endT], weights_4[i:(i + lengthT)])[0][1])
    corrPArgMax = np.argmax(pcorr)
    print(f'{sa_target_start + corrPArgMax}-{sa_target_start + corrPArgMax + lengthT}')
