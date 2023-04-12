//
// Created by jorgejc2 on 4/11/23.
//

#ifndef MNISTANDROID_MFCC_TOOLS_H
#define MNISTANDROID_MFCC_TOOLS_H

#include <vector>
#include <cmath>

using namespace std;

namespace mfcc {

    void preemphasis(float* samples, int num_samples, float b);

    int getMelFilterBanks (float** MelFilterArray, int nfft, int numFilters, int frameSize, int sampleRate);

    int calculateDCTCoefficients(float** DCTArray, int melCoeffecients, int numFilters);

    void gemmMultiplication(const float* in_1, const float* in_2, float* out, int m, int k, int n);

    int downsample(float* samples, float** new_samples, int num_samples, int original_sampling_rate, int new_sampling_rate);

    void applyFirFilter(float* samples, int num_samples);

    float firFilter(float sample);

    void resetFirCircBuf();

}

#endif //MNISTANDROID_MFCC_TOOLS_H