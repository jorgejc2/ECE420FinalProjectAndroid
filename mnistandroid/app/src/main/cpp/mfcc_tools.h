//
// Created by jorgejc2 on 4/11/23.
//

#ifndef MNISTANDROID_MFCC_TOOLS_H
#define MNISTANDROID_MFCC_TOOLS_H

#include <vector>
#include <cmath>

using namespace std;

namespace mfcc {

    void int16ToFloat(const int16_t* original_samples, float* new_samples, int num_samples);

    void preemphasis(float* samples, int num_samples, float b);

    int getMelFilterBanks (float** MelFilterArray, int nfft, int numFilters, int sampleRate);

    int calculateDCTCoefficients(float** DCTArray, int melCoeffecients, int numFilters);

    void gemmMultiplication(const float* in_1, const float* in_2, float* out, int m, int k, int n);

    int downsample(float* samples, float** new_samples, int num_samples, int original_sampling_rate, int new_sampling_rate);

    void applyFirFilter(float* samples, int num_samples);

    void applyFirFilterSeries(float* samples, int num_samples);

    float firFilter(float sample);

    void resetFirCircBuf();

    void viewMelFilters(float* mel_filter_array, float* output, int mel_filter_size);

}

#endif //MNISTANDROID_MFCC_TOOLS_H
