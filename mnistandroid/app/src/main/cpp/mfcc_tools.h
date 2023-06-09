//
// Created by jorgejc2 on 4/11/23.
//

#ifndef MNISTANDROID_MFCC_TOOLS_H
#define MNISTANDROID_MFCC_TOOLS_H

#include <vector>
#include <cmath>

using namespace std;

namespace mfcc {

    int createWav(int16_t* samples, uint8_t** wavFile, int num_samples, int sample_rate);

    void createImage(float* samples, uint32_t* image_out, int pixel_width, int pixel_height, int samples_rows, int samples_cols);

    void int16ToFloat(const int16_t* original_samples, float* new_samples, int num_samples);

    void floatToInt16(const float* original_samples, int16_t* new_samples, int num_samples);

    void normalizeData(float* samples, int num_samples);

    void preemphasis(float* samples, int num_samples, float b);

    int getMelFilterBanks (float** MelFilterArray, int nfft, int numFilters, int sampleRate);

    int calculateDCTCoefficients(float** DCTArray, int melCoeffecients, int numFilters);

    void gemmMultiplication(const float* in_1, const float* in_2, float* out, int m, int k, int n);

    int downsample(float* samples, float** new_samples, int num_samples, int original_sampling_rate, int new_sampling_rate);

    void applyFirFilterSeries(float* samples, int num_samples);
}

#endif //MNISTANDROID_MFCC_TOOLS_H
