//
// Created by jorgejc2 on 4/11/23.
//

#ifndef MNISTANDROID_MFCC_TOOLS_H
#define MNISTANDROID_MFCC_TOOLS_H

#include <vector>
#include <cmath>

using namespace std;

namespace mfcc {

    int getMelFilterBanks (float** MelFilterArray, int nfft, int numFilters, int frameSize, int sampleRate);

    int calculateDCTCoefficients(float** DCTArray, int melCoeffecients, int numFilters);

    void gemmMultiplication(const float* in_1, const float* in_2, float* out, int m, int k, int n);
}

#endif //MNISTANDROID_MFCC_TOOLS_H
