//
// Created by jorgejc2 on 4/11/23.
//

#include "mfcc_tools.h"

using namespace std;

#define N_TAPS 51
const float fir_coefficients [N_TAPS] = {-0.01141003352268849, -0.013262878999151668, -0.012421570191556987, -0.008824561658407391, -0.0029776637056192915, 0.004098971913804342, 0.011030338450131785, 0.016329730040982957, 0.018685369327607042, 0.017239493231304706, 0.011806989763287312, 0.002988485010502018, -0.007850704665670786, -0.018741751293484914, -0.02739481121155779, -0.03156139343863706, -0.029418861185771146, -0.01991252167772265, -0.002994989095316821, 0.020282611087152663, 0.047849778038868376, 0.07686050260107538, 0.10406863704643883, 0.12627886838288652, 0.14080366814266507, 0.1458543018661343, 0.14080366814266507, 0.12627886838288652, 0.10406863704643883, 0.07686050260107538, 0.047849778038868376, 0.020282611087152663, -0.002994989095316821, -0.01991252167772265, -0.029418861185771146, -0.03156139343863706, -0.02739481121155779, -0.018741751293484914, -0.007850704665670786, 0.002988485010502018, 0.011806989763287312, 0.017239493231304706, 0.018685369327607042, 0.016329730040982957, 0.011030338450131785, 0.004098971913804342, -0.0029776637056192915, -0.008824561658407391, -0.012421570191556987, -0.013262878999151668, -0.01141003352268849};
// Circular Buffer
float circBuf[N_TAPS] = {};
int circBufIdx = 0;

namespace mfcc {

    /* MFCC algorithms */
    void preemphasis(float* samples, int num_samples, float b) {

        float prev_sample = 0.0;

        for (int i = 0; i < num_samples; i++) {
            samples[i] = samples[i] - b*prev_sample;
            prev_sample = samples[i];
        }

        return;
    }

    /*
     * Description
     */
    int getMelFilterBanks (float** MelFilterArray, int nfft, int numFilters, int frameSize, int sampleRate) {

        /* initializing the numFilters x frameSize filter bank 2d array */
        int fbank_rows = numFilters;
        int fbank_cols = nfft / 2 + 1;
        int fbank_size = fbank_rows * fbank_cols;
        float * curr_bank = new float[fbank_size];
        int width = (nfft/2) + 1;

        /* bank holding the mel filter banks */
        #define curr_bank_(i1, i0) curr_bank[(i1*width) + i0]

        /* lower and upper mel frequency bound */
        float low_freq = 0;
        float high_freq = 2595.0 * log10(1 + sampleRate/700.0);

        int num_bins = numFilters + 2;

        float mel_step = (high_freq - low_freq) / (num_bins - 1.0);
        int bins[num_bins];

        float mel_point;
        float hz_point;
        for (int i = 0; i < num_bins; i++) {
            mel_point = i * mel_step;
            hz_point = 700 * ( pow(10.0, mel_point / 2595.0) - 1 );
            bins[i] = int( (nfft + 1) * hz_point / (sampleRate*2) );
        }

        int f_m_minus;
        int f_m;
        int f_m_plus;

        /* calculating the mel filter banks */
        for (int m = 1; m < numFilters + 1; m++) {
            f_m_minus = bins[m - 1];
            f_m = bins[m];
            f_m_plus = bins[m + 1];

            /* in order to coalescence memory, I store the transpose of fbank */
            for (int k = f_m_minus; k < f_m; k++) {
                curr_bank_(m-1, k) = (2*(k - bins[m-1])) / (bins[m] - bins[m - 1]);
            }
            for (int k = f_m; k < f_m_plus; k++) {
                curr_bank_(m-1, k) = (2*(bins[m+1] - k)) / (bins[m + 1] - bins[m]);
            }
        }

        *MelFilterArray = curr_bank;

        return fbank_size;

        #undef curr_bank_
    };

    /*
     * Description:
     */
    int calculateDCTCoefficients(float** DCTArray, int melCoeffecients, int numFilters) {
        int dct_size = melCoeffecients * numFilters;
        float* curr_dct = new float[dct_size];
        #define curr_dct_(i1, i0) curr_dct[i1 * numFilters + i0]

        /* fill in the dct array */
        for (int n = 0; n < melCoeffecients; n++) {
            for (int m = 0; m < numFilters; m++) {
                curr_dct_(n, m) = cos(M_PI * n * (m-0.5) / numFilters);
            }
        }

        /* return the newly created dct array */
        *DCTArray = curr_dct;

        return dct_size;

        #undef curr_dct_
    };

    /*
     * Description: Performs basic gemm multiplication of input 1 of dimensions mxk with input 2 of
     * dimensions kxn. Stores output as mxn array
     * Inputs:
     *          in_1 -- input array 1
     *          in_2 -- input array 2
     *          m -- rows of input array 1
     *          k -- shared dimension between array 1 and 2
     *          n -- columns of input array 2
     * Outputs:
     *          out -- output
     * Returns:
     *          None
     */
    void gemmMultiplication (const float* in_1, const float* in_2, float* out, int m, int k, int n) {
        #define in_1_(i1, i0) in_1[i1 * k + i0]
        #define in_2_(i1, i0) in_2[i1 * n + i0]
        #define out_(i1, i0) out[i1 * n + i0]

        /* iterate through input 1's rows (k) */
        for (int row = 0; row < m; row++) {
            /* iterate through input 2's columns (n) */
            for (int col = 0; col < n; col++) {
                /* iterate through the shared dimension and calculate cell */
                for (int i = 0; i < k; i++) {
                    out_(row, col) += in_1_(row, i) * in_2_(i, col);
                }
            }
        }

        #undef in_1_
        #undef in_2_
        #undef out_
    };

    int downsample(float* samples, float** new_samples, int num_samples, int original_sampling_rate, int new_sampling_rate) {

        /* calculate the factor by which we have to down sample based on the original and new sampling rate */
        int down_sampling_factor = original_sampling_rate / new_sampling_rate;
        int new_samples_size = num_samples / down_sampling_factor;
        float* new_samples_ = new float[new_samples_size];

        /* down sample */
        for (int i = 0; i < num_samples; i+=down_sampling_factor) {
            new_samples_[i / down_sampling_factor] = samples[i];
        }

        /* return newly downsampled array */
        *new_samples = new_samples_;

        /* return size of newly downsamples array */
        return new_samples_size;
    }

    void applyFirFilter(float* samples, int num_samples) {
        /* use the firFilter helper function that applies the filter to an individual sample */
        for (int i = 0; i < num_samples; i++) {
            samples[i] = firFilter(samples[i]);
        }
        /* reset the global value used for the fir filter */
        resetFirCircBuf();

        return;
    }

    // FirFilter Function
    float firFilter(float sample) {
        // This function simulates sample-by-sample processing. Here you will
        // implement an FIR filter such as:
        //
        // y[n] = a x[n] + b x[n-1] + c x[n-2] + ...
        //
        // You will maintain a circular buffer to store your prior samples
        // x[n-1], x[n-2], ..., x[n-k]. Suggested initializations circBuf
        // and circBufIdx are given.
        //
        // Input 'sample' is the current sample x[n].
        // ******************** START YOUR CODE HERE ******************** //
        float output;
        double inter = 0.0; // to preserve accuracy during the calculation

        /* insert sample into buffer */
        circBuf[circBufIdx] = sample;

        /* perform convolution */
        for (int i = 0; i < N_TAPS; i++) {
            inter += fir_coefficients[i] * circBuf[ (((circBufIdx - i) % N_TAPS) + N_TAPS) % N_TAPS];
        }
        output = float(inter);
        /* update pointer */
        circBufIdx = (circBufIdx + 1) % N_TAPS;
        // ********************* END YOUR CODE HERE ********************* //
        return output;
    }

    void resetFirCircBuf() {
        circBufIdx = 0;
        for (int i = 0; i < N_TAPS; i++)
            circBuf[i] = 0;
    }

};