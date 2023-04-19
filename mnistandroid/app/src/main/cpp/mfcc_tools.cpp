//
// Created by jorgejc2 on 4/11/23.
//

#include "mfcc_tools.h"
#include <cassert>

using namespace std;

#define N_TAPS 51
const float fir_coefficients [N_TAPS] = {-0.0009154526549517794, -0.0011114542578674472, -0.0011748111621119072, -0.0009912202839374121, -0.00040566875669237703, 0.0006928645047039383, 0.0022655744122396246, 0.004040327999447895, 0.005495567242988037, 0.005941591839728333, 0.004700288764397637, 0.0013495643820975742, -0.004031166285818187, -0.010692574509175845, -0.017194526140070547, -0.021572582846390315, -0.021684691762467997, -0.015679166273036733, -0.002483205556162394, 0.017807916482357168, 0.043749754550979104, 0.07264355863368925, 0.10091303015030824, 0.12470091418085558, 0.14056680266815558, 0.14613752935346994, 0.14056680266815555, 0.12470091418085556, 0.10091303015030822, 0.07264355863368924, 0.0437497545509791, 0.01780791648235716, -0.0024832055561623935, -0.015679166273036726, -0.021684691762467987, -0.021572582846390312, -0.017194526140070544, -0.01069257450917584, -0.004031166285818185, 0.0013495643820975731, 0.004700288764397637, 0.005941591839728333, 0.005495567242988034, 0.004040327999447891, 0.0022655744122396216, 0.0006928645047039375, -0.00040566875669237703, -0.0009912202839374117, -0.0011748111621119064, -0.0011114542578674466, -0.0009154526549517794};
// Circular Buffer
float circBuf[N_TAPS] = {};
int circBufIdx = 0;

namespace mfcc {

    void int16ToFloat(const int16_t* original_samples, float* new_samples, int num_samples) {
        for (int i = 0; i < num_samples; i++)
//            new_samples[i] = float((original_samples[i]>>2)/(32768.0));
            new_samples[i] = float(original_samples[i]/32768.0);
    }

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
    int getMelFilterBanks (float** MelFilterArray, int nfft, int numFilters, int sampleRate) {

        /* initializing the numFilters x (nfft / 2 + 1) filter bank 2d array */
        int fbank_rows = numFilters;
        int fbank_cols = int(nfft / 2 + 1);
        int fbank_size = fbank_rows * fbank_cols;
        float * curr_bank = new float[fbank_size];

        /* bank holding the mel filter banks */
        #define curr_bank_(i1, i0) curr_bank[(i1*fbank_cols) + i0]

        /* lower and upper mel frequency bound */
        float low_freq = 0;
        float high_freq = 2595.0 * log10f(1.0 + ((sampleRate/2)/700.0));

        int num_bins = numFilters + 2;

        float mel_step = (high_freq - low_freq) / (num_bins - 1.0);
        int bins[num_bins];

        float mel_point;
        float hz_point;
        for (int i = 0; i < num_bins; i++) {
            mel_point = i * mel_step;
            hz_point = 700 * ( powf(10.0, mel_point / 2595.0) - 1.0 );
            bins[i] = int( (nfft + 1) * hz_point / sampleRate );
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
            float curr_val;
            for (int k = f_m_minus; k < f_m; k++) {
                curr_val = (float)((2.0*(k - bins[m-1])) / (bins[m] - bins[m - 1]));
                assert(curr_val >= 0);
                curr_bank_(m-1, k) = curr_val;
            }
            for (int k = f_m; k < f_m_plus; k++) {
                curr_val = (float)((2.0*(bins[m+1] - k)) / (bins[m + 1] - bins[m]));
                assert(curr_val >= 0);
                curr_bank_(m-1, k) = curr_val;
            }
        }

        for (int i = 0; i < fbank_size; i++)
            curr_bank[i] = 1;

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
        #define curr_dct_(i1, i0) curr_dct[(i1 * numFilters) + i0]

        /* fill in the dct array */
        for (int n = 0; n < melCoeffecients; n++) {
            for (int m = 0; m < numFilters; m++) {
                curr_dct_(n, m) = cosf((M_PI * n * (m-0.5)) / numFilters);
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
//        #define in_1_(i1, i0) in_1[(i1 * k) + i0]
//        #define in_2_(i1, i0) in_2[(i1 * n) + i0]
//        #define out_(i1, i0) out[(i1 * n) + i0]

        /* iterate through input 1's rows (k) */
        for (int row = 0; row < m; row++) {
            /* iterate through input 2's columns (n) */
            for (int col = 0; col < n; col++) {
                /* iterate through the shared dimension and calculate cell */
//                out_(row, col) = 0.0; // initialize
                out[row * n + col] = 0.0;
                float sum = 0.0;
                for (int i = 0; i < k; i++) {
//                    float result = in_1_(row, i) * in_2_(i, col);
                    sum += in_1[row * k + i] * in_2[i*n + col];
//                    out_(row, col) += (std::isinf(result) || std::isnan(result)) ? 0 : result;
//                    out_(row, col) += result;
                }
                out[row *n + col] = sum;
            }
        }

//        #undef in_1_
//        #undef in_2_
//        #undef out_
    };

    int downsample(float* samples, float** new_samples, int num_samples, int original_sampling_rate, int new_sampling_rate) {

        /* calculate the factor by which we have to down sample based on the original and new sampling rate */
        int down_sampling_factor = original_sampling_rate / new_sampling_rate; // based on our parameters, this should be 6
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

    void applyFirFilterSeries(float* samples, int num_samples) {
        float circBuf_[N_TAPS] = {};
        int circBufIdx_ = 0;

        float sum;
        for (int i = 0; i < num_samples; i++) {
            circBuf_[circBufIdx_] = samples[i];

            sum = 0.0;

            for (int n = 0; n < N_TAPS; n++) {
                sum += fir_coefficients[n] * circBuf_[(((circBufIdx_ - n) % N_TAPS) + N_TAPS) % N_TAPS];
            }

            samples[i] = sum;
            circBufIdx_ = (circBufIdx_ + 1) % N_TAPS;
        }

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
        float output = 0.0;

        /* insert sample into buffer */
        circBuf[circBufIdx] = sample;

        /* perform convolution */
        for (int i = 0; i < N_TAPS; i++) {
            output += fir_coefficients[i] * circBuf[ (((circBufIdx - i) % N_TAPS) + N_TAPS) % N_TAPS];
        }
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