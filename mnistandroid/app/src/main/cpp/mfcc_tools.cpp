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

typedef struct  WAV_HEADER
{
    /* RIFF Chunk Descriptor */
    uint8_t         RIFF[4];        // RIFF Header Magic header
    uint32_t        ChunkSize;      // RIFF Chunk Size
    uint8_t         WAVE[4];        // WAVE Header
    /* "fmt" sub-chunk */
    uint8_t         fmt[4];         // FMT header
    uint32_t        Subchunk1Size;  // Size of the fmt chunk
    uint16_t        AudioFormat;    // Audio format 1=PCM,6=mulaw,7=alaw,     257=IBM Mu-Law, 258=IBM A-Law, 259=ADPCM
    uint16_t        NumOfChan;      // Number of channels 1=Mono 2=Sterio
    uint32_t        SamplesPerSec;  // Sampling Frequency in Hz
    uint32_t        bytesPerSec;    // bytes per second
    uint16_t        blockAlign;     // 2=16-bit mono, 4=16-bit stereo
    uint16_t        bitsPerSample;  // Number of bits per sample
    /* "data" sub-chunk */
    uint8_t         Subchunk2ID[4]; // "data"  string
    uint32_t        Subchunk2Size;  // Sampled data length
} wav_hdr;

namespace mfcc {

    int createWav(int16_t* samples, uint8_t** wavFile, int num_samples, int sample_rate) {
        /* setting up header */

        int wavFileSize = sizeof(wav_hdr) + (num_samples * 2);
        uint8_t* wavFile_ = new uint8_t[wavFileSize];

        wav_hdr wav_header;
        wav_header.RIFF[0] = 'R';
        wav_header.RIFF[1] = 'I';
        wav_header.RIFF[2] = 'F';
        wav_header.RIFF[3] = 'F';

        wav_header.WAVE[0] = 'W';
        wav_header.WAVE[1] = 'A';
        wav_header.WAVE[2] = 'V';
        wav_header.WAVE[3] = 'E';

        wav_header.fmt[0] = 'f';
        wav_header.fmt[1] = 'm';
        wav_header.fmt[2] = 't';
        wav_header.fmt[3] = ' ';

        wav_header.Subchunk1Size = 16; // 16 for PCM
        wav_header.AudioFormat = 1; // 1 for PCM
        wav_header.NumOfChan = 1; // 1 for mono
        wav_header.SamplesPerSec = sample_rate;
        wav_header.bitsPerSample = 16;
        wav_header.bytesPerSec = (sample_rate * 1 * 16) / 8;
        wav_header.blockAlign = 2; // 2 = 16 bit mono

        wav_header.Subchunk2ID[0] = 'd';
        wav_header.Subchunk2ID[1] = 'a';
        wav_header.Subchunk2ID[2] = 't';
        wav_header.Subchunk2ID[3] = 'a';
        wav_header.Subchunk2Size = (num_samples * 1 * 16) / 8; // size of the data past the header in bytes

        wav_header.ChunkSize = 36 + wav_header.Subchunk2Size; // size of entire file excluding ChunkID and ChunkSize (minus 8 bytes)

        uint8_t* wav_header_byte = (uint8_t*)(&wav_header); // cast wav_header into byte format
        /* copy over the header file */
        for (int i = 0; i < sizeof(wav_hdr); i++) {
            wavFile_[i] = wav_header_byte[i];
        }

        int start_idx = sizeof(wav_hdr); // starting index of where data will get copied to

        /* note that data is stored as little endian */
        int16_t mask = 0x00FF;
        int wavFileIdx1, wavFileIdx2;
        for (int i = 0; i < num_samples; i++) {
            wavFileIdx1 =  start_idx + i * 2;
            wavFileIdx2 = wavFileIdx1 + 1;
            wavFile_[wavFileIdx1] = mask & samples[i];
            wavFile_[wavFileIdx2] = mask & (samples[i] >> 8);
        }

        *wavFile = wavFile_;

        return wavFileSize;
    }

    void createImage(float* samples, uint32_t* image_out, int pixel_width, int pixel_height, int samples_rows, int samples_cols) {
        const int viridis_size = 20;
        int num_samples = samples_rows * samples_cols;
        uint32_t viridis_palette[viridis_size] = {
                0x440154,
                0x481567,
                0x482677,
                0x453771,
                0x404788,
                0x39568C,
                0x33638D,
                0x2D708E,
                0x287D8E,
                0x238A8D,
                0x1F968B,
                0x20A387,
                0x29AF7F,
                0x3CBB75,
                0x55C667,
                0x73D055,
                0x95D840,
                0xB8DE29,
                0xDCE319,
                0xFDE725
        };

        /* normalize the data */
        float *normalized_samples = new float[num_samples];
        int num_above_10 = 0;
        int num_below_neg_10 = 0;
        for (int i = 0; i < num_samples; i++) {
            if (samples[i] > 10) {
                normalized_samples[i] = 10;
                num_above_10++;
            }
            else if (samples[i] < -10) {
                normalized_samples[i] = -10;
                num_below_neg_10++;
            }
            else
                normalized_samples[i] = samples[i];
            }
        normalizeData(normalized_samples, num_samples);

        /* fill in canvas based on linearly normalized samples */
        int horizontal_step = pixel_width / samples_cols;
        int vertical_step = pixel_height / samples_rows;
        int horizontal_count = 1;
        int vertical_count = 1;
        int x_idx, y_idx, viridis_idx;
        uint32_t r, g, b;
        uint32_t mask = 0x0000FF;

        float percent;
        for (int pixel_row = 0; pixel_row < pixel_height; pixel_row++) {
            if (pixel_row >= vertical_count * vertical_step && vertical_count < samples_rows)
                vertical_count++;

            horizontal_count = 1;
            for (int pixel_col =  0; pixel_col < pixel_width; pixel_col++) {
                if (pixel_col >= horizontal_count * horizontal_step && horizontal_count < samples_cols)
                    horizontal_count++;

                x_idx = horizontal_count - 1;
                y_idx = samples_rows - vertical_count;

                percent = normalized_samples[y_idx * samples_cols + x_idx];

                if (percent > 1)
                    percent = 1;
                if (percent < 0)
                    percent = 0;

                viridis_idx = (viridis_size - 1) * percent;
                b = mask & viridis_palette[viridis_idx];
                g = mask & (viridis_palette[viridis_idx] >> 8);
                r = mask & (viridis_palette[viridis_idx] >> 16);

                image_out[pixel_row * pixel_width + pixel_col] = (0xFF << 24) | (b << 16) | (g << 8) | r;
            }
        }

        /* free normalized samples */
        delete [] normalized_samples;
    }

    void int16ToFloat(const int16_t* original_samples, float* new_samples, int num_samples) {
        for (int i = 0; i < num_samples; i++)
//            new_samples[i] = float((original_samples[i]>>2)/(32768.0));
            new_samples[i] = float(original_samples[i]/32768.0);
    }

    void floatToInt16(const float* original_samples, int16_t* new_samples, int num_samples) {
        for (int i = 0; i < num_samples; i++)
            new_samples[i] = int16_t(original_samples[i] * 32768.0);
    }

    /* normalizes samples between 0 and 1 */
    void normalizeData(float* samples, int num_samples) {
        float max_val = std::numeric_limits<float>::min();
        float min_val = std::numeric_limits<float>::max();

        /* find the min and max values from the audio clip */
        float curr_sample;
        for (int i = 0; i < num_samples; i++) {
            curr_sample = samples[i];
            if (std::isnan(curr_sample)) {
                curr_sample = 0;
                samples[i] = 0;
            }
            if (curr_sample > max_val)
                max_val = curr_sample;
            if (curr_sample < min_val)
                min_val = curr_sample;
        }

        /* adjust max_val */
        max_val -= min_val;

        /* normalize between 0 and 1 */
        for (int i = 0; i < num_samples; i++) {
            samples[i] = (samples[i] - min_val) / max_val;
        }

        return;
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
            hz_point = 700 * ( pow(10.0, mel_point / 2595) - 1 );
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
                curr_bank[(m-1)*fbank_cols + k] = curr_val;
            }
            for (int k = f_m; k < f_m_plus; k++) {
                curr_val = (float)((2.0*(bins[m+1] - k)) / (bins[m + 1] - bins[m]));
                assert(curr_val >= 0);
                curr_bank[(m-1)*fbank_cols + k] = curr_val;
            }
        }

        *MelFilterArray = curr_bank;

        return fbank_size;
    };

    /*
     * Description:
     */
    int calculateDCTCoefficients(float** DCTArray, int melCoeffecients, int numFilters) {
        int dct_size = melCoeffecients * numFilters;
        float* curr_dct = new float[dct_size];

        /* fill in the dct array */
        for (int n = 0; n < melCoeffecients; n++) {
            for (int m = 0; m < numFilters; m++) {
                curr_dct[n * numFilters + m] = cos((M_PI * n * (m-0.5)) / numFilters);
            }
        }

        /* return the newly created dct array */
        *DCTArray = curr_dct;

        return dct_size;
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

    void viewMelFilters(float* mel_filter_array, float* output, int mel_filter_size) {
        for (int i = 0; i < mel_filter_size; i++)
            output[i] = mel_filter_array[i];
    }

};