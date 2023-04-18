//
// Created by daran on 1/12/2017 to be used in ECE420 Sp17 for the first time.
// Modified by dwang49 on 1/1/2018 to adapt to Android 7.0 and Shield Tablet updates.
//

#include <jni.h>
#include "ece420_main.h"
#include "ece420_lib.h"
#include "kiss_fft/kiss_fft.h"
#include "mfcc_tools.h"

// JNI Function
extern "C" {
JNIEXPORT void JNICALL
    Java_mariannelinhares_mnistandroid_MainActivity_writeNewFreq(JNIEnv *env, jclass, jint);

JNIEXPORT void JNICALL
    Java_mariannelinhares_mnistandroid_MainActivity_getCompleteSamplesBuffer(JNIEnv *env, jclass clazz,
                                                                             jobject bufferPtr);
JNIEXPORT void JNICALL
Java_mariannelinhares_mnistandroid_MainActivity_resetParameters(JNIEnv *env, jclass clazz);

JNIEXPORT void JNICALL
    Java_mariannelinhares_mnistandroid_MainActivity_performMFCC(JNIEnv *env, jclass clazz,
            jobject bufferPtr, jfloatArray outputArray);

JNIEXPORT jintArray JNICALL
    Java_mariannelinhares_mnistandroid_MainActivity_getRowAndCol(JNIEnv *env, jclass clazz);
}

// Student Variables
#define EPOCH_PEAK_REGION_WIGGLE 30
#define VOICED_THRESHOLD 200000000
#define FRAME_SIZE 1024
#define BUFFER_SIZE (3 * FRAME_SIZE)
#define F_S 48000
float bufferIn[BUFFER_SIZE] = {};
float bufferOut[BUFFER_SIZE] = {};
int newEpochIdx = FRAME_SIZE;

// We have two variables here to ensure that we never change the desired frequency while
// processing a frame. Thread synchronization, etc. Setting to 300 is only an initializer.
int FREQ_NEW_ANDROID = 300;
int FREQ_NEW = 300;
bool isWritingSamples = false; // to protect thread that waits for samples to be written
float rawSamples[FRAME_SIZE] = {}; // holds raw samples to be copied over to front end

const int threeSecondSampleSize = F_S * 3;
int threeSecondSamples_idx = 0;
int16_t threeSecondSamples[threeSecondSampleSize] = {};

/* processing parameters */
const int fs = 48000; // sampling rate (pre down sampling)
const int down_sampled_fs = 8000; // sampling rate (after down sampling)
const int nfft = 256; // size of fft frames
const int noverlap = -1; // -1 means to use the default overlap of 50%
const int nfilt = 40; // number of filter banks to create
const int num_ceps = 13; // number of cepstral coefficients to output
const int nn_data_cols = 48; // number of frames to work with
//const int nn_data_cols = 1100;
const int nn_data_rows = 12; // is always num_ceps - 1
bool coeffecients_initialized = false;
float* MelFilterArray = nullptr;
float* DCTArray = nullptr;

/* Hanning Window */
float hanning_window[nfft] = {};
bool hanning_window_initialized = false;
float window_scaling_factor = 0;


bool lab5PitchShift(float *bufferIn_temp) {
    // Lab 4 code is condensed into this function
    int periodLen = detectBufferPeriod(bufferIn);
    float freq = ((float) F_S) / periodLen;

    // If voiced
    if (periodLen > 0) {

        LOGD("Frequency detected: %f\r\n", freq);

        // Epoch detection - this code is written for you, but the principles will be quizzed
        std::vector<int> epochLocations;
        findEpochLocations(epochLocations, bufferIn, periodLen);

        // In this section, you will implement the algorithm given in:
        // https://courses.engr.illinois.edu/ece420/lab5/lab/#buffer-manipulation-algorithm
        //
        // Don't forget about the following functions! API given on the course page.
        //
        // getHanningCoef();
        // findClosestInVector();
        // overlapAndAdd();
        // *********************** START YOUR CODE HERE  **************************** //

        /* calculate new epoch spacing */
        int new_epoch_spacing = F_S / FREQ_NEW;

        /* incremement through new epochs based on spacing */
        for (int i = newEpochIdx; i < 2 * FRAME_SIZE; i += new_epoch_spacing) {
            /* can optimize this by keeping track of the last epoch we mapped to later */
            int curr_epoch_idx = findClosestInVector(epochLocations, i, 0, epochLocations.size());
            int curr_epoch = epochLocations[curr_epoch_idx];
            /* boundary check and find left and right indices for calculating p0 */
            int left_epoch;
            int right_epoch;
            if (curr_epoch_idx == 0)
                left_epoch = 0;
            else
                left_epoch = epochLocations[curr_epoch_idx - 1];
            if (curr_epoch_idx == epochLocations.size() - 1)
                right_epoch = BUFFER_SIZE - 1;
            else
                right_epoch = epochLocations[curr_epoch_idx + 1];

            /* calculate p0 */
            int p0 = (right_epoch - left_epoch) / 2;

            int window_len = 2*p0 + 1;
            /* apply window to input centered around original epoch, and add it to output centered at new epoch */
            for (int j = 0; j < window_len; j++) {
                int windowed_idx = j; // index into window
                int buffer_in_idx = (curr_epoch - p0) + j; // data to use centered around original epoch
                int buffer_out_idx = (i - p0) + j; // location to add windowed data centered around new epoch
                /* only sum overlapped data if indices are valid */
                if ((buffer_out_idx < BUFFER_SIZE && buffer_out_idx > 0) && (buffer_in_idx < BUFFER_SIZE && buffer_in_idx > 0))
                    bufferOut[buffer_out_idx] += getHanningCoef(window_len, windowed_idx) * bufferIn[buffer_in_idx];
            }

        }






        // ************************ END YOUR CODE HERE  ***************************** //
    }

    // Final bookkeeping, move your new pointer back, because you'll be
    // shifting everything back now in your circular buffer
    newEpochIdx -= FRAME_SIZE;
    if (newEpochIdx < FRAME_SIZE) {
        newEpochIdx = FRAME_SIZE;
    }

    return (periodLen > 0);
}

void ece420ProcessFrame(sample_buf *dataBuf) {
    // Keep in mind, we only have 20ms to process each buffer!
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);

    // Get the new desired frequency from android
    FREQ_NEW = FREQ_NEW_ANDROID;

    // Data is encoded in signed PCM-16, little-endian, mono
    int16_t data[FRAME_SIZE];
    for (int i = 0; i < FRAME_SIZE; i++) {
        data[i] = ((uint16_t) dataBuf->buf_[2 * i]) | (((uint16_t) dataBuf->buf_[2 * i + 1]) << 8);
    }

    /* not needed since this was for lab 5 so it wastes time

//    // Shift our old data back to make room for the new data
//    for (int i = 0; i < 2 * FRAME_SIZE; i++) {
//        bufferIn[i] = bufferIn[i + FRAME_SIZE - 1];
//    }
//
//    // Finally, put in our new data.
//    for (int i = 0; i < FRAME_SIZE; i++) {
//        bufferIn[i + 2 * FRAME_SIZE - 1] = (float) data[i];
//    }
//
//    // The whole kit and kaboodle -- pitch shift
//    bool isVoiced = lab5PitchShift(bufferIn);
//
//    if (isVoiced) {
//        for (int i = 0; i < FRAME_SIZE; i++) {
//            int16_t newVal = (int16_t) bufferOut[i];
//
//            uint8_t lowByte = (uint8_t) (0x00ff & newVal);
//            uint8_t highByte = (uint8_t) ((0xff00 & newVal) >> 8);
//            dataBuf->buf_[i * 2] = lowByte;
//            dataBuf->buf_[i * 2 + 1] = highByte;
//        }
//    }
//
//    // Very last thing, update your output circular buffer!
//    for (int i = 0; i < 2 * FRAME_SIZE; i++) {
//        bufferOut[i] = bufferOut[i + FRAME_SIZE - 1];
//    }
//
//    /* the 'past' buffer was perfectly reconstructed and sent, so we can shift it out */
//    for (int i = 0; i < FRAME_SIZE; i++) {
//        bufferOut[i + 2 * FRAME_SIZE - 1] = 0;
//    }
    /* for our purposes, we need to save the samples to the front end */

    for (int i = 0; i < FRAME_SIZE; i++) {
        if (threeSecondSamples_idx + i >= threeSecondSampleSize)
            break;

        threeSecondSamples[threeSecondSamples_idx + i] = data[i];
    }
    if (threeSecondSamples_idx < threeSecondSampleSize)
        threeSecondSamples_idx += FRAME_SIZE;

    gettimeofday(&end, NULL);
    LOGD("Time delay: %ld us",  ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec)));
}

// Returns lag l that maximizes sum(x[n] x[n-k])
int detectBufferPeriod(float *buffer) {

    float totalPower = 0;
    for (int i = 0; i < BUFFER_SIZE; i++) {
        totalPower += buffer[i] * buffer[i];
    }

    if (totalPower < VOICED_THRESHOLD) {
        return -1;
    }

    // FFT is done using Kiss FFT engine. Remember to free(cfg) on completion
    kiss_fft_cfg cfg = kiss_fft_alloc(BUFFER_SIZE, false, 0, 0);

    kiss_fft_cpx buffer_in[BUFFER_SIZE];
    kiss_fft_cpx buffer_fft[BUFFER_SIZE];

    for (int i = 0; i < BUFFER_SIZE; i++) {
        buffer_in[i].r = bufferIn[i];
        buffer_in[i].i = 0;
    }

    kiss_fft(cfg, buffer_in, buffer_fft);
    free(cfg);


    // Autocorrelation is given by:
    // autoc = ifft(fft(x) * conj(fft(x))
    //
    // Also, (a + jb) (a - jb) = a^2 + b^2
    kiss_fft_cfg cfg_ifft = kiss_fft_alloc(BUFFER_SIZE, true, 0, 0);

    kiss_fft_cpx multiplied_fft[BUFFER_SIZE];
    kiss_fft_cpx autoc_kiss[BUFFER_SIZE];

    for (int i = 0; i < BUFFER_SIZE; i++) {
        multiplied_fft[i].r = (buffer_fft[i].r * buffer_fft[i].r)
                              + (buffer_fft[i].i * buffer_fft[i].i);
        multiplied_fft[i].i = 0;
    }

    kiss_fft(cfg_ifft, multiplied_fft, autoc_kiss);
    free(cfg_ifft);

    // Move to a normal float array rather than a struct array of r/i components
    float autoc[BUFFER_SIZE];
    for (int i = 0; i < BUFFER_SIZE; i++) {
        autoc[i] = autoc_kiss[i].r;
    }

    // We're only interested in pitches below 1000Hz.
    // Why does this line guarantee we only identify pitches below 1000Hz?
    int minIdx = F_S / 1000;
    int maxIdx = BUFFER_SIZE / 2;

    int periodLen = findMaxArrayIdx(autoc, minIdx, maxIdx);
    float freq = ((float) F_S) / periodLen;

    // TODO: tune
    if (freq < 50) {
        periodLen = -1;
    }

    return periodLen;
}


void findEpochLocations(std::vector<int> &epochLocations, float *buffer, int periodLen) {
    // This algorithm requires that the epoch locations be pretty well marked

    int largestPeak = findMaxArrayIdx(bufferIn, 0, BUFFER_SIZE);
    epochLocations.push_back(largestPeak);

    // First go right
    int epochCandidateIdx = epochLocations[0] + periodLen;
    while (epochCandidateIdx < BUFFER_SIZE) {
        epochLocations.push_back(epochCandidateIdx);
        epochCandidateIdx += periodLen;
    }

    // Then go left
    epochCandidateIdx = epochLocations[0] - periodLen;
    while (epochCandidateIdx > 0) {
        epochLocations.push_back(epochCandidateIdx);
        epochCandidateIdx -= periodLen;
    }

    // Sort in place so that we can more easily find the period,
    // where period = (epochLocations[t+1] + epochLocations[t-1]) / 2
    std::sort(epochLocations.begin(), epochLocations.end());

    // Finally, just to make sure we have our epochs in the right
    // place, ensure that every epoch mark (sans first/last) sits on a peak
    for (int i = 1; i < epochLocations.size() - 1; i++) {
        int minIdx = epochLocations[i] - EPOCH_PEAK_REGION_WIGGLE;
        int maxIdx = epochLocations[i] + EPOCH_PEAK_REGION_WIGGLE;

        int peakOffset = findMaxArrayIdx(bufferIn, minIdx, maxIdx) - minIdx;
        peakOffset -= EPOCH_PEAK_REGION_WIGGLE;

        epochLocations[i] += peakOffset;
    }
}

void overlapAddArray(float *dest, float *src, int startIdx, int len) {
    int idxLow = startIdx;
    int idxHigh = startIdx + len;

    int padLow = 0;
    int padHigh = 0;
    if (idxLow < 0) {
        padLow = -idxLow;
    }
    if (idxHigh > BUFFER_SIZE) {
        padHigh = BUFFER_SIZE - idxHigh;
    }

    // Finally, reconstruct the buffer
    for (int i = padLow; i < len + padHigh; i++) {
        dest[startIdx + i] += src[i];
    }
}

void applyHanning(float* frame) {
    /* initialize the global hanning window array if it has not already */
    if (!hanning_window_initialized) {
        window_scaling_factor = 0;
        for (int n = 0; n < nfft; n++) {
            hanning_window[n] = (0.5 * (1 - cos(2*(1.0*n/(nfft-1)))));
            window_scaling_factor += hanning_window[n] * hanning_window[n];
        }
    }

    /* apply the hanning winow to the frame */
    for (int i = 0; i < nfft; i++)
        frame[i] *= hanning_window[i];

    return;
}

int performSTFT(float *samples, float** frequencies, int num_samples, int sample_rate, bool oneSided){

    int noverlap_ = noverlap;

    if (noverlap < 0)
        noverlap_ = nfft / 2;

    /* Determine how many FFT's need to be computed */
    int step = nfft - noverlap_;
    int num_ffts = ceil((float)num_samples/step);

    /* trim FFT's that are out of bounds */
    while ( (num_ffts - 1)*step + (nfft - 1) >= num_samples)
        num_ffts--;

    int stft_output_size = num_ffts * (nfft / 2 + 1);
    float* frequencies_ = new float[stft_output_size];

    // Apply fft with KISS_FFT engine
    kiss_fft_cfg cfg = kiss_fft_alloc(nfft, 0, NULL, NULL);
    kiss_fft_cpx in[nfft];
    kiss_fft_cpx out[nfft];

    /* applying fft for all frames */
//    std::vector<std::vector<float>> stft_result(int(nfft/2 + 1), std::vector<float>(num_ffts, 0));
    float curr_samples [nfft];
    for (int n = 0; n < num_ffts; n++) {

        /* get the current samples we want */
        for (int i = 0; i < nfft; i++) {
            curr_samples[i] = samples[n * step];
        }

        /* apply hanning window */
        applyHanning(curr_samples);

        /* copy over frame into kiss_fft buffer */
        for (int i = 0; i < nfft; i++) {
            in[i].r = curr_samples[i];
            in[i].i = 0;
        }

        /* conduct the fft */
        kiss_fft(cfg, in, out);

        /* convert output to decibals and store result */
        if (oneSided) {
            for (int i = 0; i < int(nfft/2 + 1); i++)
//                stft_result[i][n] = 10.0 * log10(2*(out[i].r*out[i].r + out[i].i*out[i].i)/(down_sampled_fs*window_scaling_factor));
//                frequencies_[i * num_ffts + n] = 10.0 * log10(2*(out[i].r*out[i].r + out[i].i*out[i].i)/(down_sampled_fs*window_scaling_factor));
                frequencies_[i * num_ffts + n] = (out[i].r*out[i].r + out[i].i*out[i].i);
        }
        else {
            for (int i = 0; i < nfft; i++)
//                stft_result[i][n] = 10.0 * log10(2*(out[i].r*out[i].r + out[i].i*out[i].i)/(down_sampled_fs*window_scaling_factor));
                frequencies_[i * num_ffts + n] = (out[i].r*out[i].r + out[i].i*out[i].i);
        }
    }
    kiss_fft_free(cfg);

    *frequencies = frequencies_;

    return stft_output_size;
}

int performMFCC(int16_t* samples, float** mfcc_frequencies, int num_samples, int num_frames, float preemphasis_b) {

    float* f_samples = new float[num_samples];

    mfcc::int16ToFloat(samples, f_samples, num_samples);

    /* apply FIR filter */
    mfcc::applyFirFilter(f_samples, num_samples);

    /* downsample */
    float* down_sampled_sig;
    int down_sampled_sig_size = mfcc::downsample(f_samples, &down_sampled_sig, num_samples, fs, down_sampled_fs);

    delete [] f_samples; // no longer needed

    /* apply preemphasis filter */
    mfcc::preemphasis(down_sampled_sig, down_sampled_sig_size, preemphasis_b);

    /* apply stft */
    float* stft_output = nullptr;
    int stft_output_size = performSTFT(down_sampled_sig, &stft_output, down_sampled_sig_size, down_sampled_fs, true);
    int stft_num_frames = stft_output_size / (nfft / 2 + 1);

    delete [] down_sampled_sig; // no longer needed

    float* trimmed_stft_output = new float [(nfft / 2 + 1) * nn_data_cols];

    for (int row = 0; row < (nfft / 2 + 1); row++) {
        for (int col = 0; col < nn_data_cols; col++) {
            trimmed_stft_output[row * nn_data_cols + col] = stft_output[row * stft_num_frames + col];
        }
    }
    delete [] stft_output;

    /* apply the mel filter banks */
    int mel_filtered_output_size = nfilt * num_frames;
    float* mel_filtered_output = new float[mel_filtered_output_size];
//    mfcc::gemmMultiplication(MelFilterArray, stft_output, mel_filtered_output, nfilt, nfft / 2 + 1, num_frames);
    mfcc::gemmMultiplication(MelFilterArray, trimmed_stft_output, mel_filtered_output, nfilt, nfft / 2 + 1, num_frames);

    int numNeg = 0;
    int numZero = 0;
    int inf = 0;
    int nan = 0;
    /* take log base 10 of output before applying DCT */
    for (int i = 0; i < mel_filtered_output_size; i++) {
        float temp = mel_filtered_output[i];
        if (temp == 0)
            numZero++;
        if (temp < 0)
            numNeg++;
        mel_filtered_output[i] = log10f(temp);
        if (std::isinf(mel_filtered_output[i]))
            inf++;
        if (std::isnan(mel_filtered_output[i]))
            nan++;
    }

    /* perform DCT */
    int ceps_output_size = num_ceps * num_frames;
    float* ceps_output = new float [ceps_output_size];

    mfcc::gemmMultiplication(DCTArray, mel_filtered_output, ceps_output, num_ceps, nfilt, num_frames);

    /* no longer need mel_filterd_output array */
    delete [] mel_filtered_output;

    /* trim the output to the desired size */
    int final_output_size = nn_data_cols * nn_data_rows;
    float* final_output = new float[final_output_size];

    for (int i = 0; i < nn_data_rows; i++) {
        for (int j = 0; j < nn_data_cols; j++) {
            final_output[i * nn_data_cols + j] = ceps_output[i * num_frames + j];
        }
    }

    delete [] ceps_output; // no longer needed

    /* return the final output and its size */
    *mfcc_frequencies = final_output;
    return final_output_size;
};


JNIEXPORT void JNICALL
Java_mariannelinhares_mnistandroid_MainActivity_writeNewFreq(JNIEnv *env, jclass, jint newFreq) {
    FREQ_NEW_ANDROID = (int) newFreq;
    return;
}

JNIEXPORT void JNICALL
Java_mariannelinhares_mnistandroid_MainActivity_getCompleteSamplesBuffer(JNIEnv *env, jclass clazz,
                                                                         jobject bufferPtr) {
    // TODO: implement getCompleteSamplesBuffer()
    jshort *buffer = (jshort *) env->GetDirectBufferAddress(bufferPtr);
    // thread-safe, kinda
    for (int i = 0; i < threeSecondSampleSize; i++)
        buffer[i] = threeSecondSamples[i];

    return;
}

JNIEXPORT void JNICALL
Java_mariannelinhares_mnistandroid_MainActivity_resetParameters(JNIEnv *env, jclass clazz) {
    // TODO: implement resetParameters()
    threeSecondSamples_idx = 0;
}

JNIEXPORT void JNICALL
Java_mariannelinhares_mnistandroid_MainActivity_performMFCC(JNIEnv *env, jclass clazz, jobject bufferPtr, jfloatArray outputArray) {
    // TODO: implement performMFCC()
//    jfloat *buffer = (jfloat *) env->GetDirectBufferAddress(bufferPtr);
    /* typecasting should hopefully work since jfloat and float should be the same data struct */
    int16_t * buffer = (jshort *) env->GetDirectBufferAddress(bufferPtr);

    /* Initialize DCT and MelFilter arrays if not initialized */
    if (!coeffecients_initialized) {
        mfcc::getMelFilterBanks(&MelFilterArray, nfft, nfilt, down_sampled_fs);
        mfcc::calculateDCTCoefficients(&DCTArray, num_ceps, nfilt);
        coeffecients_initialized = true;
    }

    int buffer_size = 48000 * 3; // should be size of samples from recording

    /* create array that will hold the final output */
    float* final_output = nullptr;
    int final_output_size = performMFCC(buffer, &final_output, buffer_size, nn_data_cols, 0.6);

    // Copy the final_output to the outputArray
    env->SetFloatArrayRegion(outputArray, 0, final_output_size, final_output);
    delete [] final_output;

    return;
};

JNIEXPORT jintArray JNICALL
Java_mariannelinhares_mnistandroid_MainActivity_getRowAndCol(JNIEnv *env, jclass clazz) {
    jintArray result = env->NewIntArray(2);
    if (result == nullptr) {
        return nullptr; // Out of memory error
    }

    jint tempArray[] = {nn_data_rows, nn_data_cols};
    env->SetIntArrayRegion(result, 0, 2, tempArray);
    return result;
};
