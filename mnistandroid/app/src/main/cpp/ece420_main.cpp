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
            jobject bufferPtr, jfloatArray outputArray, jshortArray trimmed_audio, jintArray canvas);

JNIEXPORT jintArray JNICALL
    Java_mariannelinhares_mnistandroid_MainActivity_getRowAndCol(JNIEnv *env, jclass clazz);

JNIEXPORT jintArray JNICALL
    Java_mariannelinhares_mnistandroid_MainActivity_getMFCCParams(JNIEnv *env, jobject clazz);

JNIEXPORT void JNICALL
Java_mariannelinhares_mnistandroid_MainActivity_createWavFile(JNIEnv *env, jclass clazz,
                                                              jobject buffer_ptr, jint num_samples, jint sample_rate,
                                                              jbyteArray wav_file);
}

// Student Variables
#define VOICED_THRESHOLD 20000000
#define FRAME_SETBACK 2
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
const int nn_data_cols = 28; // number of frames to work with
const float preemphasis_b = 0.97;
//const int nn_data_cols = 1100;
const int nn_data_rows = 12; // is always num_ceps - 1
bool coeffecients_initialized = false;
float* MelFilterArray = nullptr;
float* DCTArray = nullptr;

/* Hanning Window */
float hanning_window[nfft] = {};
bool hanning_window_initialized = false;
float window_scaling_factor = 0;

/*
 * Description: Gets called periodically to store samples, creating a 3 second buffer with a
 *              sampling rate of 48 kHz
 * Inputs:
 *          dataBuf -- buffer holding a portion of samples
 * Outputs:
 *          None
 * Returns:
 *          None
 * Effects: Fills the threeSecondSamples global array and updates its global index. No difficult
 *          processing should take place here since it must return before the next set of samples
 *          are called.
 */
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

/*
 * Description: This function determines if a frame has been voiced based on the energy in the frame
 *              and a user defined energy threshold
 * Inputs:
 *          frame -- the array of samples in the frame
 *          num_samples -- the number of samples in the frame
 *          threshold -- the minimum threshold to meet for the frame to be considered voiced
 * Outputs:
 *          None
 * Returns:
 *          isVoiced -- 1 if the frame was voiced, 0 if not
 */
int processFrame(int16_t* frame, int num_samples, int threshold) {
    int isVoiced = 0;

    int sum = 0;
    for (int i = 0; i < num_samples; i++) {
        sum += pow(abs(frame[i]),2);
    }

    if (sum > threshold)
        isVoiced = 1;

    return isVoiced;
}

/*
 * Description: Using pitch detection, trim audio data to a desired frame size and surrounding a section
 *              of voiced samples
 * Inputs:
 *          samples -- the array of audio samples
 *          num_samples -- the number of samples in the given audio array
 *          frameSize -- the desired number of frames to trim the samples
 *          nfft -- the FFT size that will be used on the samples later in computation
 *          noverlap -- the amount of overlap between frames
 *          threshold -- the minimum threshold to meet for the frame to be considered voiced
 *          frame_setback -- the number of frames to start before the first voiced frame is found
 *  Outputs:
 *          trimmed_samples -- the array to be allocated and hold the trimmed samples
 *  Returns:
 *          trimmedSize -- the number of elements given to trimmed_samples
 *  Effects:
 *          Allocates memory for trimmed_samples that must be freed later to prevent memory leaks
 */
int trim_samples(float* samples, float** trimmed_samples, int num_samples, int frameSize, int nfft, int noverlap,  int threshold, int frame_setback) {

    /* calculate the number of frames and step size */
    if (noverlap < 0)
        noverlap = nfft / 2;

    int step = nfft - noverlap;

    int numFrames = ceil(num_samples / step);

    /* indicates if a frame was voiced or not */
    while ((numFrames - 1)*step + (nfft - 1) >= num_samples)
        numFrames--;

    /* find the first frame that is voiced */
    int16_t* frame = new int16_t[nfft];
    int first_frame = 0;
    int voiced;
    for (int i = 0; i < numFrames; i++) {
        mfcc::floatToInt16(&samples[i*step], frame, nfft);
        voiced = processFrame(frame, nfft, threshold);
        if (voiced > 0) {
            first_frame = i;
            break;
        }
    }

    delete [] frame;

    /* include the setback and make sure it is in bounds */
    first_frame -= frame_setback;
    first_frame = first_frame < 0 ? 0 : first_frame; // make sure to set the first_frame in bounds

    int last_frame = first_frame + frameSize; // exclusive
    int trimmedSize = (((last_frame-1) * step)+nfft) - (first_frame * step);
    float* trimmed_samples_ = new float[trimmedSize];

    int t_idx = 0; // idx in trimmed_samples_
    for (int s_idx = first_frame*step; s_idx < ((last_frame-1)*step)+nfft; s_idx++) {
        /* check if the index into the original samples array is out of bounds */
        if (s_idx >= num_samples)
            break;

        trimmed_samples_[t_idx] = samples[s_idx];
        t_idx++;
    }
    /* simpler method to store samples to test out later */
//    int samples_idx;
//    for (int i = 0; i < trimmedSize; i++) {
//        samples_idx = first_frame * step + i;
//        if (samples_idx >= num_samples)
//            break;
//        trimmed_samples_[i] = samples[first_frame * step + i];
//    }


    /* redirect user pointer and return size of trimmed array */
    *trimmed_samples = trimmed_samples_;
    return trimmedSize;
}

/*
 * Description: Applies the Hanning window to a frame of audio samples
 * Inputs:
 *          frame -- the array of audio samples
 * Outputs:
 *          None
 * Returns:
 *          None
 * Effects: Allocates the global hanning_window array as well as its window_scaling_factor in case
 *          it needs to be used for some later computation. This scaled factor thus far is not needed
 *          in the scope of the project.
 */
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

/*
 * Description: Takes an array of samples and conducts the STFT on it given a set of parameters
 * Inputs:
 *          samples -- the array of samples
 *          num_samples -- the number of samples given
 *          sample_rate -- the sampling rate of the samples
 *          oneSided -- if the real or FFT should be returned in each frame
 * Outputs:
 *          frequencies -- the fully computed STFT
 * Returns:
 *          stft_output_size -- the number of samples in the final output
 * Effects: Allocates memory for frequencies that must be freed later to prevent a memory leak.
 */
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
            curr_samples[i] = samples[n * step + i];
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
        float curr_val;
        if (oneSided) {
            for (int i = 0; i < int(nfft/2 + 1); i++) {
                curr_val = fabs((out[i].r * out[i].r) + (out[i].i * out[i].i));
                assert(curr_val >= 0);
                frequencies_[i * num_ffts + n] = curr_val;
            }
        }
        else {
            for (int i = 0; i < nfft; i++) {
                curr_val = fabs((out[i].r * out[i].r) + (out[i].i * out[i].i));
                assert(curr_val >= 0);
                frequencies_[i * num_ffts + n] = curr_val;
            }
        }
    }
    kiss_fft_free(cfg);

    *frequencies = frequencies_;

    return stft_output_size;
}

/*
 * Description: Performs the Mel Frequency Cepstral Coefficient algorithm on a set of samples.
 * Inputs:
 *          samples -- the array of audio samples
 *          num_samples -- the number of samples given
 *          preemphasis_b -- the coefficient to be used in the preemphasis step
 * Outputs:
 *          mfcc_frequencies -- the final mfcc output
 *          ret_rows -- the number of rows in the final output
 *          ret_cols -- the number of columns in the final output
 * Returns:
 *          ceps_output_size -- the number of elements in the final output
 */
int performMFCC(float* samples, float** mfcc_frequencies, int num_samples, float preemphasis_b, int &ret_rows, int &ret_cols) {

    /* apply preemphasis filter */
    mfcc::preemphasis(samples, num_samples, preemphasis_b);

    /* apply stft */
    float* stft_output = nullptr;
    int stft_output_size = performSTFT(samples, &stft_output, num_samples, down_sampled_fs, true);
    int stft_num_frames = stft_output_size / int(nfft / 2 + 1);

    /* apply the mel filter banks */
    int mel_filtered_output_size = nfilt * stft_num_frames;
    float* mel_filtered_output = new float[mel_filtered_output_size];
    mfcc::gemmMultiplication(MelFilterArray, stft_output, mel_filtered_output, nfilt, int(nfft / 2 + 1), stft_num_frames);

    delete [] stft_output;

    /* take log base 10 of output before applying DCT */
    for (int i = 0; i < mel_filtered_output_size; i++) {
        float temp = mel_filtered_output[i];
        temp = (temp == 0.0) ? std::numeric_limits<float>::epsilon() : temp;
        mel_filtered_output[i] = log10(temp);
    }

    /* perform DCT */
    int ceps_output_size = num_ceps * stft_num_frames;
    float* ceps_output = new float [ceps_output_size];

    mfcc::gemmMultiplication(DCTArray, mel_filtered_output, ceps_output, num_ceps, nfilt, stft_num_frames);

    /* no longer need mel_filterd_output array */
    delete [] mel_filtered_output;

    ret_rows = num_ceps;
    ret_cols = stft_num_frames;

    *mfcc_frequencies = ceps_output;
    return ceps_output_size;
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
    for (int i = 0; i < threeSecondSampleSize; i++) {
        threeSecondSamples[i] = 0;
    }
}

JNIEXPORT void JNICALL
Java_mariannelinhares_mnistandroid_MainActivity_performMFCC(JNIEnv *env, jclass clazz, jobject bufferPtr, jfloatArray outputArray, jshortArray trimmed_audio, jintArray canvas) {
    // TODO: implement performMFCC()

    int16_t * buffer = (jshort *) env->GetDirectBufferAddress(bufferPtr);

    /* Initialize DCT and MelFilter arrays if not initialized */
    if (!coeffecients_initialized) {
        mfcc::getMelFilterBanks(&MelFilterArray, nfft, nfilt, down_sampled_fs);
        mfcc::calculateDCTCoefficients(&DCTArray, num_ceps, nfilt);
        coeffecients_initialized = true;
    }

    int buffer_size = 48000 * 3; // should be size of samples from recording

    /* filter and downsample */
    float* f_samples = new float[buffer_size];

    mfcc::int16ToFloat(buffer, f_samples, buffer_size);

    /* apply FIR filter */
    mfcc::applyFirFilterSeries(f_samples, buffer_size);

    /* downsample */
    float* down_sampled_sig;
    int down_sampled_sig_size = mfcc::downsample(f_samples, &down_sampled_sig, buffer_size, fs, down_sampled_fs);

    delete [] f_samples; // no longer needed

    /* calling trimming algorithm that is based on pitch detection */
    float* trimmed_samples = nullptr;
    int trimmed_samples_size = trim_samples(down_sampled_sig, &trimmed_samples, down_sampled_sig_size, nn_data_cols, nfft, noverlap, VOICED_THRESHOLD, FRAME_SETBACK);

    /* copy back the trimmed audio for debugging purposes */
    int16_t* trimmed_samples_int16 = new int16_t[trimmed_samples_size];
    mfcc::floatToInt16(trimmed_samples, trimmed_samples_int16, trimmed_samples_size);
    env->SetShortArrayRegion(trimmed_audio, 0, trimmed_samples_size, trimmed_samples_int16);
    delete [] trimmed_samples_int16;

    /* create array that will hold the final output */
    float* mfcc_output = nullptr;
    int mfcc_rows;
    int mfcc_cols;
    performMFCC(trimmed_samples, &mfcc_output, trimmed_samples_size, preemphasis_b, mfcc_rows, mfcc_cols);

    delete [] trimmed_samples; // array is no longer needed

    int final_output_size = nn_data_cols*nn_data_rows;
    float* final_output = new float[final_output_size];

    /* return the computed mfcc output except the first row */
    for (int row = 0; row < nn_data_rows; row++) {
        for (int col = 0; col < nn_data_cols; col++) {
            final_output[row * nn_data_cols + col] = mfcc_output[(row+1) * mfcc_cols + col];
        }
    }

    delete [] mfcc_output;

    /* draw the mfcc to the canvas */
    uint32_t* image = new uint32_t[400*300];
    mfcc::createImage(final_output, image, 400, 300, nn_data_rows, nn_data_cols);

    env->SetIntArrayRegion(canvas, 0, 400*300, (int*)image);
    delete [] image;

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
    /* DEBUG, temporarily returning mel filter size */
    jint tempArray[] = {nn_data_rows, nn_data_cols};
//    jint tempArray[] = {nfilt, int(nfft/2+1)};
    env->SetIntArrayRegion(result, 0, 2, tempArray);
    return result;
};

JNIEXPORT jintArray JNICALL
Java_mariannelinhares_mnistandroid_MainActivity_getMFCCParams(JNIEnv *env, jobject clazz) {
    // TODO: implement getMFCCParams()
    /*
     * Params:
     * mfcc_rows
     * mfcc_cols
     * nfft
     * nfilt
     * noverlap
     * num_ceps
     * downsampled_fs
     * step
     * trimmed size
     */
    jintArray result = env->NewIntArray(9);
    if (result == nullptr)
        return nullptr; // out of memory error

    /* calculating the step size */
    int step = nfft - noverlap;

    /* calculating the size needed for an array holding the trimmed samples */
    int trimmed_size = ((nn_data_cols-1) * step)+nfft;

    jint tempArray[] = {nn_data_rows, nn_data_cols, nfft, nfilt, noverlap, num_ceps, down_sampled_fs, step, trimmed_size};
    env->SetIntArrayRegion(result, 0, 9, tempArray);
    return result;
}

JNIEXPORT void JNICALL
Java_mariannelinhares_mnistandroid_MainActivity_createWavFile(JNIEnv *env, jclass clazz,
                                                              jobject buffer_ptr, jint num_samples, jint sample_rate,
                                                              jbyteArray wav_file) {
    // TODO: implement createWavFile()
    /* grabbing pointer to audio samples */
    int16_t * buffer = (jshort *) env->GetDirectBufferAddress(buffer_ptr);

    uint8_t* wav_file_ = nullptr;
    int wav_file_size = mfcc::createWav(buffer, &wav_file_, num_samples, sample_rate);

    /* casting wav_file_ to signed byte but should not make a difference */
    env->SetByteArrayRegion(wav_file, 0, wav_file_size, (int8_t *)wav_file_);

    delete [] wav_file_;

    return;
}