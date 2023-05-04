package mariannelinhares.mnistandroid;
//package com.ece420.lab5; // package for getting lab 5 functions

/*
   Copyright 2016 Narrative Nights Inc. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   From: https://raw.githubusercontent
   .com/miyosuda/TensorFlowAndroidMNIST/master/app/src/main/java/jp/narr/tensorflowmnist
   /DrawModel.java
*/

//An activity is a single, focused thing that the user can do. Almost all activities interact with the user,
//so the Activity class takes care of creating a window for you in which you can place your UI with setContentView(View)
import android.Manifest;
import android.app.Activity;
//PointF holds two float coordinates
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
//A mapping from String keys to various Parcelable values (interface for data container values, parcels)
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.os.Bundle;
//Object used to report movement (mouse, pen, finger, trackball) events.
// //Motion events may hold either absolute or relative movements and other data, depending on the type of device.
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
//This class represents the basic building block for user interface components.
// A View occupies a rectangular area on the screen and is responsible for drawing
import android.view.View;
//A user interface element the user can tap or click to perform an action.
import android.view.WindowManager;
import android.widget.Button;
//A user interface element that displays text to the user. To provide user-editable text, see EditText.
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
//Resizable-array implementation of the List interface. Implements all optional list operations, and permits all elements,
// including null. In addition to implementing the List interface, this class provides methods to
// //manipulate the size of the array that is used internally to store the list.
import java.io.FileDescriptor;
import java.io.OutputStream;
import java.nio.IntBuffer;
import java.nio.MappedByteBuffer;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
// basic list
import java.util.List;
//encapsulates a classified image
//public interface to the classification class, exposing a name and the recognize function
import mariannelinhares.mnistandroid.models.Classifier;
//contains logic for reading labels, creating classifier, and classifying
import mariannelinhares.mnistandroid.models.TensorFlowClassifier;
//class for drawing MNIST digits by finger
//class for drawing the entire app
// for creating directories and files from java
import java.io.File;
import android.os.Environment;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Timer;
import android.os.CountDownTimer;
import java.util.Calendar;
import java.util.Date;

/* tensor flow lite requirements */
import org.tensorflow.lite.Interpreter;

/* added OnRequestPerm... which i believe opens dialogue to ask for speaker permission */
    public class MainActivity extends Activity implements ActivityCompat.OnRequestPermissionsResultCallback {

    private static final int PIXEL_WIDTH = 28;

    /**********************************************************************************************/
    /**                                 LAB 5 SETUP                                              **/
    // UI Variables from Lab 5
    Button   controlButton;
    TextView statusView;
    String  nativeSampleRate;
    String  nativeSampleBufSize;
    boolean supportRecording;
    Boolean isPlaying = false;

    // Static Values from Lab 5
    private static final int AUDIO_ECHO_REQUEST = 0;
    private static final int EXTERNAL_STORAGE_REQUEST = 0;
    private static final int FRAME_SIZE = 1024;

    private short[] audio_samples;
    private Timer timer;

    /* wav audio private variables */
    private String last_raw_audio = "";
    private String last_trimmed_audio = "";
    private byte[] raw_wav_file;
    private byte[] trimmed_wav_file;

    /**********************************************************************************************/


    // ui elements
    private Button trimmedBtn, replayBtn;
    private TextView resText;
    private TextView tfliteResultText;
    private List<Classifier> mClassifiers = new ArrayList<>();

    // views
    ImageView mfccView;
    Bitmap bitmap;
    Canvas canvas;

    private static final String RAW_AUDIO = "/sdcard/data/raw_samples/";
    private static final String PROCESSED_AUDIO = "/sdcard/data/processed_samples/";
    private static final String TRIMMED_AUDIO = "/sdcard/data/trimmed_samples/";
    private static final String MFCC_IMAGES = "/sdcard/data/mfcc_images/";
    private static final String RAW_WAV_AUDIO = "/sdcard/data/raw_wav_audio/";
    private static final String TRIMMED_WAV_AUDIO = "/sdcard/data/trimmed_wav_audio/";


    /* tensorflow lite private variables */
    private Interpreter tflite = null;

    @Override
    // In the onCreate() method, you perform basic application startup logic that should happen
    //only once for the entire life of the activity.
    protected void onCreate(Bundle savedInstanceState) {
        //initialization
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

    /**********************************************************************************************/
    /**                                 LAB 5 SETUP                                              **/
        // Google NDK Stuff
        controlButton = (Button)findViewById((R.id.capture_control_button));
        statusView = (TextView)findViewById(R.id.statusView);
        queryNativeAudioParameters();
        // initialize native audio system
        updateNativeAudioUI();
        if (supportRecording) {
            // Native Setting: 48k Hz Sampling Frequency and 128 Frame Size
            createSLEngine(Integer.parseInt(nativeSampleRate), FRAME_SIZE);
        }

        /* should be able to hold 3 seconds worth of data */
        audio_samples = new short[48000 * 3];
        timer = new Timer();

        /**********************************************************************************************/

        /* create directories if they did not exist before */
        File directory = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + RAW_AUDIO);
        directory.mkdirs();
        directory = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + PROCESSED_AUDIO);
        directory.mkdirs();
        directory = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + TRIMMED_AUDIO);
        directory.mkdirs();
        directory = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + MFCC_IMAGES);
        directory.mkdirs();
        directory = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + RAW_WAV_AUDIO);
        directory.mkdirs();
        directory = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + TRIMMED_WAV_AUDIO);
        directory.mkdirs();

        /* setting up mfcc imageview */
        mfccView = (ImageView) this.findViewById(R.id.mfccView);
        bitmap =  Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888);
        canvas = new Canvas(bitmap);
        canvas.drawColor(Color.BLACK);
        mfccView.setImageBitmap(bitmap);

        //clear button
        //clear the drawing when the user taps
        trimmedBtn = (Button) findViewById(R.id.btn_trimmed);
        trimmedBtn.setEnabled(false);

        //class button
        //when tapped, this performs classification on the drawn image
        replayBtn = (Button) findViewById(R.id.btn_replay);
        replayBtn.setEnabled(false);

        // res text
        //this is the text that shows the output of the classification
        resText = (TextView) findViewById(R.id.tfRes);
        tfliteResultText = (TextView) findViewById(R.id.tfliteResults);

        //load up our saved model to perform inference from local storage
        loadModel();
        loadTFLiteModel();
    }

    //the activity lifecycle

    @Override
    //OnResume() is called when the user resumes his Activity which he left a while ago,
    // //say he presses home button and then comes back to app, onResume() is called.
    protected void onResume() {
//        drawView.onResume();
        super.onResume();
    }

    @Override
    //OnPause() is called when the user receives an event like a call or a text message,
    // //when onPause() is called the Activity may be partially or completely hidden.
    protected void onPause() {
//        drawView.onPause();
        super.onPause();
    }
    //creates a model object in memory using the saved tensorflow protobuf model file
    //which contains all the learned weights
    private void loadModel() {
        //The Runnable interface is another way in which you can implement multi-threading other than extending the
        // //Thread class due to the fact that Java allows you to extend only one class. Runnable is just an interface,
        // //which provides the method run.
        // //Threads are implementations and use Runnable to call the method run().
        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    //add 2 classifiers to our classifier arraylist
                    //the tensorflow classifier and the keras classifier
                    mClassifiers.add(
                            TensorFlowClassifier.create(getAssets(), "TensorFlow",
                                    "opt_mnist_convnet-tf.pb", "labels.txt", PIXEL_WIDTH,
                                    PIXEL_WIDTH,
                                    "input", "output", true));
                    mClassifiers.add(
                            TensorFlowClassifier.create(getAssets(), "Keras",
                                    "opt_mnist_convnet-keras.pb", "labels.txt", PIXEL_WIDTH,
                                    PIXEL_WIDTH,
                                    "conv2d_1_input", "dense_2/Softmax", false));
                } catch (final Exception e) {
                    //if they aren't found, throw an error!
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
    }

    /**********************************************************************************************/
    /**                                 LAB 5 METHODS                                            **/

    @Override
    protected void onDestroy() {
        if (supportRecording) {
            if (isPlaying) {
                stopPlay();
            }
            deleteSLEngine();
            isPlaying = false;
        }
        super.onDestroy();
    }

    private void startEcho() {
        if(!supportRecording){
            return;
        }
        if (!isPlaying) {
            if(!createSLBufferQueueAudioPlayer()) {
                statusView.setText(getString(R.string.error_player));
                return;
            }
            if(!createAudioRecorder()) {
                deleteSLBufferQueueAudioPlayer();
                statusView.setText(getString(R.string.error_recorder));
                return;
            }
            startPlay();   // this must include startRecording()
            statusView.setText(getString(R.string.status_echoing));

            /* count down timer implementation */
            new CountDownTimer(3000, 1000) {

                public void onTick(long millisUntilFinished) {
                    // logic to set the EditText could go here
                    // do nothing
                    controlButton.setText("" + (1 + (millisUntilFinished / 1000 )));
                }

                public void onFinish() {
                    isPlaying = false;
                    controlButton.setText(getString(R.string.StartEcho));

                    stopPlay();  //this must include stopRecording()
                    updateNativeAudioUI();
                    deleteAudioRecorder();
                    deleteSLBufferQueueAudioPlayer();

                    /* short is 2 bytes, 48000*3 total samples for 3 seconds of audio */
                    ShortBuffer buffer = ByteBuffer.allocateDirect(48000 * 3 * 2)
                            .order(ByteOrder.LITTLE_ENDIAN)
                            .asShortBuffer();

                    /* get the three seconds worth of data from the C++ end */
                    getCompleteSamplesBuffer(buffer);

                    /* save the samples to the global float buffer */
                    for (int i = 0; i < 48000 * 3; i++)
                        audio_samples[i] = buffer.get();
                    buffer.rewind();

                    Date currentTime = Calendar.getInstance().getTime();
                    File rootPath = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + RAW_AUDIO, currentTime + "_rawsamples.csv");
                    writeSamplesToCSVFromShortArray(rootPath, audio_samples);

                    /* reset parameters for the 3 second buffer in the C++ end */
                    resetParameters();

                    buffer.rewind();
                    /* get size to allocate for new buffer and perform mfcc */
                    int [] mfcc_dim = getRowAndCol();
                    int [] mfcc_params = getMFCCParams();
                    int mfcc_rows = mfcc_dim[0];
                    int mfcc_cols = mfcc_dim[1];
                    int mfcc_output_size = mfcc_cols * mfcc_rows;
                    float [] mfcc_output = new float[mfcc_output_size];
                    int [] mfcc_canvas = new int[400*300];

                    /* find out size of trimmed output */
                    /*
                     * Params:
                     * 0 -- mfcc_rows
                     * 1 -- mfcc_cols
                     * 2 -- nfft
                     * 3 -- nfilt
                     * 4 -- noverlap
                     * 5 -- num_ceps
                     * 6 -- downsampled_fs
                     * 7 -- step
                     * 8 -- trimmed size
                     */
                    int nfft = mfcc_params[2];
                    int noverlap = mfcc_params[4];
                    int downsampled_fs = mfcc_params[6];
                    if (noverlap < 0)
                        noverlap = nfft / 2;
                    int trimmed_size = mfcc_params[8];
                    short [] trimmed_output = new short[trimmed_size];

                    performMFCC(buffer, mfcc_output, trimmed_output, mfcc_canvas);

                    /* save the raw audio and trimmed audio as wav files */
                    buffer.rewind();
                    int raw_wav_file_bytes = 44 + (48000*3*2);
                    raw_wav_file = new byte[raw_wav_file_bytes];
                    createWavFile(buffer, buffer.capacity(), 48000, raw_wav_file);

                    /* write raw wav file */
                    FileOutputStream fos = null;
                    try {
                        last_raw_audio = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + RAW_WAV_AUDIO + currentTime + "_raw_audio.wav";
                        fos = new FileOutputStream(last_raw_audio);
                        fos.write(raw_wav_file, 0, raw_wav_file.length);
                        fos.close();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    try {
                        if(fos != null)
                            fos.close();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    /* end write raw wav file */

                    /* now that we have at least one wav file saved, we can enable the replay button */
                    replayBtn.setEnabled(true);

                    /* now save the trimmed audio */
                    int trimmed_wav_file_bytes = 44 + trimmed_output.length * 2;
                    trimmed_wav_file = new byte[trimmed_wav_file_bytes];
                    ShortBuffer trimmed_output_buffer = ByteBuffer.allocateDirect(trimmed_output.length * 2)
                            .order(ByteOrder.LITTLE_ENDIAN)
                            .asShortBuffer();
                    for (int i = 0; i < trimmed_output.length; i++)
                        trimmed_output_buffer.put(trimmed_output[i]);
                    trimmed_output_buffer.rewind();
                    createWavFile(trimmed_output_buffer, trimmed_output.length, downsampled_fs, trimmed_wav_file);

                    /* write trimmed wav file */
                    fos = null;
                    try {
                        last_trimmed_audio = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + TRIMMED_WAV_AUDIO + currentTime + "_trimmed_audio.wav";
                        fos = new FileOutputStream(last_trimmed_audio);
                        fos.write(trimmed_wav_file, 0, trimmed_wav_file.length);
                        fos.close();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    try {
                        if(fos != null)
                            fos.close();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    /* end write trimmed wav file */
                    /* now that we have at least one trimmed wav file saved, we can enable the replay button */
                    trimmedBtn.setEnabled(true);

                    /* save results to csv files */
                    rootPath = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + TRIMMED_AUDIO, currentTime + "_trimmedsamples.csv");
                    writeSamplesToCSVFromShortArray(rootPath, trimmed_output);

                    rootPath = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + PROCESSED_AUDIO, currentTime + "_processed.csv");
                    writeSamplesToCSVFromFloatArray(rootPath, mfcc_output, mfcc_output_size);

                    /* draw to the canvas */
                    bitmap = Bitmap.createBitmap(400, 300, Bitmap.Config.ARGB_8888);
                    bitmap.copyPixelsFromBuffer(makeBuffer(mfcc_canvas, mfcc_canvas.length));
                    mfccView.setImageBitmap(bitmap);

                    OutputStream stream = null;
                    try {
                        stream = new FileOutputStream(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + MFCC_IMAGES + currentTime + "_mfcc.png");
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                    /* Write bitmap to file using JPEG or PNG and 80% quality hint for JPEG. */
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
                    try {
                        stream.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    // Prepare input data (mfcc images)
                    byte[][][][] inputArray = new byte[1][300][400][3];

                    float[][] outputArray = new float[1][10]; // Assuming you have a classification problem with NUM_CLASSES classes

                    /*
                        For images, the data must be loaded differently.
                        There are actually 4 channels but the fourth channel, alpha, is always
                        0xFF, thus incorporating wastes memory and computation time.
                    */
                    int mask = 0x000000FF;
                    for (int row = 0; row < 300; row++) {
                        for (int col = 0; col < 400; col++) {
                            for (int channel = 0; channel < 3; channel++) {
                                inputArray[0][row][col][channel] = (byte)(mask & (mfcc_canvas[row * 400 + col] >> (channel*8)));
                            }
                        }
                    }

                    tflite.run(inputArray, outputArray);

                    /* flatten output probabilities */
                    float [] flat_outputArray = outputArray[0];
                    /* sort from least to greatest */
                    float [] sorted_outputArray = mergeSort(flat_outputArray);

                    /* get the top 3 probabilities */
                    int first_label = 0;
                    int second_label = 0;
                    int third_label = 0;
                    for (int i = 0; i < 10; i++) {
                        if (flat_outputArray[i] == sorted_outputArray[9]) {
                            first_label = i;
                        }
                        if (flat_outputArray[i] == sorted_outputArray[8]) {
                            second_label = i;
                        }
                        if (flat_outputArray[i] == sorted_outputArray[7]) {
                            third_label = i;
                        }
                    }

                    /* display the resultson the UI */
                    String tfres = String.format("1) Predicted Label: %d, Prob: %f\n" +
                            "2) Predicted Label: %d, Prob: %f\n" +
                            "3) Predicted Label: %d, Prob: %f\n", first_label, flat_outputArray[first_label],
                            second_label, flat_outputArray[second_label], third_label, flat_outputArray[third_label]);
                    tfliteResultText.setText(tfres);
                }

            }.start();
        } else {
            stopPlay();  //this must include stopRecording()
            updateNativeAudioUI();
            deleteAudioRecorder();
            deleteSLBufferQueueAudioPlayer();

            /* reset parameters for the 3 second buffer in the C++ end */
            resetParameters();
        }
        isPlaying = !isPlaying;
        controlButton.setText(getString((isPlaying == true) ?
                R.string.StopEcho: R.string.StartEcho));
    }

    public void playTrimmedAudio(View view) {

        MediaPlayer player;
        try {
            /* disable audio buttons */
            replayBtn.setEnabled(false);
            trimmedBtn.setEnabled(false);
            FileInputStream fis = new FileInputStream(last_trimmed_audio);
            FileDescriptor afd = fis.getFD();
            player = new MediaPlayer();
            player.setDataSource(afd);
            player.prepare();
            player.start();
            player.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {

                @Override
                public void onCompletion(MediaPlayer mp) {
                    // TODO Auto-generated method stub
                    mp.release();
                    /* reenable audio buttons */
                    replayBtn.setEnabled(true);
                    trimmedBtn.setEnabled(true);
                }

            });
            player.setLooping(false);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void replayLastAudio(View view) {

        MediaPlayer player;
        try {
            /* disable audio buttons */
            replayBtn.setEnabled(false);
            trimmedBtn.setEnabled(false);
            FileInputStream fis = new FileInputStream(last_raw_audio);
            FileDescriptor afd = fis.getFD();
            player = new MediaPlayer();
            player.setDataSource(afd);
            player.prepare();
            player.start();
            player.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {

                @Override
                public void onCompletion(MediaPlayer mp) {
                    // TODO Auto-generated method stub
                    mp.release();
                    /* reenable audio buttons */
                    replayBtn.setEnabled(true);
                    trimmedBtn.setEnabled(true);
                }

            });
            player.setLooping(false);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /*
     * Not a good method for capturing clean audio data, but a good example of how to run a
     * periodic thread
     */
//    private class UpdateStftTask extends AsyncTask<Void, FloatBuffer, Void> {
//        @Override
//        protected Void doInBackground(Void... params) {
//
//            // Float == 4 bytes
//            // Note: We're using FloatBuffer instead of float array because interfacing with JNI
//            // with a FloatBuffer allows direct memory sharing, versus having to copy to some
//            // intermediate location first.
//            // http://stackoverflow.com/questions/10697161/why-floatbuffer-instead-of-float
//
//            /* multiply by 2 because for 50% overlap, we will now publish two FFT's per call */
//            FloatBuffer buffer = ByteBuffer.allocateDirect(FRAME_SIZE * 4)
//                    .order(ByteOrder.LITTLE_ENDIAN)
//                    .asFloatBuffer();
//
//            getSamplesBuffer(buffer);
//
//            // Update screen, needs to be done on UI thread
//            publishProgress(buffer);
//
//            return null;
//        }
//
//        protected void onProgressUpdate(FloatBuffer... newBufferSamples) {
//
//            int newBufferCapacity = newBufferSamples[0].capacity();
//            for (int i = 0; i < newBufferCapacity; i++) {
//                if (audio_samples_idx + i >= audio_samples.length)
//                    break;
//                audio_samples[audio_samples_idx + i] = newBufferSamples[0].get();
//            }
//            audio_samples_idx += newBufferCapacity;
//            recording_duration += 10; // this should presumably take 10 ms
//
//            /* check if we are past 3 seconds of recording */
//            if (recording_duration >= 30000) {
//                /* make sure asynnchronous task is stopped */
//                timer.cancel();
//                recording_duration = 0; // reset back to 0 ms
//                audio_samples_idx = 0;
//
//                /* write 3 second array of samples to a csv file */
//                File rootPath = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + DNAME, "test_samples.csv");
//                writeSamplesToCSV(rootPath);
//
//            }
//
//            newBufferSamples[0].rewind();
//        }
//    }

    public void onEchoClick(View view) {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) !=
                PackageManager.PERMISSION_GRANTED) {
            statusView.setText(getString(R.string.status_record_perm));
            ActivityCompat.requestPermissions(
                    this,
                    new String[] { Manifest.permission.RECORD_AUDIO },
                    AUDIO_ECHO_REQUEST);
        }
        startEcho();

        /* checking permissions to write to SD card and creating directory */
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) !=
                PackageManager.PERMISSION_GRANTED) {
            statusView.setText(getString(R.string.status_record_perm));
            ActivityCompat.requestPermissions(
                    this,
                    new String[] { Manifest.permission.WRITE_EXTERNAL_STORAGE },
                    EXTERNAL_STORAGE_REQUEST);
        }

        return;

    }

    public void getLowLatencyParameters(View view) {
        updateNativeAudioUI();
        return;
    }

    private void queryNativeAudioParameters() {
        AudioManager myAudioMgr = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
        nativeSampleRate  =  myAudioMgr.getProperty(AudioManager.PROPERTY_OUTPUT_SAMPLE_RATE);
        nativeSampleBufSize =myAudioMgr.getProperty(AudioManager.PROPERTY_OUTPUT_FRAMES_PER_BUFFER);
        int recBufSize = AudioRecord.getMinBufferSize(
                Integer.parseInt(nativeSampleRate),
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT);
        supportRecording = true;
        if (recBufSize == AudioRecord.ERROR ||
                recBufSize == AudioRecord.ERROR_BAD_VALUE) {
            supportRecording = false;
        }
    }

    private void updateNativeAudioUI() {
        if (!supportRecording) {
            statusView.setText(getString(R.string.error_no_mic));
            controlButton.setEnabled(false);
            return;
        }

        statusView.setText("nativeSampleRate    = " + nativeSampleRate + "\n" +
                "nativeSampleBufSize = " + nativeSampleBufSize + "\n");

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        /*
         * if any permission failed, the sample could not play
         */
        if (AUDIO_ECHO_REQUEST != requestCode) {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            return;
        }

        if (EXTERNAL_STORAGE_REQUEST != requestCode) {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            return;
        }

        if (grantResults.length != 1  ||
                grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            /*
             * When user denied permission, throw a Toast to prompt that RECORD_AUDIO
             * is necessary; also display the status on UI
             * Then application goes back to the original state: it behaves as if the button
             * was not clicked. The assumption is that user will re-click the "start" button
             * (to retry), or shutdown the app in normal way.
             */
            statusView.setText(getString(R.string.error_no_permission));
            Toast.makeText(getApplicationContext(),
                    getString(R.string.prompt_permission),
                    Toast.LENGTH_SHORT).show();
            return;
        }

        /*
         * When permissions are granted, we prompt the user the status. User would
         * re-try the "start" button to perform the normal operation. This saves us the extra
         * logic in code for async processing of the button listener.
         */
        statusView.setText("RECORD_AUDIO permission granted, touch " +
                getString(R.string.StartEcho) + " to begin");

        // The callback runs on app's thread, so we are safe to resume the action
        startEcho();
    }

    private void writeTextData(File file, String data) {
        FileOutputStream fileOutputStream = null;
        try {
            fileOutputStream = new FileOutputStream(file);
            fileOutputStream.write(data.getBytes());
            Toast.makeText(this, "Done writing to " + file.getAbsolutePath(), Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (fileOutputStream != null) {
                try {
                    fileOutputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void writeSamplesToCSVFromBuf(File file, ShortBuffer buff) {
        FileOutputStream fileOutputStream = null;
        try {
            fileOutputStream = new FileOutputStream(file);
            for (int i = 0; i < 48000 * 3; i++) {
                fileOutputStream.write(String.format("%d\n", buff.get()).getBytes());
            }
            Toast.makeText(this, "Done writing to " + file.getAbsolutePath(), Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (fileOutputStream != null) {
                try {
                    fileOutputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void writeSamplesToCSVFromFloatArray(File file, float[] buff, int buff_size) {
        FileOutputStream fileOutputStream = null;
        try {
            fileOutputStream = new FileOutputStream(file);
            for (int i = 0; i < buff_size; i++) {
                fileOutputStream.write(String.format("%f\n", buff[i]).getBytes());
            }
            Toast.makeText(this, "Done writing to " + file.getAbsolutePath(), Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (fileOutputStream != null) {
                try {
                    fileOutputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private void writeSamplesToCSVFromShortArray(File file, short[] buff) {
        FileOutputStream fileOutputStream = null;
        try {
            fileOutputStream = new FileOutputStream(file);
            for (int i = 0; i < 48000 * 3; i++) {
                fileOutputStream.write(String.format("%d\n", buff[i]).getBytes());
            }
            Toast.makeText(this, "Done writing to " + file.getAbsolutePath(), Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (fileOutputStream != null) {
                try {
                    fileOutputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    /* basic merge sort */
    private float [] copyArr(float [] source, int startIdx, int endIdx) {
        /* non-inclusive endIdx */

        float [] copied_arr = new float[endIdx - startIdx];
        int copied_arr_idx = 0;

        for (int i = startIdx; i < endIdx; i++) {
            copied_arr[copied_arr_idx] = source[i];
            copied_arr_idx++;
        }

        return copied_arr;
    }

    private float [] mergeSort(float [] values) {
        int N = values.length;

        // base cases

        if (N < 2) {
            float[] retArr = new float[values.length];

            if (values.length == 1) {
                retArr[0] = values[0];
                return retArr;
            }
            else if (values[0] < values [1]) {
                retArr[0] = values[0];
                retArr[1] = values[1];
                return retArr;
            }
            else {
                retArr[0] = values[1];
                retArr[1] = values[0];
                return retArr;
            }
        }

        // recursive case

        int halfLength = N/2;

        float [] leftArr = mergeSort(copyArr(values, 0, halfLength));
        float [] rightArr = mergeSort(copyArr(values, halfLength, N));

        int leftN = leftArr.length;
        int rightN = rightArr.length;

        int leftIdx = 0;
        int rightIdx = 0;

        float [] mergedArr = new float[N];

        for (int i = 0; i < N; i++) {
            // check if indices are valid

            if (leftIdx >= leftN) {
                mergedArr[i] = rightArr[rightIdx];
                rightIdx++;
            }
            else if (rightIdx >= rightN) {
                mergedArr[i] = leftArr[leftIdx];
                leftIdx++;
            }
            else {
                if (leftArr[leftIdx] < rightArr[rightIdx]) {
                    mergedArr[i] = leftArr[leftIdx];
                    leftIdx++;
                }
                else {
                    mergedArr[i] = rightArr[rightIdx];
                    rightIdx++;
                }
            }
        }

        return mergedArr;
    }

    /* tensor flow lite functions */
    private void loadTFLiteModel() {

        new Thread(new Runnable() {
            @Override
            public void run() {
                try {
                    // Load the TFLite model file
                    /* this loads in the data set for Jorge */
//                    AssetFileDescriptor fileDescriptor = getAssets().openFd("my_model_image_300_400_mfcc_12_28.tflite");
                    /* this loads in the data set for Jonathan */
                    AssetFileDescriptor fileDescriptor = getAssets().openFd("jonathan_model_image_300_400_mfcc_12_28.tflite");
                    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
                    FileChannel fileChannel = inputStream.getChannel();
                    long startOffset = fileDescriptor.getStartOffset();
                    long declaredLength = fileDescriptor.getDeclaredLength();
                    MappedByteBuffer modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

                    // Initialize the TFLite interpreter
                    tflite = new Interpreter(modelBuffer);

                } catch (IOException e) {
                    // Error occurred while loading the model
                    throw new RuntimeException("Error initializing the TFLite interpreter!", e);
                }
            }
        }).start();
    }

    /* for drawing to the imageView */
    private IntBuffer makeBuffer(int[] src, int n) {
        IntBuffer dst = ByteBuffer.allocateDirect(n * 4)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asIntBuffer();
        for (int i = 0; i < n; i++) {
            dst.put(src[i]);
        }
        dst.rewind();
        return dst;
    }

    /*
     * Loading our Libs
     */
    static {
        System.loadLibrary("echo");
    }

    /*
     * jni function implementations...
     */
    public static native void createSLEngine(int rate, int framesPerBuf);
    public static native void deleteSLEngine();

    public static native boolean createSLBufferQueueAudioPlayer();
    public static native void deleteSLBufferQueueAudioPlayer();

    public static native boolean createAudioRecorder();
    public static native void deleteAudioRecorder();
    public static native void startPlay();
    public static native void stopPlay();

    public static native void writeNewFreq(int freq);
    public static native void getCompleteSamplesBuffer(ShortBuffer bufferPtr);
    public static native void resetParameters();

    /* function for doing mfcc */
    public static native void performMFCC(ShortBuffer bufferPtr, float[] outputArray, short[] trimmed_audio, int [] canvas);
    public static native void createWavFile(ShortBuffer bufferPtr, int num_samples, int sample_rate, byte[] wavFile);
    public native int[] getRowAndCol();
    public native int[] getMFCCParams();
}