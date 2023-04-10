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
import android.graphics.Color;
import android.graphics.PointF;
//A mapping from String keys to various Parcelable values (interface for data container values, parcels)
import android.graphics.Rect;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.os.AsyncTask;
import android.os.Bundle;
//Object used to report movement (mouse, pen, finger, trackball) events.
// //Motion events may hold either absolute or relative movements and other data, depending on the type of device.
import android.os.Handler;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.view.MotionEvent;
//This class represents the basic building block for user interface components.
// A View occupies a rectangular area on the screen and is responsible for drawing
import android.view.View;
//A user interface element the user can tap or click to perform an action.
import android.view.WindowManager;
import android.widget.Button;
//A user interface element that displays text to the user. To provide user-editable text, see EditText.
import android.widget.TextView;
import android.widget.Toast;
//Resizable-array implementation of the List interface. Implements all optional list operations, and permits all elements,
// including null. In addition to implementing the List interface, this class provides methods to
// //manipulate the size of the array that is used internally to store the list.
import java.util.ArrayList;
// basic list
import java.util.List;
//encapsulates a classified image
//public interface to the classification class, exposing a name and the recognize function
import mariannelinhares.mnistandroid.models.Classification;
import mariannelinhares.mnistandroid.models.Classifier;
//contains logic for reading labels, creating classifier, and classifying
import mariannelinhares.mnistandroid.models.TensorFlowClassifier;
//class for drawing MNIST digits by finger
import mariannelinhares.mnistandroid.views.DrawModel;
//class for drawing the entire app
import mariannelinhares.mnistandroid.views.DrawView;
// for creating directories and files from java
import java.io.File;
import android.os.Environment;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Timer;
import java.util.TimerTask;

/* added OnRequestPerm... which i believe opens dialogue to ask for speaker permission */
public class MainActivity extends Activity implements View.OnClickListener, View.OnTouchListener,
        ActivityCompat.OnRequestPermissionsResultCallback {
//public class MainActivity extends Activity implements View.OnClickListener, View.OnTouchListener {

    private static final int PIXEL_WIDTH = 28;

    /**********************************************************************************************/
    /**                                 LAB 5 SETUP                                              **/
    // UI Variables from Lab 5
    Button   controlButton;
    TextView statusView;
    TextView freq_status_view;
    String  nativeSampleRate;
    String  nativeSampleBufSize;
    boolean supportRecording;
    Boolean isPlaying = false;

    // Static Values from Lab 5
    private static final int AUDIO_ECHO_REQUEST = 0;
    private static final int EXTERNAL_STORAGE_REQUEST = 0;
    private static final int FRAME_SIZE = 1024;
    private static final int MIN_FREQ = 50;

    private float[] audio_samples;
    private int audio_samples_idx = 0;
    private Timer timer;
    private int recording_duration = 0; // duration of recording in ms, should end at 30,000 ms


    /**********************************************************************************************/


    // ui elements
    private Button clearBtn, classBtn;
    private TextView resText;
    private List<Classifier> mClassifiers = new ArrayList<>();

    // views
    private DrawModel drawModel;
    private DrawView drawView;
    private PointF mTmpPiont = new PointF();

    private float mLastX;
    private float mLastY;

    private static final String FILENAME = "data.txt";
    private static final String DNAME = "/sdcard/data/audio/"; //  "myfiles";

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
        audio_samples = new float[48000 * 3];
        timer = new Timer();

        /**********************************************************************************************/

        //get drawing view from XML (where the finger writes the number)
        drawView = (DrawView) findViewById(R.id.draw);
        //get the model object
        drawModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);

        //init the view with the model object
        drawView.setModel(drawModel);
        // give it a touch listener to activate when the user taps
        drawView.setOnTouchListener(this);

        //clear button
        //clear the drawing when the user taps
        clearBtn = (Button) findViewById(R.id.btn_clear);
        clearBtn.setOnClickListener(this);

        //class button
        //when tapped, this performs classification on the drawn image
        classBtn = (Button) findViewById(R.id.btn_class);
        classBtn.setOnClickListener(this);

        // res text
        //this is the text that shows the output of the classification
        resText = (TextView) findViewById(R.id.tfRes);

        // tensorflow
        //load up our saved model to perform inference from local storage
        loadModel();
    }

    //the activity lifecycle

    @Override
    //OnResume() is called when the user resumes his Activity which he left a while ago,
    // //say he presses home button and then comes back to app, onResume() is called.
    protected void onResume() {
        drawView.onResume();
        super.onResume();
    }

    @Override
    //OnPause() is called when the user receives an event like a call or a text message,
    // //when onPause() is called the Activity may be partially or completely hidden.
    protected void onPause() {
        drawView.onPause();
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
                                    "input", "output", true));
                    mClassifiers.add(
                            TensorFlowClassifier.create(getAssets(), "Keras",
                                    "opt_mnist_convnet-keras.pb", "labels.txt", PIXEL_WIDTH,
                                    "conv2d_1_input", "dense_2/Softmax", false));
                } catch (final Exception e) {
                    //if they aren't found, throw an error!
                    throw new RuntimeException("Error initializing classifiers!", e);
                }
            }
        }).start();
    }

    @Override
    public void onClick(View view) {
        //when the user clicks something
        if (view.getId() == R.id.btn_clear) {
            //if its the clear button
            //clear the drawing
            drawModel.clear();
            drawView.reset();
            drawView.invalidate();
            //empty the text view
            resText.setText("");
        } else if (view.getId() == R.id.btn_class) {
            //if the user clicks the classify button
            //get the pixel data and store it in an array
            float pixels[] = drawView.getPixelData();

            //init an empty string to fill with the classification output
            String text = "";
            //for each classifier in our array
            for (Classifier classifier : mClassifiers) {
                //perform classification on the image
                final Classification res = classifier.recognize(pixels);
                //if it can't classify, output a question mark
                if (res.getLabel() == null) {
                    text += classifier.name() + ": ?\n";
                } else {
                    //else output its name
                    text += String.format("%s: %s, %f\n", classifier.name(), res.getLabel(),
                            res.getConf());
                }
            }
            resText.setText(text);
        }
    }

    @Override
    //this method detects which direction a user is moving
    //their finger and draws a line accordingly in that
    //direction
    public boolean onTouch(View v, MotionEvent event) {
        //get the action and store it as an int
        int action = event.getAction() & MotionEvent.ACTION_MASK;
        //actions have predefined ints, lets match
        //to detect, if the user has touched, which direction the users finger is
        //moving, and if they've stopped moving

        //if touched
        if (action == MotionEvent.ACTION_DOWN) {
            //begin drawing line
            processTouchDown(event);
            return true;
            //draw line in every direction the user moves
        } else if (action == MotionEvent.ACTION_MOVE) {
            processTouchMove(event);
            return true;
            //if finger is lifted, stop drawing
        } else if (action == MotionEvent.ACTION_UP) {
            processTouchUp();
            return true;
        }
        return false;
    }

    //draw line down

    private void processTouchDown(MotionEvent event) {
        //calculate the x, y coordinates where the user has touched
        mLastX = event.getX();
        mLastY = event.getY();
        //user them to calcualte the position
        drawView.calcPos(mLastX, mLastY, mTmpPiont);
        //store them in memory to draw a line between the
        //difference in positions
        float lastConvX = mTmpPiont.x;
        float lastConvY = mTmpPiont.y;
        //and begin the line drawing
        drawModel.startLine(lastConvX, lastConvY);
    }

    //the main drawing function
    //it actually stores all the drawing positions
    //into the drawmodel object
    //we actually render the drawing from that object
    //in the drawrenderer class
    private void processTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        drawView.calcPos(x, y, mTmpPiont);
        float newConvX = mTmpPiont.x;
        float newConvY = mTmpPiont.y;
        drawModel.addLineElem(newConvX, newConvY);

        mLastX = x;
        mLastY = y;
        drawView.invalidate();
    }

    private void processTouchUp() {
        drawModel.endLine();
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

            /* start timer */
            final Handler handler = new Handler();
            TimerTask doAsynchronousTask = new TimerTask() {
                @Override
                public void run() {
                    handler.post(new Runnable() {
                        public void run() {
                            try {
                                /* need to fix this but will be a function that writes to buffer, and timer task should stop
                                once buffer is filled all the way ( 3 second buffer )
                                 */
                                UpdateStftTask performStftUiUpdate = new UpdateStftTask();
                                performStftUiUpdate.execute();
                            } catch (Exception e) {
                                // TODO Auto-generated catch block
                            }
                        }
                    });
                }
            };
            timer.schedule(doAsynchronousTask, 0, 10); // execute every 10 ms
        } else {
            stopPlay();  //this must include stopRecording()
            updateNativeAudioUI();
            deleteAudioRecorder();
            deleteSLBufferQueueAudioPlayer();

            /* stop timer */
//            timer.cancel();
//            recording_duration = 0; // reset back to 0 ms
//            audio_samples_idx = 0;


        }
        isPlaying = !isPlaying;
        controlButton.setText(getString((isPlaying == true) ?
                R.string.StopEcho: R.string.StartEcho));
    }

    private class UpdateStftTask extends AsyncTask<Void, FloatBuffer, Void> {
        @Override
        protected Void doInBackground(Void... params) {

            // Float == 4 bytes
            // Note: We're using FloatBuffer instead of float array because interfacing with JNI
            // with a FloatBuffer allows direct memory sharing, versus having to copy to some
            // intermediate location first.
            // http://stackoverflow.com/questions/10697161/why-floatbuffer-instead-of-float

            /* multiply by 2 because for 50% overlap, we will now publish two FFT's per call */
            FloatBuffer buffer = ByteBuffer.allocateDirect(FRAME_SIZE * 4)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asFloatBuffer();

            getSamplesBuffer(buffer);

            // Update screen, needs to be done on UI thread
            publishProgress(buffer);

            return null;
        }

        protected void onProgressUpdate(FloatBuffer... newBufferSamples) {

            int newBufferCapacity = newBufferSamples[0].capacity();
            for (int i = 0; i < newBufferCapacity; i++) {
                if (audio_samples_idx + i >= audio_samples.length)
                    break;
                audio_samples[audio_samples_idx + i] = newBufferSamples[0].get();
            }
            audio_samples_idx += newBufferCapacity;
            recording_duration += 10; // this should presumably take 10 ms

            /* check if we are past 3 seconds of recording */
            if (recording_duration >= 30000) {
                /* make sure asynnchronous task is stopped */
                timer.cancel();
                recording_duration = 0; // reset back to 0 ms
                audio_samples_idx = 0;

                /* write 3 second array of samples to a csv file */
                File rootPath = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + DNAME, "test_samples.csv");
                writeSamplesToCSV(rootPath);

            }

            newBufferSamples[0].rewind();
        }
    }

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

        File rootPath = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + DNAME, FILENAME);
        writeTextData(rootPath, "Hello world");
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

        File rootPath = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS) + DNAME, FILENAME);
        writeTextData(rootPath, "Hello world");
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

    private void writeSamplesToCSV(File file) {
        FileOutputStream fileOutputStream = null;
        try {
            fileOutputStream = new FileOutputStream(file);
            for (int i = 0; i < audio_samples.length; i++) {
                fileOutputStream.write(String.format("%f\n", audio_samples[i]).getBytes());
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
    public static native void getSamplesBuffer(FloatBuffer buffer);
}