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
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
//A mapping from String keys to various Parcelable values (interface for data container values, parcels)
import android.graphics.Rect;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.MediaDataSource;
import android.media.MediaPlayer;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
//Object used to report movement (mouse, pen, finger, trackball) events.
// //Motion events may hold either absolute or relative movements and other data, depending on the type of device.
import android.os.Handler;
import android.provider.MediaStore;
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
import android.os.CountDownTimer;
import java.util.Calendar;
import java.util.Date;

/* tensor flow lite requirements */
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Tensor;


/* added OnRequestPerm... which i believe opens dialogue to ask for speaker permission */
//public class MainActivity extends Activity implements View.OnClickListener, View.OnTouchListener,
//        ActivityCompat.OnRequestPermissionsResultCallback {
    public class MainActivity extends Activity implements ActivityCompat.OnRequestPermissionsResultCallback {

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

    private short[] audio_samples;
    private int audio_samples_idx = 0;
    private Timer timer;
    private int recording_duration = 0; // duration of recording in ms, should end at 30,000 ms

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
//    private DrawModel drawModel;
//    private DrawView drawView;
//    private PointF mTmpPiont = new PointF();

    ImageView mfccView;
    Bitmap bitmap;
    Canvas canvas;
    Paint paint;

    private float mLastX;
    private float mLastY;

    private static final String FILENAME = "data.txt";
    private static final String RAW_AUDIO = "/sdcard/data/raw_samples/";
    private static final String PROCESSED_AUDIO = "/sdcard/data/processed_samples/";
    private static final String TRIMMED_AUDIO = "/sdcard/data/trimmed_samples/";
    private static final String MFCC_IMAGES = "/sdcard/data/mfcc_images/";
    private static final String RAW_WAV_AUDIO = "/sdcard/data/raw_wav_audio/";
    private static final String TRIMMED_WAV_AUDIO = "/sdcard/data/trimmed_wav_audio/";


    /* tensorflow lite private variables */
    private Interpreter tflite = null;
    final int use_dummy = 0;

    private double [] row1 = {17.954090056507162, 17.134420202199646, 18.82351505056716, 12.691770218557354, 16.47000472864695, 2.569986051352495, 2.611463290654575, -1.9272728635974279, -1.8062854858962114, -2.3772770884794645, -1.5846625592097099, 0.32049544476559333, 0.13082579852031592, -0.5955630726663632, -0.05708389621138675, 0.6306620146210697, 3.412809426428865, 9.651002038442387, 9.865285253436921, 11.872856246604671, 13.976273861364328, 12.791892374504416, 13.008364322510426, 17.630537069931304, 17.073059293494595, 14.625866497920976, -13.789432947964187, -9.930444010563154, -14.889156230726073, -15.40412695470952, -16.647555969250938, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    private double [] row2 = {1.5539615353624774, 3.7481556366959747, 2.610512879371151, -3.292260182329394, -0.8338261349616629, 0.11621704804074261, 6.056830133924504, 5.599254307594642, 6.66324438783517, 8.148450693685328, 8.627886503344987, 8.908196290483184, 9.582802129319262, 14.086574674302557, 15.357921236226689, 17.761045037182686, 20.526029157436, 24.002706509351082, 21.192010735831104, 18.295057977357057, 17.601636967983328, 12.60070924634285, 8.943322958141943, 7.27721504140753, 5.675638479803613, 5.848071146062688, 2.91747039963101, 4.0208103560328405, 5.10752570388466, 2.80319751137566, 1.0111087673049268, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    private double [] row3 = {4.2085353864425095, 1.8864789331662455, 1.8116608299062373, 2.8084724800304235, 3.6704986929130676, 9.031025803647397, 9.943442509522797, 8.112193966639028, 7.380632365501264, 8.278476434554932, 9.252190861606003, 10.307229034327793, 11.764251247515524, 12.777384504342779, 13.3090782968982, 11.806549343824841, 10.874127733111486, 12.588500749980415, 11.656534530216991, 9.892658801233093, 13.012894932127873, 9.515202600529532, 6.909593245260058, 5.948052281439882, 7.760168213442214, 8.11480005389974, -0.4437108978570329, -0.24269366987807006, -1.7050573573665773, 0.6452584954044593, 0.27597354817743797, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    private double [] row4 = {9.154530485283145, 7.740789667192769, 7.643465163827281, 7.265628311383738, 8.401363280250973, -19.089283702640113, -18.70605202942497, -21.62273418194813, -19.902609294684954, -19.92125135855562, -20.084499005934738, -19.50315703940026, -18.811194873332372, -19.378056999205906, -19.004322316363037, -17.91161888899419, -18.24583255353676, -16.572074230506228, -15.319362466571791, -10.863037731326976, -5.75991235032996, -1.3587084782398178, -0.2143208127544667, 4.843396757386257, 5.185795041702421, 9.01913586400659, 0.3937286433139497, -3.430733721598184, -6.385083292205781, 1.754063316882708, -3.2688053270651776, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    private double [] row5 = {-7.216014399515556, -9.457790376876764, -8.578818000891713, -9.61806852751709, -9.618686403645013, -8.307983772658044, -11.269940956343675, -12.606468481871138, -11.246415551257192, -11.069780067127834, -10.589032667090013, -10.558245568606399, -11.505550974661725, -9.544887490630854, -9.282130481452683, -8.114743729894903, -9.350784721282633, -8.232531074889396, -9.659573567555878, -9.267384689134724, -7.659306595510143, -6.634587675171637, -7.4903816076752765, -4.637861317362229, -6.154328367840509, -4.940031152250643, -6.785844870969893, -9.119108916490385, -10.11664145049991, -8.538814260624074, -10.209624400498413, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    private double [] row6 = {7.779367240235136, 6.8713514165078164, 8.154655682396342, 7.122404266795532, 8.62306876518846, 5.64523253587272, 7.95406032031112, 6.293829236098881, 6.168063885340055, 5.1130335620385745, 5.386805730493693, 4.678886023645577, 4.244966614649471, 2.440726094736848, 1.6961991784249004, 1.0381507596395423, -2.0485476189681533, -0.08467420084470795, 3.872387357847007, -6.923486216239046, 0.5458056410206407, 0.4308417856266627, 2.2236656281399942, 6.915017466585008, 8.65488298748111, 7.564676013867389, 5.344449924361961, 9.70091202433287, 6.925494277147561, 10.56017387055034, 10.246321295257868, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    private double [] row7 = {-4.946492068613723, -1.277759415193349, -3.2181972113785533, -3.496015639621457, -3.7701694199117117, -4.361511477476806, 3.20938391169513, 1.89568531747203, 1.9312283905861851, 1.5158057142125037, 0.12002958724253576, -0.4354346467567604, 0.2103022037558166, 0.25422975007846255, -2.5582598683211204, -4.7031257761778225, -4.73246720726341, -4.431186299379764, -5.582235667538581, -9.367544547325915, -6.7853452692515965, -2.7168496260257378, -2.2649526566272575, -3.667545951877727, -3.119138110817085, -4.12891046844904, -0.49821298440235434, -1.3803843349070484, -1.2615901434482142, 0.11209633176549678, 1.112596765652587, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    private double [] row8 = {-0.6098068623160662, 4.224580771852048, 0.24965063609220398, 1.7915164945096234, 0.4993454777121551, -3.429390998864377, -3.6033066354672374, -4.629598636382162, -2.869596546822773, -1.7543573329831317, -1.5864095914611773, 0.08000640309622323, -2.7696181119936845, -2.6969581239490235, -4.542845611881809, -5.534051338142526, -6.031874751647745, -3.85664990772877, -4.102049021179447, -2.357076200707108, -2.244703371325563, 1.4757071614736161, -2.2682089713921108, 1.3826456738226776, -0.8045892144604538, 1.4667962151446912, -0.7869712742877265, -1.5239700857865117, -3.307494504717217, -3.214116066840672, -3.433723555516111, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    private double [] row9 = {-9.576808690846088, -8.834755867832621, -9.616253757793002, -8.512682266318464, -7.123900287268096, -7.789415576608349, -3.7625135845328797, -3.939811220354241, -0.8175711434238476, 0.19492655343274945, 1.5276464526369118, 0.35253486124431216, 2.283592405134869, 1.152820385966445, -1.5685921864821415, -1.1446229586903884, -1.4861482196401095, -1.8327807612435092, -8.637231531368686, -4.006916387416697, -1.9548891596552231, -7.241309737832162, -5.836913219224373, -5.94001322638014, -8.624721801248564, -8.46083676687682, -1.5835147253788748, -1.3343192551112084, -4.814750657005303, -2.6821553352990595, -1.5846529278304569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    private double [] row10 = {3.409544132221265, 4.444465189440969, 5.779899276130373, 5.387863573199064, 7.510383384504154, -0.5768938485804497, -3.1856024072949585, -4.240631864002894, -4.915085996188106, -4.166251640985384, -4.03499700614479, -4.621708299467147, -5.338073457213545, -2.5754646750234294, -3.2511751198963124, -3.5874786689289735, -2.8655095834487643, -3.0106489038528164, -2.887338309635184, -1.9866587568740668, 2.41964941974033, 0.41321429639074553, 0.7887054003775955, 0.3392738634980675, 3.7936529447811327, 3.548877235494176, 2.8548793748643284, 0.8168692720363299, -2.2492803835315422, 0.3431245324682159, -0.6470013169419243, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    private double [] row11 = {-5.275847067550232, -3.7602681891276277, -3.3858839486430363, -3.2536046299463663, -5.210514774539606, -9.271401732880722, -8.893487654232406, -10.951039403516331, -9.34629362982836, -8.219022962717705, -7.793793307799313, -5.862411689440904, -5.930107616466966, -5.2821308136790845, -3.8019269948340892, -1.979673059724152, -3.5324091622677427, -4.381063009897511, -4.942928730359676, -7.794374110290029, -6.382924799955393, -6.12584430445437, -5.025731716308161, -5.477988524531365, -4.030065532452349, -7.095471919999615, -4.743998153355815, -2.950549192294184, -3.246448960292778, -6.266065246422104, -4.307627161633517, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    private double [] row12 = {4.608795255997088, 6.698753287757723, 8.236016873501589, 7.700499116598714, 5.245252962919166, 3.8466566539754545, 0.2557723577548203, -2.224424702878995, -1.7538192820271359, -1.6108894693193423, -1.5526189819315854, -1.4217778875485052, 0.6884055890191354, 0.19267831195985646, 2.3121828630044594, 4.709698465608981, 4.254232469349811, 4.867784266241455, 3.6521102731532706, 5.176660526987325, 5.1262866046084, 7.18305450651508, 7.307006051256813, 9.797304678514347, 6.722168772648709, 7.086903804628322, 2.1571777229279383, 2.0079520449434693, -0.6501586804137662, -3.51383108746593, -3.809193133424525, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};


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

//        //get drawing view from XML (where the finger writes the number)
//        drawView = (DrawView) findViewById(R.id.draw);
//        //get the model object
//        drawModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);
//
//        //init the view with the model object
//        drawView.setModel(drawModel);
//        // give it a touch listener to activate when the user taps
//        drawView.setOnTouchListener(this);

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
                    int step = nfft - noverlap;
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

                    // Prepare input data
                    float[][][][] inputArray = new float[1][mfcc_rows][mfcc_cols][1]; // Assuming your input is a 1D array
                    float[][] outputArray = new float[1][10]; // Assuming you have a classification problem with NUM_CLASSES classes

                    /* loads dummy data into the input array to test the tensor flow lite model */
                    if (use_dummy == 1) {
                        for (int i = 0; i < 48; i++) {
                            inputArray[0][0][i][0] = (float) row1[i];
                            inputArray[0][1][i][0] = (float) row2[i];
                            inputArray[0][2][i][0] = (float) row3[i];
                            inputArray[0][3][i][0] = (float) row4[i];
                            inputArray[0][4][i][0] = (float) row5[i];
                            inputArray[0][5][i][0] = (float) row6[i];
                            inputArray[0][6][i][0] = (float) row7[i];
                            inputArray[0][7][i][0] = (float) row8[i];
                            inputArray[0][8][i][0] = (float) row9[i];
                            inputArray[0][9][i][0] = (float) row10[i];
                            inputArray[0][10][i][0] = (float) row11[i];
                            inputArray[0][11][i][0] = (float) row12[i];
                        }
                    }
                    else {
                        float temp;
                        for (int row = 0; row < mfcc_rows; row++) {
                            for (int col = 0; col < mfcc_cols; col++) {
                                /* need to cut out NaN since it destroys CNN result */
                                temp = mfcc_output[row * mfcc_cols + col];
                                inputArray[0][row][col][0] = Float.isNaN(temp) ? 0 : temp;
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

        // reursive case

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
//                    AssetFileDescriptor fileDescriptor = getAssets().openFd("my_model_12_48.tflite");
//                    AssetFileDescriptor fileDescriptor = getAssets().openFd("my_model_12_28.tflite");
                    AssetFileDescriptor fileDescriptor = getAssets().openFd("my_model_12_28_normalized.tflite");
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