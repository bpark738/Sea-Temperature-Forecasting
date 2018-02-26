package temperature;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.conf.preprocessor.RnnToCnnPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.CnnToRnnPreProcessor;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.eval.RegressionEvaluation;

import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;

import java.io.File;

public class Sea3D {

    private static File baseDir = new File("src/main/resources/");

    public static void main(String[] args) throws Exception{

        // File info:

        // Time series sequence length 50
        // Neural network takes in a time series sequence of length 50 and predicts the next time step
        // Time step: 1 day
        // Sequence: 1 csv contains a sequence of length 50, with 52 features representing a portion of the
        // sea temperature grid
        // Contains portions of korea bay, bohai sea, black sea, arabian sea, bengal sea, japan sea
        // mediterranean sea, okhotsk sea


        // Set parameters

        int batchSize = 20;

        int V_HEIGHT = 13;
        int V_WIDTH = 4;
        int kernelSize = 2;
        boolean regression = true;
        int numSkipLines = 1;
        int numChannels = 1;

        // Set directories

        File baseDir = new File("/Users/Briton/Desktop/Skymind/weather/fifty/");
        File featuresDir = new File(baseDir, "features");
        File labelsDir = new File(baseDir, "labels");


        // Initialize SequenceRecordReaders for train split

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(numSkipLines, ",");
        trainFeatures.initialize( new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 1, 1700));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader(numSkipLines, ",");
        trainLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", 1, 1700));

        DataSetIterator train = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, batchSize,
                10, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);


        // Initialize for test split

        SequenceRecordReader testFeatures = new CSVSequenceRecordReader(numSkipLines, ",");
        testFeatures.initialize( new NumberedFileInputSplit(featuresDir.getAbsolutePath() + "/%d.csv", 1701, 2104));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader(numSkipLines, ",");
        testLabels.initialize(new NumberedFileInputSplit(labelsDir.getAbsolutePath() + "/%d.csv", 1701, 2104));

        DataSetIterator test = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, batchSize,
                10, regression, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);;

        // Set LSTM network configuration

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new ConvolutionLayer.Builder(kernelSize, kernelSize)
                        .nIn(1) //1 channel
                        .nOut(7)
                        .stride(2, 2)
                        .learningRate(0.005)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new GravesLSTM.Builder()
                        .activation(Activation.SOFTSIGN)
                        .nIn(84)
                        .nOut(200)
                        .learningRate(0.0005)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .layer(2, new RnnOutputLayer.Builder(LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(200)
                        .learningRate(0.0005)
                        .nOut(52)
                        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                        .gradientNormalizationThreshold(10)
                        .build())
                .inputPreProcessor(0, new RnnToCnnPreProcessor(V_HEIGHT, V_WIDTH, numChannels))
                .inputPreProcessor(1, new CnnToRnnPreProcessor(6, 2, 7 ))
                .pretrain(false).backprop(true)
                .build();

        // Initialize network

        MultiLayerNetwork net = new MultiLayerNetwork(conf);

        net.init();

        // Train model on training set

        for(int i =0 ; i < 25; i++){
            System.out.println("Epoch");
            System.out.println(i);
            net.fit( train );
            train.reset();
        }

        RegressionEvaluation eval = net.evaluateRegression(test);

        test.reset();

       while(test.hasNext()) {
            DataSet next = test.next();
            INDArray features = next.getFeatureMatrix();
            INDArray labels = next.getLabels();
            INDArray inMask = next.getFeaturesMaskArray();
            INDArray outMask = next.getLabelsMaskArray();
            INDArray predicted = net.output(features,false, inMask, outMask);

            eval.evalTimeSeries(labels, predicted, outMask);
       }
       System.out.println( eval.stats() );
    }
}