package org.deeplearning4j;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.examples.advanced.features.transferlearning.editlastlayer.EditLastLayerOthersFrozen;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

import java.io.File;
import java.util.*;

public class NehcConv {
    private static final Logger LOGGER = org.slf4j.LoggerFactory.getLogger(EditLastLayerOthersFrozen.class);
    protected static final int numClasses = 2;
    protected static final long seed = 12345;

    private static final int batchSize = 32;
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 3;
    private static final int outputNum = 2;
    private static final int numEpochs = 20;

    public static void main(String[] args) throws Exception{
        Random randNumGen = new Random(seed);
        //file paths
        File trainData = new File("D:\\CBIS-DDSM-train");
        File testData = new File("D:\\CBIS-DDSM-test");

        //file split
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);

        //initialize the record reader
        //add listener
        recordReader.initialize(train);
        //recordReader.setListeners(new LogRecordListener());

        //dataset iterator
        DataSetIterator dataIterate = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        // Scale pixel values to 0-1
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1, 16);
        dataIterate.setPreProcessor(scaler);

        //initialize pretrained VGG16 with IMAGENET weights

        ZooModel zooModel = VGG16.builder()
            .build();
        ComputationGraph net = (ComputationGraph)zooModel.initPretrained(PretrainedType.IMAGENET);

        Map<Integer, Double> learningRateSchedule = new HashMap<>();
        learningRateSchedule.put(0, 1e-7);
        learningRateSchedule.put(100, 9e-8);
        learningRateSchedule.put(200, 4e-8);
        learningRateSchedule.put(400, 8e-9);
        learningRateSchedule.put(800, 1e-9);

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION, learningRateSchedule), 0.9))
            .seed(seed)
            .build();
        ComputationGraph nehcNet = new TransferLearning.GraphBuilder(net)
            .fineTuneConfiguration(fineTuneConf)
            .removeVertexKeepConnections("predictions")
            .addLayer("predictions", new OutputLayer
                .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(4096)
                .nOut(2)
                .weightInit(WeightInit.RELU)
                .activation(Activation.SOFTMAX)
                .build(), "fc2")
            .setOutputs("predictions")
            .build();
        nehcNet.setListeners(new ScoreIterationListener(10));
        LOGGER.info(nehcNet.summary());
        LOGGER.info("*****Train Model*****");

        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(test);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);

        scaler.fit(testIter);
        testIter.setPreProcessor(scaler);
        /*
        for (int i = 1; i < 3; i++)
        {
            DataSet ds = testIter.next();
            System.out.println(ds);
            System.out.println(testIter.getLabels());
        }
        System.exit(0);

         */

        LOGGER.info("Eval stats BEFORE fit.....");
        Evaluation eval0 = nehcNet.evaluate(testIter);
        LOGGER.info(eval0.stats() + "\n");
        testIter.reset();
        nehcNet.setListeners(new ScoreIterationListener(10));

        int iter = 1;
        for (int i = 0; i < numEpochs; i++){
            //nehcNet.fit(dataIterate);
            while (dataIterate.hasNext()) {
                nehcNet.fit(dataIterate.next());
                if (iter % 50 == 0)
                {
                    LOGGER.info("Evaluate model at iter "+iter +" ....");
                    Evaluation eval = nehcNet.evaluate(testIter);
                    LOGGER.info(eval.stats());
                    testIter.reset();
                }
                iter++;
            }
            LOGGER.info("Completed epoch {}", i);
            dataIterate.reset();
        }


    }
}
