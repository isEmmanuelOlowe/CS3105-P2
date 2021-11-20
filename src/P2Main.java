import java.io.IOException;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.jblas.util.Logger;

import minet.Dataset;
import minet.layer.*;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import minet.optim.Optimizer;
import minet.optim.SGD;
import minet.util.Pair;

// For file import
import java.io.FileReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import org.json.simple.*;
import org.json.simple.parser.*;

// Helper Method Class
import java.util.Arrays;

public class P2Main {

    static Dataset trainset;
    static Dataset devset;
    static Dataset testset;

    public static void printUsage() {
        System.out.println("Input not recognised. Usage is:");
        System.out.println(
                "\tjava -cp lib/jblas-1.2.5.jar:minet:. P2Main <train_data> <test_data> <random_seed> <json_setting_file> [<apply_preprocessing>]");
        System.out.println("\nExample 1: build an ANN for Part1 and Part2 (no preprocessing is applied)");
        System.out.println(
                "\tjava -cp lib/jblas-1.2.5.jar:minet:. data/Part1/train.txt data/Part1/test.txt 123 settings/example.json");
        System.out.println("Example 2: build an ANN for Part3 (preprocessing is applied)");
        System.out.println(
                "\tjava -cp lib/jblas-1.2.5.jar:minet:. data/Part3/train.txt data/Part3/test.txt 123 settings/example.json 1");
        System.out.println("Example 2: build an ANN for Part3 (preprocessing is not applied)");
        System.out.println(
                "\tjava -cp lib/jblas-1.2.5.jar:minet:. data/Part3/train.txt data/Part3/test.txt 123 settings/example.json");
    }

    // public static void bruteForceCheck(file) {
    //     int cycles = 0;
    //     int index = 0;
    //     int features = trainset.getInputDims();
    //     int trainLength = trainset.getSize();
    //     int testLength = testset.getSize();
    //     double[][] xTrain = trainset.getX();
    //     double[][] xTest = testset.getX();
    //     while (cycles < 7) {
    //         double[][] xTrainFeature = new double[trainLength][features - cycles];
    //         double[][] xTestFeature = new double[testLength][features - cycles];
    //         String[][] output = new String[2][27];
    //         for (int i = 0; i < trainLength; i++) {
    //             int offset = 0;
    //             for (int j = cycles; j < features; j++) {
    //                 if (j == index) {
    //                     offset++;
    //                 }
    //                 else{
    //                     xTrainFeature[i][j] = xTrain[i][j - offset];
    //                 }
    //             }
    //         }

    //         for (int i = 0; i < testLength; i++) {
    //             int offset = 0;
    //             for (int j = cycles; j < features; j++) {
    //                 if (j == index) {
    //                     offset++;
    //                 }
    //                 else {
    //                     xTestFeature[i][j] = xTest[i][j - offset];
    //                 }
    //             }
    //         }
    //         // Making theme defaults
    //         trainset = new Dataset(xTrainFeature, trainset.getY());
    //         testset = new Dataset(xTrainFeature, trainset.getY());

    //         double[][] standard = preprocess_trainset();
    //         preprocess_testset(standard);
    //         ANN ann = new ANN();
    //         buildTrainNetwork(ann, file);
    //         double testAcc = ann.eval(testset);
    //         output[0][cycles + index] =  "0".repeat(cycles) + "0".repeat(index - cycle)+ "1".repeat(features - cycles - index);
    //         index++;
    //         if (index == features){
    //             cycles++;
    //             index = cycles;
    //         }
    //     }
    // }
    /**
     * apply data preprocessing (imputation of missing values and standardisation)
     * on trainset (Part 3 only)
     */
    public static double[][] preprocess_trainset() {
        //// YOUR CODE HERE (PART 3 ONLY)
        double[][] xValues = trainset.getX();
        double[] mean = new double[xValues[0].length];
        double[] sd = new double[xValues[0].length];
        // Computes the mean excluding missing entries
        for (int i = 0; i < xValues.length; i++) {
            for (int j = 0; j < xValues[i].length; j++) {
                    if (xValues[i][j] != 99999) {
                        mean[j] += xValues[i][j];
                    }
            }
        }

        // subdivides the means
        for (int i = 0; i < mean.length; i++) {
            mean[i] /= xValues.length;
        }

        // Calculates the standard deviation using non-empty values
        for (int i = 0; i < xValues.length; i++) {
            for (int j = 0; j < xValues[i].length; j++) {
                    if (xValues[i][j] != 99999) {
                        sd[j] += Math.pow(xValues[i][j] - mean[j],  2);
                    }
            }
        }
        // adjusting standard deviation
        for (int i = 0; i < sd.length; i++) {
            sd[i] /= xValues.length - 1;
        }

        // Standardising data
        // mean of empty values is zero since centred around zero
        for (int i = 0; i < xValues.length; i++) {
            for (int j = 0; j < xValues[i].length; j++) {
                if (xValues[i][j] == 99999) {
                    xValues[i][j] = 0;
                }
                else {
                    xValues[i][j] = (xValues[i][j] - mean[j]) / sd[j];
                }
            }
        }
        trainset = new Dataset(xValues, trainset.getY());
        double[][] standard = {mean, sd};
        return standard;
    }

    /**
     * apply data preprocessing (imputation of missing values and standardisation)
     * on testset (Part 3 only)
     */
    public static void preprocess_testset(double[][] standardisations) {
        //// YOUR CODE HERE (PART 3 ONLY)
        System.out.println(testset);
        double[][] xValues = testset.getX();
        for (int i = 0; i < xValues.length; i++) {
            for (int j = 0; j < xValues[i].length; j++) {
                if (xValues[i][j] == 99999) {
                    xValues[i][j] = 0;
                }
                else {
                    xValues[i][j] = (xValues[i][j] - standardisations[0][j]) / standardisations[1][j];
                }
            }
        }

        testset = new Dataset(xValues, testset.getY());
        
    }
    public static void buildTrainNetwork(ANN ann, String file, Random rnd) {
        JSONParser parser = new JSONParser();
        int hiddenLayers;
        int hiddenLayerNodes;
        String activationFunction;
        double learningRate;
        int batchSize;
        int epochs;
        int patience;
        //// YOUR CODE HERE
        Object obj = parser.parse(new FileReader(file));
        JSONObject jsonObject = (JSONObject) obj;
        hiddenLayers = ((Long) jsonObject.get("n_hidden_layers")).intValue();
        hiddenLayerNodes = ((Long) jsonObject.get("n_nodes_per_hidden_layer")).intValue();
        activationFunction = (String) jsonObject.get("activation_function");
        learningRate = ((Double) jsonObject.get("learning_rate")).doubleValue();
        batchSize = ((Long) jsonObject.get("batchsize")).intValue();
        epochs = ((Long) jsonObject.get("nEpochs")).intValue();
        patience = ((Long) jsonObject.get("patience")).intValue();

        System.out.println("Error Logging Prints...");
        System.out.println("Dimension Test Set Size: " + trainset.getSize());
        System.out.println("Dimension Dev Set Size: " + devset.getSize());
        // build and train an ANN with the given data and parameters

        int OUTPUT_DIMENSIONS = 3;
        // building the network
        Layer network = ann.build(trainset.getInputDims(), OUTPUT_DIMENSIONS, hiddenLayers, hiddenLayerNodes,
                activationFunction);
        Loss crossEntropy = new CrossEntropy();
        Optimizer sGradientDescent = new SGD(network, learningRate);
        // training the network
        ann.train(crossEntropy, sGradientDescent, trainset, devset, batchSize, epochs, patience, rnd);
    }
    public static void main(String[] args) {
        if (args.length < 4) {
            printUsage();
            return;
        }
        try {
            // set jblas random seed (for reproducibility)
            org.jblas.util.Random.seed(Integer.parseInt(args[2]));
            Random rnd = new Random(Integer.parseInt(args[2]));

            // turn off jblas info messages
            Logger.getLogger().setLevel(Logger.WARNING);

            // load train and test data into trainset and testset
            System.out.println("Loading data...");
            //// YOUR CODE HERE
            System.out.println(Arrays.toString(args));
            trainset = Dataset.loadTxt(args[0]);
            testset = Dataset.loadTxt(args[1]);

            // check whether data-preprocessing is applied (Part 3)
            boolean preprocess = false;
            if (args.length == 5) {
                if (!args[4].equals("0") && !args[4].equals("1")) {
                    System.out.println("HERE" + args[4]);
                    printUsage();
                    System.out
                            .println("\nError: <apply_preprocessing> must be either empty (off), or 0 (off) or 1 (on)");
                    return;
                }
                if (args[4].equals("1"))
                    preprocess = true;
            }

            // apply data-processing on trainset
            double[][] standardisation;
            if (preprocess) {
                standardisation = preprocess_trainset();
            }
            // to get the compiler to leave me alone!
            else {
                standardisation = new double[0][0];
            }

            // split train set into train set (trainset) and validation set, also called
            // development set (devset)
            // suggested split ratio: 80/20
            trainset.shuffle(rnd); // shuffle the train data before we split. NOTE: this line was updated on Nov
                                   // 11th.
            //// YOUR CODE HERE
            int length = trainset.getSize();
            int devLength = (int) Math.floor(length * 0.2);
            double[][] devSetX = Arrays.copyOfRange(trainset.getX(), 1, devLength);
            double[][] devSetY = Arrays.copyOfRange(trainset.getY(), 1, devLength);

            double[][] trainingSetX = Arrays.copyOfRange(trainset.getX(), devLength, length);
            double[][] trainingSetY = Arrays.copyOfRange(trainset.getY(), devLength, length);
            devset = new Dataset(devSetX, devSetY);
            trainset = new Dataset(trainingSetX, trainingSetY);

            // read all parameters from the provided json setting file (see
            // settings/example.json for an example)
            //// YOUR CODE HERE
            ANN ann = new ANN();
            buildTrainNetwork(ann, args[3], rnd);
            // evaluate the trained ANN on the test set and report results
            try {
                // apply data-preprocessing on testset
                if (preprocess)
                    preprocess_testset(standardisation);
                double testAcc = ann.eval(testset);
                System.out.println("accuracy on test set: " + testAcc);
            } catch (Exception e) {
                System.out.println(e.getMessage());
                return;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}
