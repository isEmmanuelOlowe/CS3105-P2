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

import java.util.TreeMap;
import java.util.ArrayList;
import java.util.Arrays;
// For file import
import java.io.FileReader;
import java.io.FileWriter;
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


    /**
     * A experiment to determine which combination of features results in the highest evaulation
     * 
     * @param rnd for reproducibility
     * @throws Exception an exceptions thrown
     */
    public static void featureImportance(Random rnd, String file) throws Exception {
        // So data is not contained with results from test set
        double[][] standard = preprocess_trainset();
        preprocess_testset(standard, devset);
        // extracts new devset for experiment
        int length = trainset.getSize();
        int devLength = (int) Math.floor(length * 0.2);
        double[][] exSetX = Arrays.copyOfRange(trainset.getX(), 1, devLength);
        double[][] exSetY = Arrays.copyOfRange(trainset.getY(), 1, devLength);

        double[][] trainingSetX = Arrays.copyOfRange(trainset.getX(), devLength, length);
        double[][] trainingSetY = Arrays.copyOfRange(trainset.getY(), devLength, length);
        Dataset experimentset = new Dataset(exSetX, exSetY);
        trainset = new Dataset(trainingSetX, trainingSetY);
        
        String[] FEATURES = {"1", "2", "3", "4", "5", "6", "7"};

        TreeMap<String, Double> data = new TreeMap<String, Double>();
        testFeaures(trainset, experimentset, devset, FEATURES, rnd, data, file);
        print_results(data, "Features in Use", "data/features/features.csv");
        
    }

    /**
     * Evaluates the model for the current features and recursively removes features
     * @param trainset the training data
     * @param devset the development data
     * @param testset the test data for evaulation | doesn't use test.txt is subset of trainset
     * @param features features being evaulated
     * @param rnd for reproducibility
     * @param data stores the results of the experiments
     * @throws Exception
     */
    public static void testFeaures(Dataset trainset, Dataset devset, Dataset testset, String[] features, Random rnd, TreeMap<String, Double> data, String file) throws Exception {
        // Run neural network add output to 
        ANN ann = new ANN();
        buildTrainNetwork(ann, file, rnd, trainset, devset);
        double eval = ann.eval(testset);
        data.put(Arrays.toString(features).replaceAll(",", ""), eval);
        if (features.length > 1) {
            for (int i = 0; i < features.length; i++) {
                Dataset newTrainset = extract(trainset, i);
                Dataset newDevset = extract(devset, i);
                Dataset newTestset = extract(testset, i);
                String[] newFeatures = removeIndex(features, i);
                testFeaures(newTrainset, newDevset, newTestset, newFeatures, rnd, data, file);
            }
        }

    }
    
    /**
     * Removes a feature from the dataset
     * @param dataset the dataset being manipulated
     * @param index the feature being removed
     * @return a new dataset without removed feature
     */
    public static Dataset extract(Dataset dataset, int index) {
        double[][] x = dataset.getX();
        double[][] newX = new double[dataset.getSize()][dataset.getInputDims() - 1];
        for (int i = 0; i < dataset.getSize(); i++) {
            int offset = 0;
            for (int j = 0; j < dataset.getInputDims() - 1; j++) {
                if (j != index) {
                    newX[i][j] = x[i][j + offset];
                }
                else {
                    offset++;
                }
            }
        }
        return new Dataset(newX, dataset.getY());
    }

    /**
     * Removes an index from an array
     * @param array the array being manipulated
     * @param index the index being removed
     * @return a new array without the removed index
     */
    public static String[] removeIndex(String[] array, int index) {
        String[] newArray = new String[array.length - 1];
        for (int i = 0; i < newArray.length; i++) {
            int offset = 0;
            if (i != index) {
                newArray[i] = array[i + offset];
            }
            else {
                offset++;
            }
        }
        return newArray;
    }


    /**
     * apply data preprocessing (imputation of missing values and standardisation)
     * on trainset (Part 3 only)
     * 
     * @param rnd for reproducibility
     * @throws Exception for any exceptions thrown
     */
    public static void randomHyperParameters(Random rnd) throws Exception{
        // So data is not contained with results from test set
        double[][] standard = preprocess_trainset();
        preprocess_testset(standard, devset);
        // extracts new devset for experiment
        int length = trainset.getSize();
        int devLength = (int) Math.floor(length * 0.2);
        double[][] exSetX = Arrays.copyOfRange(trainset.getX(), 1, devLength);
        double[][] exSetY = Arrays.copyOfRange(trainset.getY(), 1, devLength);

        double[][] trainingSetX = Arrays.copyOfRange(trainset.getX(), devLength, length);
        double[][] trainingSetY = Arrays.copyOfRange(trainset.getY(), devLength, length);
        Dataset experimentset = new Dataset(exSetX, exSetY);
        trainset = new Dataset(trainingSetX, trainingSetY);
        // Search Space for Hidden Layers
        int MAX_HIDDEN_LAYERS = 3;
        // Search Spaces for Nodes Per Hidden Layer
        int MAX_HIDDEN_LAYER_NODES = 40;
        // Search Space for Activation Function
        String[] activationFunction = {"ReLU", "Sigmoid", "Tanh", "Softmax"};
        // Will hold the various learning rates use form 0.1 to 1 (search space)
        double learningRate;
        // The maximum increments of learning rate by 0.01
        double MAX_INCREMENT = 4;
        // output dimensions
        int OUTPUT_DIMS = 3;
        // default values
        int batchSize = 128;
        int epochs = 2000;
        int patience = 100;

        //Stores all the data Generated
        TreeMap<String, Double> data = new TreeMap<String, Double>();

        ANN ann = new ANN();
        for (int i = 1; i < MAX_HIDDEN_LAYERS; i++) {
            for (int j = 22; j < MAX_HIDDEN_LAYER_NODES; j++) {
                // resonable starting value
                learningRate = 1;
                for (int k = 0; k < activationFunction.length; k++) {
                    for (int l = 0; l < MAX_INCREMENT; l++) {
                        learningRate += 0.05;
                        Layer network = ann.build(trainset.getInputDims(), OUTPUT_DIMS, i, j, activationFunction[k]);
                        Loss crossEntropy = new CrossEntropy();
                        Optimizer sGradientDescent = new SGD(network, learningRate);
                        ann.train(crossEntropy, sGradientDescent, 
                                trainset, experimentset, batchSize, epochs, patience, rnd);
                        double testAcc = ann.eval(devset);
                        data.put(i+","+j+","+activationFunction[k]+","+learningRate,testAcc);
                    }
                }
            }
        }
        print_results(data, 
                "Number of Hidden Layers, Number of Nodes per Hidden Layer, Activation function, learning rate, accuracy\n", "data/experiments/hyperparameters.csv");
    }
    
    /**
     * Prints the results of the experiments to a file
     * @param data the results being printed
     * @throws Exception for any filewrting errors thrown
     */
    public static void print_results(TreeMap<String, Double> data, String head, String filePath) throws Exception {
        FileWriter myWriter = new FileWriter(filePath);
        myWriter.write(head);
        for (String key : data.keySet()) {
            myWriter.write(key + "," + data.get(key) + "\n");
        }
        myWriter.close();
    }
    
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
    public static void preprocess_testset(double[][] standardisations, Dataset dataset) {
        //// YOUR CODE HERE (PART 3 ONLY)
        double[][] xValues = dataset.getX();
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

        testset = new Dataset(xValues, dataset.getY());
        
    }

    /**
     * Builds and trains an artificial neural network
     * @param ann the neural network being built and trained
     * @param file the file of all the setting for the neural network
     * @param rnd random number for reproducibility
     * @throws Exception for any exceptions thrown
     */
    public static void buildTrainNetwork(ANN ann, String file, Random rnd, Dataset train, Dataset dev) throws Exception {
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

        // build and train an ANN with the given data and parameters

        int OUTPUT_DIMENSIONS = 3;
        // building the network
        Layer network = ann.build(train.getInputDims(), OUTPUT_DIMENSIONS, hiddenLayers, hiddenLayerNodes,
                activationFunction);
        Loss crossEntropy = new CrossEntropy();
        Optimizer sGradientDescent = new SGD(network, learningRate);
        // training the network
        ann.train(crossEntropy, sGradientDescent, train, dev, batchSize, epochs, patience, rnd);
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
            if (preprocess) {
                double[][] standardisation = preprocess_trainset();
                preprocess_testset(standardisation, testset);
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

            // Determins which experiment is being performed
            if (args.length == 6) {
                System.out.println("ARGS: " + args[5]);
                if (args[5].equals("1")) {
                    randomHyperParameters(rnd);
                }
                else if (args[5].equals("2")) {
                    featureImportance(rnd, args[3]);
                }
                return;
            }
            // read all parameters from the provided json setting file (see
            // settings/example.json for an example)
            //// YOUR CODE HERE
            ANN ann = new ANN();
            buildTrainNetwork(ann, args[3], rnd, trainset, devset);
            // evaluate the trained ANN on the test set and report results
            try {
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
