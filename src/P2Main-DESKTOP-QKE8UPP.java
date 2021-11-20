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

    public static void printUsage(){
        System.out.println("Input not recognised. Usage is:");
        System.out.println("\tjava -cp lib/jblas-1.2.5.jar:minet:. P2Main <train_data> <test_data> <random_seed> <json_setting_file> [<apply_preprocessing>]");
        System.out.println("\nExample 1: build an ANN for Part1 and Part2 (no preprocessing is applied)");
        System.out.println("\tjava -cp lib/jblas-1.2.5.jar:minet:. data/Part1/train.txt data/Part1/test.txt 123 settings/example.json");
        System.out.println("Example 2: build an ANN for Part3 (preprocessing is applied)");
        System.out.println("\tjava -cp lib/jblas-1.2.5.jar:minet:. data/Part3/train.txt data/Part3/test.txt 123 settings/example.json 1");
        System.out.println("Example 2: build an ANN for Part3 (preprocessing is not applied)");
        System.out.println("\tjava -cp lib/jblas-1.2.5.jar:minet:. data/Part3/train.txt data/Part3/test.txt 123 settings/example.json");
    }


    /** 
     * apply data preprocessing (imputation of missing values and standardisation) on trainset (Part 3 only)     
    */
    public static void preprocess_trainset(){        
        //// YOUR CODE HERE (PART 3 ONLY) 
    }

    /** 
     * apply data preprocessing (imputation of missing values and standardisation) on testset (Part 3 only)     
    */
    public static void preprocess_testset(){        
        //// YOUR CODE HERE (PART 3 ONLY)
    }

    public static void main(String[] args){        
        if (args.length < 4){
            printUsage();
            return;
        }        

        // set jblas random seed (for reproducibility)		
		org.jblas.util.Random.seed(Integer.parseInt(args[2]));
		Random rnd = new Random(Integer.parseInt(args[2]));
		
        // turn off jblas info messages
        Logger.getLogger().setLevel(Logger.WARNING);

		// load train and test data into trainset and testset
		System.out.println("Loading data...");              
		//// YOUR CODE HERE
        Dataset trainset = Dataset.loadTxt(args[1])
        Dataset testset = Dataset.loadTxt(args[2])
        
        // check whether data-preprocessing is applied (Part 3)
        boolean preprocess = false;
        if (args.length==5){
            if (!args[4].equals("0") && !args[4].equals("1")){
                System.out.println("HERE" + args[4]);
                printUsage();
                System.out.println("\nError: <apply_preprocessing> must be either empty (off), or 0 (off) or 1 (on)");
                return;
            }            
            if (args[4].equals("1"))
                 preprocess = true;   
        }
        
        // apply data-processing on trainset
        if (preprocess)
            preprocess_trainset();

        // split train set into train set (trainset) and validation set, also called development set (devset)
        // suggested split ratio: 80/20        
        trainset.shuffle(rnd); // shuffle the train data before we split. NOTE: this line was updated on Nov 11th.
        //// YOUR CODE HERE
        int length = trainset.getSize();
        int devLength = (int) Math.floor(length * 0.2);
        double[][] devSetX = Arrays.copyOfRange(trainset.getX(), 1, devLength);
        double[][] devSetY = Arrays.copyOfRange(trainset.getY(), 1, devLength);

        double[][] trainingSetX = Arrays.copyOfRange(trainset.getX(), devLength, length);
        double[][] trainingSetY = Arrays.copyOfRange(trainset.getY(), devLength, length);

        Dataset devset = new Dataset(devSetX, devSetY);
        trainset = new Dataset(trainingSetX, trainingSetY);

        // read all parameters from the provided json setting file (see settings/example.json for an example)
        //// YOUR CODE HERE
        JSONParser parser = new JSONParser();
        int hiddenLayers;
        int hiddenLayerNodes;
        String activationFunction;
        double learningRate;
        int batchSize;
        int epochs;
        int patience;
        try {
            Object obj = parser.parse(new FileReader(args[4]));
            JSONObject jsonObject = (JSONObject) obj;
            hiddenLayers = (int) jsonObject.get("n_hidden_layers");
            hiddenLayerNodes = (int) jsonObject.get("n_nodes_per_hidden_layer");
            activationFunction = (String) jsonObject.get("activation_function");
            learningRate = (int) jsonObject.get("learning_rate");
            batchSize = (int) jsonObject.get("batchsize");
            epochs = (int) jsonObject.get("nEpochs");
            patience = (int) jsonObject.get("patience");
        } catch (Exception e) {
            e.printStackTrace();
        }
		
		// build and train an ANN with the given data and parameters
        ANN ann = new ANN();
        //// YOUR CODE HERE        
        Layer layer = ann.build(trainset.getInputDims(), trainset.getOutDims(), hiddenLayers, hiddenLayerNodes, activationFunction);
        Loss crossEntropy = new CrossEntropy();
        Optimizer sGradientDescent = new SGD(layer, learningRate);
        ann.train(crossEntropy, sGradientDescent, trainset, devset, batchSize, epochs, patience, rnd);
        // evaluate the trained ANN on the test set and report results	
        try{
            // apply data-preprocessing on testset        
            if (preprocess)
                preprocess_testset();
		    double testAcc = ann.eval(testset);
		    System.out.println("accuracy on test set: " + testAcc);
        } catch(Exception e){
            System.out.println(e.getMessage());
            return;
        }
	}
}