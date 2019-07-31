import os
import sys
import numpy as np
import json
from functools import partial, update_wrapper

# local imports
filedir  = os.path.dirname(os.path.realpath(__file__))
DRACOdir = os.path.dirname(filedir)
basedir  = os.path.dirname(DRACOdir)
sys.path.append(basedir)

# import with ROOT
from pyrootsOfTheCaribbean.evaluationScripts import plottingScripts


# imports with keras
import utils.generateJTcut as JTcut
import data_frame

import keras
import keras.optimizers as optimizers
import keras.models as models
import keras.layers as layer
from keras import backend as K
import pandas as pd

# Limit gpu usage
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.tensorflow_backend.set_session(tf.Session(config=config))




class EarlyStopping(keras.callbacks.Callback):
    ''' custom implementation of early stopping
        with options for
            - stopping when val/train loss difference exceeds a percentage threshold
            - stopping when val loss hasnt increased for a set number of epochs '''

    def __init__(self, monitor = "loss", value = None, min_epochs = 20, stopping_epochs = None, patience = 10, verbose = 0):
        super(keras.callbacks.Callback, self).__init__()
        self.val_monitor = "val_"+monitor
        self.train_monitor = monitor
        self.patience = patience
        self.n_failed = 0

        self.stopping_epochs = stopping_epochs
        self.best_epoch = 0
        self.best_validation = 999.
        self.min_epochs = min_epochs
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs = {}):
        current_val = logs.get(self.val_monitor)
        if epoch == 0:
            self.best_validation = current_val
        current_train = logs.get(self.train_monitor)

        if current_val is None or current_train is None:
            warnings.warn("Early stopping requires {} and {} available".format(
                self.val_monitor, self.train_monitor), RuntimeWarning)

        if current_val < self.best_validation:
            self.best_validation = current_val
            self.best_epoch = epoch

        # check loss by percentage difference
        if self.value:
            if (current_val-current_train)/(abs(current_train)) > self.value and epoch > self.min_epochs:
                if self.verbose > 0:
                    print("\nEpoch {}: early stopping threshold reached".format(epoch))
                self.n_failed += 1
                if self.n_failed > self.patience:
                    self.model.stop_training = True

        # check loss by validation performance increase
        if self.stopping_epochs:
            if self.best_epoch + self.stopping_epochs < epoch and epoch > self.min_epochs:
                if self.verbose > 0:
                    print("\nValidation loss has not decreased for {} epochs".format( epoch - self.best_epoch ))
                self.model.stop_training = True



class DNN():
    def __init__(self,
            save_path,
            input_samples,
            event_category,
            train_variables,
            weight_variables,
            train_epochs    = 500,
            test_percentage = 0.2,
            eval_metrics    = None,
            shuffle_seed    = None,
            balanceSamples  = False,
            evenSel         = None,
            useSystematics  = None):

        # save some information
        # list of samples to load into dataframe
        self.input_samples = input_samples

        # output directory for results
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs( self.save_path )

        # name of event category (usually nJet/nTag category)
        self.JTstring       = event_category
        self.event_category = JTcut.getJTstring(event_category)
        self.categoryLabel  = JTcut.getJTlabel(event_category)
        # selection
        self.evenSel = ""
        self.oddSel = "1."
        if not evenSel == None:
            if evenSel == True:
                self.evenSel = "(Evt_Odd==0)"
                self.oddSel  = "(Evt_Odd==1)"
            elif evenSel == False:
                self.evenSel = "(Evt_Odd==1)"
                self.oddSel  = "(Evt_Odd==0)"

        # list of input variables
        self.train_variables = train_variables
        self.weight_variables = weight_variables
        self.useSystematics = useSystematics

        # percentage of events saved for testing
        self.test_percentage = test_percentage

        # number of train epochs
        self.train_epochs = train_epochs

        # additional metrics for evaluation of the training process
        self.eval_metrics = eval_metrics

        # load data set
        self.data = self._load_datasets(shuffle_seed, balanceSamples)
        self.event_classes = self.data.output_classes

        # save variable norm
        self.cp_path = self.save_path+"/checkpoints/"
        if not os.path.exists(self.cp_path):
            os.makedirs(self.cp_path)
        out_file = self.cp_path + "/variable_norm.csv"
        self.data.norm_csv.to_csv(out_file)
        print("saved variabe norms at "+str(out_file))

        # make plotdir
        self.plot_path = self.save_path+"/plots/"
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)

        # layer names for in and output (needed for c++ implementation)
        self.inputName = "inputLayer"
        self.outputName = "outputLayer"



    def _load_datasets(self, shuffle_seed, balanceSamples):
        ''' load data set '''
        return data_frame.DataFrame(
            input_samples       = self.input_samples,
            event_category      = self.event_category,
            train_variables     = self.train_variables,
            weight_variables    = self.weight_variables,
            test_percentage     = self.test_percentage,
            shuffleSeed         = shuffle_seed,
            balanceSamples      = balanceSamples,
            evenSel             = self.evenSel)


    def _load_architecture(self, config):
        ''' load the architecture configs '''
        # defnie default network configuration
        self.architecture = {
            "layers":                   [200],
            "loss_function":            "categorical_crossentropy",
            "Dropout":                  0.2,
            "L2_Norm":                  1e-5,
            "batch_size":               5000,
            "optimizer":                optimizers.Adagrad(decay=0.99),
            "activation_function":      "elu",
            "output_activation":        "Softmax",
            "earlystopping_percentage": None,
            "earlystopping_epochs":     None,
            }

        for key in config:
            self.architecture[key] = config[key]

    def load_trained_model(self, inputDirectory):
        ''' load an already trained model '''
        checkpoint_path = inputDirectory+"/checkpoints/trained_model.h5py"

        # get the model
        self.model = keras.models.load_model(checkpoint_path)
        self.model.summary()

        # evaluate test dataset
        self.model_eval = self.model.evaluate(
            self.data.get_test_data(as_matrix = True),
            self.data.get_test_labels())

        # save predicitons
        self.model_prediction_vector = self.model.predict(
            self.data.get_test_data(as_matrix = True) )
        self.model_train_prediction  = self.model.predict(
            self.data.get_train_data(as_matrix = True) )

        # save predicted classes with argmax
        self.predicted_classes = np.argmax( self.model_prediction_vector, axis = 1)

        # save confusion matrix
        from sklearn.metrics import confusion_matrix
        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # print evaluations
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector)
        print("\nROC-AUC score: {}".format(self.roc_auc_score))


    def predict_event_query(self, query ):
        events = self.data.get_full_df().query( query )
        print(str(events.shape[0]) + " events matched the query '"+str(query)+"'.")

        for index, row in events.iterrows():
            print("========== DNN output ==========")
            print("Event: "+str(index))
            for var in row.values:
                print(var)
            print("-------------------->")
            output = self.model.predict( np.array([list(row.values)]) )[0]
            for i, node in enumerate(self.event_classes):
                print(str(node)+" node: "+str(output[i]))
            print("-------------------->")




    def build_default_model(self):
        ''' build default straight forward DNN from architecture dictionary '''

        # infer number of input neurons from number of train variables
        number_of_input_neurons     = self.data.n_input_neurons

        # get all the architecture settings needed to build model
        number_of_neurons_per_layer = self.architecture["layers"]
        dropout                     = self.architecture["Dropout"]
        activation_function         = self.architecture["activation_function"]
        if activation_function == "leakyrelu":
            activation_function = "linear"
        l2_regularization_beta      = self.architecture["L2_Norm"]
        output_activation           = self.architecture["output_activation"]

        # define input layer
        Inputs = keras.layers.Input(
            shape = (number_of_input_neurons,),
            name  = self.inputName)

        X = Inputs
        self.layer_list = [X]
        if self.useSystematics: Weights = keras.layers.Input(shape=(3,), name="CSV_weights")

        # loop over dense layers
        for iLayer, nNeurons in enumerate(number_of_neurons_per_layer):
            X = keras.layers.Dense(
                units               = nNeurons,
                activation          = activation_function,
                kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
                name                = "DenseLayer_"+str(iLayer)
                )(X)

            if self.architecture["activation_function"] == "leakyrelu":
                X = keras.layers.LeakyReLU(alpha=0.3)(X)

            # add dropout percentage to layer if activated
            if not dropout == 0:
                X = keras.layers.Dropout(dropout, name = "DropoutLayer_"+str(iLayer))(X)

        # generate output layer
        X = keras.layers.Dense(
            units               = self.data.n_output_neurons,
            activation          = output_activation.lower(),
            kernel_regularizer  = keras.regularizers.l2(l2_regularization_beta),
            name                = self.outputName
            )(X)

        # define model
        if self.useSystematics: model = models.Model(inputs = [Inputs, Weights], outputs = [X])
        else: model = models.Model(inputs = [Inputs], outputs = [X])

        # Loss for Systematics
        def histLoss(y_true, y_pred):
            return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) + WeightLoss(y_true, y_pred, Weights)

        def WeightLoss(y_true, y_pred, CSV_weights):
            #value_range = [0.0, 1.0]
            lam = 20.
            #histBins = 5
            nom_weights, weightsUp, weightsDown = tf.split(CSV_weights, num_or_size_splits=3, axis=1)
            #weightsUp = tf.Print(weightsUp, [weightsUp, weightsDown])

            def gauss_filter(x, mean, width):
                return K.exp(-1.0 * (x - mean)**2 / 2.0 / width**2)

            def count_bin(mean, width):
                f = gauss_filter(y_pred, mean, width)
                #f = tf.Print(f, [f, nom_weights])
                f = f*nom_weights
                #f = tf.Print(f, [f, weightsUp, weightsDown])
                return K.sum(f), K.sum(f*weightsUp), K.sum(f*weightsDown)

            bins = np.linspace(0, 1, 11)
            width = bins[1] - bins[0]
            mids = bins[:-1] + 0.5 * width
            l = 0.0
            for mean in mids:
                nominal, shiftedUp, shiftedDown = count_bin(mean, width)
                #nominal = tf.Print(nominal, [nominal, shiftedUp, shiftedDown])
                l += K.square((nominal - shiftedUp) / K.clip(nominal, K.epsilon(), None)) + K.square((nominal - shiftedDown) / K.clip(nominal, K.epsilon(), None))
            l /= K.cast(len(mids), K.floatx())

            #hist = tf.to_float(tf.histogram_fixed_width(y_pred*nom_weights, value_range, nbins=histBins))
            #histUp = tf.to_float(tf.histogram_fixed_width(y_pred*nom_weights*weightsUp, value_range, nbins=histBins))
            #histDown = tf.to_float(tf.histogram_fixed_width(y_pred*nom_weights*weightsDown, value_range, nbins=histBins))
            #hist=tf.Print(hist, [hist, histUp, histDown], "Up, Down: ", summarize=10)

            #batchSize = tf.to_float(K.sum(y_true) + K.sum(1-y_true))
            #L = K.square((hist-histUp)/hist) + K.square((hist-histDown)/hist)
            #L = tf.Print(L, [L], summarize=10)
            #L = lam * K.sum(L)/batchSize
            #L = tf.Print(L, [L])
            return lam*l

        def wrapped_partial(func, *args, **kwargs):
            partial_func = partial(func, *args, **kwargs)
            update_wrapper(partial_func, func)
            return partial_func

        WeightLoss_ = wrapped_partial(WeightLoss, CSV_weights=Weights)

        model.summary()

        if self.useSystematics: return model, histLoss, WeightLoss_
        else: return model

    def build_model(self, config = None, model = None):
        ''' build a DNN model
            use options defined in 'config' dictionary '''

        if config:
            self._load_architecture(config)
            print("loading non default net configs")

        if model == None:
            print("building model from config")
            if self.useSystematics: model, histLoss, WeightLoss_ = self.build_default_model()
            else: model = self.build_default_model()



        # compile the model
        used_loss = histLoss if self.useSystematics else self.architecture["loss_function"]
        used_metrics = ["acc", WeightLoss_] if self.useSystematics else self.eval_metrics
        print used_loss
        model.compile(
            loss        = used_loss,
            optimizer   = self.architecture["optimizer"],
            metrics     = used_metrics)

        # save the model
        self.model = model

        # save net information
        out_file    = self.save_path+"/model_summary.yml"
        yml_model   = self.model.to_yaml()
        with open(out_file, "w") as f:
            f.write(yml_model)

    def train_model(self):
        ''' train the model '''

        # add early stopping if activated
        callbacks = None
        if self.architecture["earlystopping_percentage"] or self.architecture["earlystopping_epochs"]:
            callbacks = [EarlyStopping(
                monitor         = "loss",
                value           = self.architecture["earlystopping_percentage"],
                min_epochs      = 50,
                stopping_epochs = self.architecture["earlystopping_epochs"],
                verbose         = 1)]

        # train main net
        x_data = [self.data.get_train_data(as_matrix = True), self.data.get_trainCSV_weights()] if self.useSystematics else self.data.get_train_data(as_matrix = True)
        self.trained_model = self.model.fit(
            x = x_data,
            y = self.data.get_train_labels(),
            batch_size          = self.architecture["batch_size"],
            epochs              = self.train_epochs,
            shuffle             = True,
            callbacks           = callbacks,
            validation_split    = 0.25,
            sample_weight       = self.data.get_train_weights()
            )

    def save_model(self, argv, execute_dir):
        ''' save the trained model '''

        # save executed command
        argv[0] = execute_dir+"/"+argv[0].split("/")[-1]
        execute_string = "python "+" ".join(argv)
        out_file = self.cp_path+"/command.sh"
        with open(out_file, "w") as f:
            f.write(execute_string)
        print("saved executed command to {}".format(out_file))

        # save model as h5py file
        out_file = self.cp_path + "/trained_model.h5py"
        self.model.save(out_file)
        print("saved trained model at "+str(out_file))

        # save config of model
        model_config = self.model.get_config()
        out_file = self.cp_path +"/trained_model_config"
        with open(out_file, "w") as f:
            f.write( str(model_config))
        print("saved model config at "+str(out_file))

        # save weights of network
        out_file = self.cp_path +"/trained_model_weights.h5"
        self.model.save_weights(out_file)
        print("wrote trained weights to "+str(out_file))

        # set model as non trainable
        for layer in self.model.layers:
            layer.trainable = False
        self.model.trainable = False

        # save checkpoint files (needed for c++ implementation)
        out_file = self.cp_path + "/trained_model"
        saver = tf.train.Saver()
        sess = K.get_session()
        save_path = saver.save(sess, out_file)
        print("saved checkpoint files to "+str(out_file))

        # produce json file with configs
        configs = self.architecture
        configs["inputName"] = self.inputName
        configs["outputName"] = self.outputName+"/"+configs["output_activation"]
        configs = {key: configs[key] for key in configs if not "optimizer" in key}

        # more information saving
        configs["inputData"] = self.input_samples.input_path
        configs["eventClasses"] = self.input_samples.getClassConfig()
        configs["JetTagCategory"] = self.JTstring
        configs["categoryLabel"] = self.categoryLabel
        configs["Selection"] = self.event_category
        configs["trainEpochs"] = self.train_epochs
        configs["trainVariables"] = self.train_variables
        configs["shuffleSeed"] = self.data.shuffleSeed
        configs["trainSelection"] = self.evenSel
        configs["evalSelection"] = self.oddSel

        # save information for binary DNN
        if self.data.binary_classification:
            configs["binaryConfig"] = {
                "minValue": self.input_samples.bkg_target,
                "maxValue": 1.}

        json_file = self.cp_path + "/net_config.json"
        with open(json_file, "w") as jf:
            json.dump(configs, jf, indent = 2, separators = (",", ": "))
        print("wrote net configs to "+str(json_file))

        # save configurations of variables for plotscript
        plot_file = self.cp_path+"/plot_config.csv"
        variable_configs = pd.read_csv(basedir+"/pyrootsOfTheCaribbean/plot_configs/variableConfig.csv").set_index("variablename", drop = True)
        variables = variable_configs.loc[self.train_variables]
        variables.to_csv(plot_file, sep = ",")
        print("wrote config of input variables to {}".format(plot_file))


    def eval_model(self):
        ''' evaluate trained model '''

        # evaluate test dataset
        eval_test = [self.data.get_test_data(as_matrix = True), self.data.get_testCSV_weights()] if self.useSystematics else self.data.get_test_data(as_matrix = True)
        self.model_eval = self.model.evaluate(
            eval_test,
            self.data.get_test_labels()
            #batch_size = 4111
            )

        # save history of eval metrics
        self.model_history = self.trained_model.history

        # save predicitons
        test_pred = [self.data.get_test_data(as_matrix = True), self.data.get_testCSV_weights()] if self.useSystematics else self.data.get_test_data(as_matrix = True)
        self.model_prediction_vector = self.model.predict(
             test_pred)
        train_pred = [self.data.get_train_data(as_matrix = True), self.data.get_trainCSV_weights()] if self.useSystematics else self.data.get_train_data(as_matrix = True)
        self.model_train_prediction  = self.model.predict(
             train_pred)

        #figure out ranges
        self.get_ranges()

        # save predicted classes with argmax
        self.predicted_classes = np.argmax( self.model_prediction_vector, axis = 1)

        # save confusion matrix
        from sklearn.metrics import confusion_matrix
        self.confusion_matrix = confusion_matrix(
            self.data.get_test_labels(as_categorical = False), self.predicted_classes)

        # print evaluations
        from sklearn.metrics import roc_auc_score
        self.roc_auc_score = roc_auc_score(self.data.get_test_labels(), self.model_prediction_vector)
        print("\nROC-AUC score: {}".format(self.roc_auc_score))

        if self.eval_metrics:
            print("model test loss: {}".format(self.model_eval[0]))
            for im, metric in enumerate(self.eval_metrics):
                print("model test {}: {}".format(metric, self.model_eval[im+1]))


    def get_ranges(self):
        if not self.data.binary_classification:
            max_ = [0.]*len(self.input_samples.samples)
            for ev in self.model_prediction_vector:
                for i,node in enumerate(ev):
                    if node>max_[i]:
                        max_[i]=node
            print("Max: ",max_)
            for i, sample in enumerate(self.input_samples.samples):
                sample.max=round(float(max_[i]),2)
                sample.min=round(float(1./len(self.input_samples.samples)),2)



    def get_input_weights(self):
        ''' get the weights of the input layer and sort input variables by weight sum '''

        # get weights
        first_layer = self.model.layers[1]
        weights = first_layer.get_weights()[0]

        self.weight_dict = {}
        for out_weights, variable in zip(weights, self.train_variables):
            w_sum = np.sum(np.abs(out_weights))
            self.weight_dict[variable] = w_sum

        # sort weight dict
        rank_path = self.save_path + "/absolute_weight_sum.csv"
        with open(rank_path, "w") as f:
            f.write("variable,weight_sum\n")
            for key, val in sorted(self.weight_dict.iteritems(), key = lambda (k,v): (v,k)):
                print("{:50s}: {}".format(key, val))
                f.write("{},{}\n".format(key,val))
        print("wrote weight ranking to "+str(rank_path))




    # --------------------------------------------------------------------
    # result plotting functions
    # --------------------------------------------------------------------
    def plot_metrics(self, privateWork = False):
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)

        ''' plot history of loss function and evaluation metrics '''
        metrics = ["loss"]
        if self.eval_metrics: metrics += self.eval_metrics

        # loop over metrics and generate matplotlib plot
        for metric in metrics:
            plt.clf()
            # get history of train and validation scores
            train_history = self.model_history[metric]
            val_history = self.model_history["val_"+metric]

            n_epochs = len(train_history)
            plot_epoch_cut = 0                                    # added
            epochs = np.arange(1+plot_epoch_cut,n_epochs+1,1)     #changed

            # plot histories
            plt.plot(epochs, train_history[plot_epoch_cut:], "b-", label = "train", lw = 2)           #changed
            plt.plot(epochs, val_history[plot_epoch_cut:], "r-", label = "validation", lw = 2)         #changed
            if privateWork:
                plt.title("CMS private work", loc = "left", fontsize = 16)

            # add title
            title = self.categoryLabel
            title = title.replace("\\geq", "$\geq$")
            title = title.replace("\\leq", "$\leq$")
            plt.title(title, loc = "right", fontsize = 16)

            # make it nicer
            plt.grid()
            plt.xlabel("epoch", fontsize = 16)
            plt.ylabel(metric.replace("_"," "), fontsize = 16)

            # add legend
            plt.legend()

            # save
            out_path = self.save_path + "/model_history_"+str(metric)+".pdf"
            plt.savefig(out_path)
            print("saved plot of "+str(metric)+" at "+str(out_path))




    def plot_outputNodes(self, log = False, printROC = False, signal_class = None,
                        privateWork = False,
                        nbins = 20, bin_range = [0.,1.]):

        ''' plot distribution in outputNodes '''
        plotNodes = plottingScripts.plotOutputNodes(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            signal_class        = signal_class,
            event_category      = self.categoryLabel,
            plotdir             = self.plot_path,
            logscale            = log)

        plotNodes.plot(ratio = False, printROC = printROC, privateWork = privateWork)


    def plot_discriminators(self, log = False, printROC = False, privateWork = False,
                        signal_class = None, nbins = 18, bin_range = [0.1,1.]):

        ''' plot all events classified as one category '''
        plotDiscrs = plottingScripts.plotDiscriminators(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            signal_class        = signal_class,
            event_category      = self.categoryLabel,
            plotdir             = self.plot_path,
            logscale            = log)

        plotDiscrs.plot(ratio = False, printROC = printROC, privateWork = privateWork)


    def plot_confusionMatrix(self, norm_matrix = True, privateWork = False, printROC = False):
        ''' plot confusion matrix '''
        plotCM = plottingScripts.plotConfusionMatrix(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            event_category      = self.categoryLabel,
            plotdir             = self.save_path)

        plotCM.plot(norm_matrix = norm_matrix, privateWork = privateWork, printROC = printROC)

    def plot_closureTest(self, log = False, privateWork = False,
                        signal_class = None, nbins = 20, bin_range = [0.,1.]):
        ''' plot comparison between train and test samples '''

        bin_range = [1./self.data.n_output_neurons, 1.]
        closureTest = plottingScripts.plotClosureTest(
            data                = self.data,
            test_prediction     = self.model_prediction_vector,
            train_prediction    = self.model_train_prediction,
            event_classes       = self.event_classes,
            nbins               = nbins,
            bin_range           = bin_range,
            signal_class        = signal_class,
            event_category      = self.categoryLabel,
            plotdir             = self.plot_path,
            logscale            = log)

        closureTest.plot(ratio = False, privateWork = privateWork)

    def plot_eventYields(self, log = False, privateWork = False, signal_class = None):
        eventYields = plottingScripts.plotEventYields(
            data                = self.data,
            prediction_vector   = self.model_prediction_vector,
            event_classes       = self.event_classes,
            event_category      = self.categoryLabel,
            signal_class        = signal_class,
            plotdir             = self.save_path,
            logscale            = log)

        eventYields.plot(privateWork = privateWork)

    def plot_binaryOutput(self, log = False, privateWork = False, printROC = False,
                        nbins = 30, bin_range = [0.,1.], name = "binary discriminator"):

        binaryOutput = plottingScripts.plotBinaryOutput(
            data                = self.data,
            predictions         = self.model_prediction_vector,
            nbins               = nbins,
            bin_range           = bin_range,
            event_category      = self.categoryLabel,
            plotdir             = self.save_path,
            logscale            = log)

        binaryOutput.plot(ratio = False, printROC = printROC, privateWork = privateWork, name = name)

def loadDNN(inputDirectory, outputDirectory):

    # get net config json
    configFile = inputDirectory+"/checkpoints/net_config.json"
    if not os.path.exists(configFile):
        sys.exit("config needed to load trained DNN not found\n{}".format(configFile))

    with open(configFile) as f:
        config = f.read()
    config = json.loads(config)

    # load samples
    input_samples = data_frame.InputSamples(config["inputData"])

    for sample in config["eventClasses"]:
        input_samples.addSample(sample["samplePath"], sample["sampleLabel"], normalization_weight = sample["sampleWeight"])

    print("shuffle seed: {}".format(config["shuffleSeed"]))
    # init DNN class
    dnn = DNN(
        save_path       = outputDirectory,
        input_samples   = input_samples,
        event_category  = config["JetTagCategory"],
        train_variables = config["trainVariables"],
        shuffle_seed    = config["shuffleSeed"]
        )

    # load the trained model
    dnn.load_trained_model(inputDirectory)


    return dnn
