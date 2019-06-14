import os
import sys
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import optparse
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(filedir)
sys.path.append(basedir)

from evaluationScripts.plotVariables import variablePlotter

usage="usage=%prog [options] \n"
usage+="USE: python plotInputVariables.py -i DIR -o DIR -v FILE  --ksscore --scalesignal=OPTION --lumiscale=FLOAT --ratio --ratiotitel=STR --privatework --log"

parser = optparse.OptionParser(usage=usage)

parser.add_option("-o", "--outputdirectory", dest="outputDir",default="plots_InputFeatures",
        help="DIR for output", metavar="outputDir")

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="InputFeatures",
        help="DIR for input", metavar="inputDir")

parser.add_option("-n", "--naming", dest="naming",default="_dnn.h5",
        help="file ending for the samples in preprocessing", metavar="naming")

parser.add_option("-v", "--variableselection", dest="variableSelection",default="example_variables",
        help="FILE for variables used to train DNNs", metavar="variableSelection")

parser.add_option("-l", "--log", dest="log", action = "store_true", default=False,
        help="activate for logarithmic plots", metavar="log")

parser.add_option("-p", "--privatework", dest="privateWork", action = "store_true", default=False,
        help="activate Private Work option", metavar="privateWork")

parser.add_option("-r", "--ratio", dest="ratio", action = "store_true", default=False,
        help="activate ratio plot", metavar="ratio")

parser.add_option("--ratiotitle", dest="ratioTitle", default="#frac{signal}{background}",
        help="STR #frac{PROCESS}{PROCESS}", metavar="title")

parser.add_option("-k", "--ksscore", dest="KSscore", action = "store_true", default=False,
        help="activate KSscore", metavar="KSscore")

parser.add_option("-s", "--scalesignal", dest="scaleSignal", default=-1,
        help="-1 to scale Signal to background Integral, FLOAT to scale Signal with float value, False to not scale Signal",
        metavar="scaleSignal")

parser.add_option("--lumiscale", dest="lumiScale", default=41.5,
        help="FLOAT to scale Luminosity", metavar="lumiScale")


(options, args) = parser.parse_args()

#import Variable Selection
if not os.path.isabs(options.variableSelection):
    sys.path.append(basedir+"/variable_sets/")
    variable_set = __import__(options.variableSelection)
elif os.path.exists(options.variableSelection):
    variable_set = __import__(options.variableSelection)
else:
    sys.exit("ERROR: Variable Selection File does not exist!")

#get input directory path
if not os.path.isabs(options.inputDir):
    data_dir = basedir+"/workdir/"+options.inputDir
elif os.path.exists(options.inputDir):
    data_dir=options.inputDir
else:
    sys.exit("ERROR: Input Directory does not exist!")

#get output directory path
if not os.path.isabs(options.outputDir):
    plot_dir = basedir+"/workdir/"+options.outputDir
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
else:
    plot_dir=options.outputDir
    if not os.path.exists(options.outputDir):
        os.makedirs(plot_dir)


# plotting options
plotOptions = {
    "ratio":        options.ratio,
    "ratioTitle":   options.ratioTitle,
    "logscale":     options.log,
    "scaleSignal":  float(options.scaleSignal),
    "lumiScale":    float(options.lumiScale),
    "KSscore":      options.KSscore,
    "privateWork":  options.privateWork,
    }
"""
   scaleSignal:
   -1:     scale to background Integral
   float:  scale with float value
   False:  dont scale
"""

# additional variables to plot
additional_variables = [
    #'Evt_CSV_Min_Tagged',
    'BDT_common5_input_transverse_sphericity_jets',
    #'Evt_M_JetsAverage',
    'Evt_CSV_Min',
    'BDT_common5_input_max_dR_jj',
    #'Evt_M_MinDeltaRLeptonTaggedJet',
    #'Evt_JetPtOverJetE',
    #'Evt_M2_TaggedJetsAverage',
    'N_Jets',
    #'Evt_blr_ETH',
    #'Evt_Dr_MinDeltaRTaggedJets',
    #'Evt_Dr_TaggedJetsAverage',
    #'BDT_common5_input_closest_tagged_dijet_mass',
    'Weight_GEN_nom',
    'Evt_HT',
    'Weight_XS',
    'BDT_common5_input_sphericity_tags',
    #'BDT_common5_input_HT_tag',
    #'BDT_common5_input_dev_from_avg_disc_btags',
    #'Evt_CSV_Average_Tagged',
    #'Evt_CSV_Average',
    'Evt_Dr_MinDeltaRLeptonTaggedJet',
    #'N_BTagsT',
    'N_BTagsM',
    ##'Evt_Odd',
    #'Evt_Deta_TaggedJetsAverage',
    'Weight_CSV',
    #'Evt_blr_ETH_transformed',
    #'BDT_common5_input_sphericity_jets',
    'Jet_Pt[2]',
    'Jet_Pt[0]',
    'Jet_Pt[1]',
    #'CSV[1]',
    #'Jet_CSV[0]',
    'Jet_CSV[3]',
    ##'class_label',
    #'memDBp',
    ]

# variables that are not plotted
ignored_variables = [
    "Weight_XS",
    "Weight_GEN_nom",
    ]

# initialize plotter
plotter = variablePlotter(
    output_dir      = plot_dir,
    variable_set    = variable_set,
    add_vars        = additional_variables,
    ignored_vars    = ignored_variables,
    plotOptions     = plotOptions
    )

naming = options.naming
# add signal samples
plotter.addSample(
    sampleName      = "ttH",
    sampleFile      = data_dir+"/ttH"+naming,
    plotColor       = ROOT.kBlue+1,
    signalSample    = True)

# add background samples
plotter.addSample(
    sampleName      = "ttbb",
    sampleFile      = data_dir+"/ttbb"+naming,
    plotColor       = ROOT.kRed+3)

plotter.addSample(
    sampleName      = "tt2b",
    sampleFile      = data_dir+"/tt2b"+naming,
    plotColor       = ROOT.kRed+2)

plotter.addSample(
    sampleName      = "ttb",
    sampleFile      = data_dir+"/ttb"+naming,
    plotColor       = ROOT.kRed-2)

plotter.addSample(
    sampleName      = "ttcc",
    sampleFile      = data_dir+"/ttcc"+naming,
    plotColor       = ROOT.kRed+1)

plotter.addSample(
    sampleName      = "ttlf",
    sampleFile      = data_dir+"/ttlf"+naming)



# add JT categories
plotter.addCategory("ge6j_ge3t")


# perform plotting routine
plotter.plot(saveKSValues = options.KSscore)
