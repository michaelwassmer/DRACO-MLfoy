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

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="combined",

parser.add_option("-i", "--inputdirectory", dest="inputDir",default="InputFeatures",


parser.add_option("-n", "--naming", dest="naming",default="_dnn.h5",
        help="file ending for the samples in preprocessing", metavar="naming")

parser.add_option("-v", "--variableselection", dest="variableSelection",default="DL_variables",
        help="FILE for variables used to train DNNs", metavar="variableSelection")

parser.add_option("-l", "--log", dest="log", action = "store_true", default=False,
        help="activate for logarithmic plots", metavar="log")

parser.add_option("-p", "--privatework", dest="privateWork", action = "store_true", default=False,
        help="activate Private Work option", metavar="privateWork")

parser.add_option("-r", "--ratio", dest="ratio", action = "store_true", default=False,
        help="activate ratio plot", metavar="ratio")

parser.add_option("--ratiotitle", dest="ratioTitle", default="#frac{signal}{background}",
        help="STR #frac{PROCESS}{PROCESS}", metavar="title")

parser.add_option("-k", "--ksscore", dest="KSscore", action = "store_true", default=True,
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
    ]

# variables that are not plotted
ignored_variables = [
    "weight",
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
    sampleName      = "ttHbb_2L",
    sampleFile      = data_dir+"/ttHbb_2L.root",
    plotColor       = ROOT.kBlue,
    signalSample    = True)

# add samples
plotter.addSample(
    sampleName      = "ttbar_bb",
    sampleFile      = data_dir+"/ttbar_bb.root",
    plotColor       = ROOT.kRed+3)

plotter.addSample(
    sampleName      = "ttbar_2b",
    sampleFile      = data_dir+"/ttbar_2b.root",
    plotColor       = ROOT.kRed+2)

plotter.addSample(
    sampleName      = "ttbar_b",
    sampleFile      = data_dir+"/ttbar_b.root",
    plotColor       = ROOT.kRed-2)

plotter.addSample(
    sampleName      = "ttbar_cc",
    sampleFile      = data_dir+"/ttbar_cc.root",
    plotColor       = ROOT.kRed+1)

plotter.addSample(
    sampleName      = "ttbar_lf",
    sampleFile      = data_dir+"/ttbar_lf.root",)

# add JT categories
plotter.addCategory("3j_2t")
#plotter.addCategory("3j_3t")
#plotter.addCategory("ge4j_2t")
#plotter.addCategory("ge4j_3t")
#plotter.addCategory("ge4j_ge4t")
#plotter.addCategory("ge4j_ge3t")

# perform plotting routine
plotter.plot(saveKSValues = options.KSscore)
