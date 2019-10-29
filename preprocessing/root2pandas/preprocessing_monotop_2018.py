import os
import sys
import optparse
# local imports
filedir = os.path.dirname(os.path.realpath(__file__))
basedir = os.path.dirname(os.path.dirname(filedir))
sys.path.append(basedir)

import root2pandas

"""
USE: python preprocessing.py --outputdirectory=DIR --variableSelection=FILE --maxentries=INT --MEM=BOOL
"""
usage="usage=%prog [options] \n"
usage+="USE: python preprocessing.py --outputdirectory=DIR --variableselection=FILE --maxentries=INT --MEM=BOOL --name=STR\n"
usage+="OR: python preprocessing.py -o DIR -v FILE -e INT -m BOOL -n STR"

parser = optparse.OptionParser(usage=usage)

parser.add_option("-o", "--outputdirectory", dest="outputDir",default="InputFeatures",
        help="DIR for output", metavar="outputDir")

parser.add_option("-v", "--variableselection", dest="variableSelection",default="example_variables",
        help="FILE for variables used to train DNNs", metavar="variableSelection")

parser.add_option("-e", "--maxentries", dest="maxEntries", default=50000,
        help="INT used for maximal number of entries for each batch (to restrict memory usage)", metavar="maxEntries")

parser.add_option("-m", "--MEM", dest="MEM", action = "store_true", default=False,
        help="BOOL to use MEM or not", metavar="MEM")

parser.add_option("-n", "--name", dest="Name", default="dnn",
        help="STR of the output file name", metavar="Name")


(options, args) = parser.parse_args()

if not os.path.isabs(options.variableSelection):
    sys.path.append(basedir+"/variable_sets/")
    variable_set = __import__(options.variableSelection)
elif os.path.exists(options.variableSelection):
    variable_set = __import__(options.variableSelection)
else:
    sys.exit("ERROR: Variable Selection File does not exist!")

if not os.path.isabs(options.outputDir):
    outputdir = basedir+"/workdir/"+options.outputDir
elif os.path.exists(options.outputDir) or os.path.exists(os.path.dirname(options.outputDir)):
    outputdir=options.outputDir
else:
    sys.exit("ERROR: Output Directory does not exist!")

# define a base event selection which is applied for all Samples
# select only events with GEN weight > 0 because training with negative weights is weird
base = "(Evt_Pt_MET >= 150. and N_LoosePhotons==0 and N_HEM_Jets==0)"

# single lepton selections
single_mu_sel = "(N_LooseElectrons == 0 and N_LooseMuons==1 and N_TightMuons == 1 and Triggered_HLT_IsoMu24_vX==1)"
single_el_sel = "(N_LooseMuons == 0 and N_LooseElectrons==1 and N_TightElectrons == 1 and (Triggered_HLT_Ele28_eta2p1_WPTight_Gsf_HT150_vX==1 or Triggered_HLT_Ele32_WPTight_Gsf_vX==1))"

base_selection = "("+base+" and ("+single_mu_sel+" or "+single_el_sel+"))"

ttH_selection = "(Evt_Odd == 1)"

# define output classes
monotop_categories = root2pandas.EventCategories()
monotop_categories.addCategory("monotop", selection = None)

ttbar_categories = root2pandas.EventCategories()
ttbar_categories.addCategory("ttbar", selection = None)

wjets_categories = root2pandas.EventCategories()
wjets_categories.addCategory("wjets", selection = None)

# initialize dataset class
dataset = root2pandas.Dataset(
    outputdir   = outputdir,
    naming      = options.Name,
    addMEM      = options.MEM,
    maxEntries  = options.maxEntries)

# add base event selection
dataset.addBaseSelection(base_selection)



ntuplesPath = "/nfs/dust/cms/user/mwassmer/MonoTop/ntuples_2018_new_skims/"
memPath = "bla"

# add samples to dataset
dataset.addSample(
    sampleName  = "monotop",
    ntuples     = ntuplesPath+"/VectorMonotop_*/*nominal*.root",
    categories  = monotop_categories,
    selections  = None,
    MEMs        = "",
    even_odd    = True
   ) 

dataset.addSample(
    sampleName  = "ttbar",
    ntuples     = ntuplesPath+"/TTTo*/*nominal*.root",
    categories  = ttbar_categories,
    selections  = None,
    MEMs        = "",
    even_odd    = True
      )

dataset.addSample(
    sampleName  = "wjets",
    ntuples     = ntuplesPath+"/WJetsToLNu*/*nominal*.root",
    categories  = wjets_categories,
    selections  = None,
    MEMs        = "",
    even_odd    = True
      )
# initialize variable list 
dataset.addVariables(variable_set.all_variables)

# define an additional variable list
additional_variables = [
    "N_Jets",
    "N_BTagsM",
    "Weight_XS",
    "Weight_CSV",
    "Weight_GEN_nom",
    "Evt_ID", 
    "Evt_Run", 
    "Evt_Lumi"]

# add these variables to the variable list
dataset.addVariables(additional_variables)

# run the preprocessing
dataset.runPreprocessing()
