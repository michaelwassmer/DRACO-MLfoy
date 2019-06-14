variables = {}
variables["4j_ge3t"] = [
    "Jet_CSV[3]",
    "CSV[1]",
    "Evt_M_MinDeltaRLeptonTaggedJet",
    "N_BTagsT",
    "Jet_Pt[0]",
    "BDT_common5_input_dev_from_avg_disc_btags",
    "BDT_common5_input_closest_tagged_dijet_mass",
    "BDT_common5_input_sphericity_jets",
    "Evt_blr_ETH",
    "memDBp",
    "Evt_Dr_MinDeltaRTaggedJets",
    "BDT_common5_input_transverse_sphericity_jets",
    "Evt_CSV_Min_Tagged",
    "Evt_CSV_Average_Tagged",
    "Evt_CSV_Average",
    "Evt_blr_ETH_transformed",
    "Evt_CSV_Min",
    "BDT_common5_input_HT_tag",
    "Evt_M_JetsAverage",
    "Evt_M2_TaggedJetsAverage",
    ]



variables["5j_ge3t"] = [
    "BDT_common5_input_max_dR_jj",
    "CSV[1]",
    "Jet_Pt[1]",
    "N_BTagsT",
    "Evt_Dr_MinDeltaRLeptonTaggedJet",
    "BDT_common5_input_sphericity_jets",
    "Jet_Pt[2]",
    "BDT_common5_input_closest_tagged_dijet_mass",
    "BDT_common5_input_sphericity_tags",
    "Evt_blr_ETH",
    "Jet_Pt[0]",
    "BDT_common5_input_HT_tag",
    "Evt_Dr_TaggedJetsAverage",
    "Evt_HT",
    "memDBp",
    "Evt_CSV_Average_Tagged",
    "Evt_CSV_Min_Tagged",
    "Evt_CSV_Average",
    "Evt_M_JetsAverage",
    "Evt_blr_ETH_transformed",
    ]

tmp = list(set(variables["5j_ge3t"] + variables["4j_ge3t"]))
variables["le5j_ge3t"] = tmp[:]
variables["le5j_ge4t"] = tmp[:]

if "memDBp" in tmp: tmp.pop(tmp.index("memDBp"))
variables["le5j_3t"] = tmp




variables["ge6j_ge3t"] = [
    "Evt_JetPtOverJetE",
    "Evt_Dr_MinDeltaRTaggedJets",
    "Evt_Dr_TaggedJetsAverage",
    "Jet_CSV[0]",
    "BDT_common5_input_closest_tagged_dijet_mass",
    "Evt_M_MinDeltaRLeptonTaggedJet",
    "CSV[1]",
    "BDT_common5_input_sphericity_jets",
    "BDT_common5_input_dev_from_avg_disc_btags",
    "N_BTagsT",
    "Evt_blr_ETH",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_M2_TaggedJetsAverage",
    "Evt_CSV_Average",
    "Evt_CSV_Average_Tagged",
    "Evt_CSV_Min_Tagged",
    "BDT_common5_input_HT_tag",
    "Evt_M_JetsAverage",
    "memDBp",
    "Evt_blr_ETH_transformed",
    ]

variables["ge6j_ge3t_all"] = [
    'Evt_CSV_Min_Tagged',
    'BDT_common5_input_transverse_sphericity_jets',
    'Evt_M_JetsAverage',
    'Evt_CSV_Min',
    'BDT_common5_input_max_dR_jj',
    'Evt_M_MinDeltaRLeptonTaggedJet',
    'Evt_JetPtOverJetE',
    'Evt_M2_TaggedJetsAverage',
    'N_Jets',
    'Evt_blr_ETH',
    'Evt_Dr_MinDeltaRTaggedJets',
    'Evt_Dr_TaggedJetsAverage',
    'BDT_common5_input_closest_tagged_dijet_mass',
    'Weight_GEN_nom',
    'Evt_HT',
    'Weight_XS',
    'BDT_common5_input_sphericity_tags',
    'BDT_common5_input_HT_tag',
    'BDT_common5_input_dev_from_avg_disc_btags',
    'Evt_CSV_Average_Tagged',
    'Evt_CSV_Average',
    'Evt_Dr_MinDeltaRLeptonTaggedJet',
    'N_BTagsT',
    'N_BTagsM',
    #'Evt_Odd',
    'Evt_Deta_TaggedJetsAverage',
    'Weight_CSV',
    'Evt_blr_ETH_transformed',
    'BDT_common5_input_sphericity_jets',
    'Jet_Pt[2]',
    'Jet_Pt[0]',
    'Jet_Pt[1]',
    'CSV[1]',
    'Jet_CSV[0]',
    'Jet_CSV[3]',
    #'class_label',
    'memDBp',
    ]



all_variables = list(set( [v for key in variables for v in variables[key] ] ))
