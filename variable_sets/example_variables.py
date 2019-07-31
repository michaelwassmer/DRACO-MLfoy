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

"""
variables["ge6j_ge3t"] = [
    "Evt_JetPtOverJetE",
    "Evt_Dr_MinDeltaRTaggedJets",
    "Evt_Dr_TaggedJetsAverage",
    "Jet_CSV[0]",
    #"Jet_CSV[3]",
    "BDT_common5_input_closest_tagged_dijet_mass",
    "Evt_M_MinDeltaRLeptonTaggedJet",
    "CSV[1]",
    #"BDT_common5_input_sphericity_jets",
    #"BDT_common5_input_sphericity_tags",
    #"BDT_common5_input_transverse_sphericity_jets",
    #"BDT_common5_input_dev_from_avg_disc_btags",
    "N_BTagsT",
    "N_BTagsM",
    "N_Jets",
    #"Evt_blr_ETH",
    "Evt_Deta_TaggedJetsAverage",
    "Evt_M2_TaggedJetsAverage",
    "Evt_CSV_Average",
    "Evt_HT",
    "Evt_CSV_Average_Tagged",
    "Evt_CSV_Min_Tagged",
    "Evt_CSV_Min",
    "BDT_common5_input_HT_tag",
    "Evt_M_JetsAverage",
    #"memDBp",
    #"Evt_blr_ETH_transformed",
    "BDT_common5_input_max_dR_jj",
    "Jet_Pt[0]",
    #"Jet_Pt[1]",
    "Jet_Pt[2]",
    "Evt_Dr_MinDeltaRLeptonTaggedJet",
    ]
"""


variables["ge6j_ge3t"] = [
    "MVA_HT_tag",
    "MVA_Evt_CSV_Average",
    "N_Jets",
    "Evt_Pt_MinDeltaRTaggedJets",
    "MVA_tagged_dijet_mass_closest_to_125",
    "Evt_M_MinDeltaRTaggedJets",
    "MVA_Evt_M2_TaggedJetsAverage",
    "MVA_Evt_Deta_JetsAverage",
    "Evt_M_TaggedJetsAverage",
    "Jet_Pt[0]",
    "CSV[3]",
    #"MVA_max_dR_bb",
    #"Evt_HT_Jets",
    #"MVA_M3",
    #"Evt_M_MinDeltaRLeptonTaggedJet",
    #"Evt_Dr_MinDeltaRJets",
    #"MVA_cos_theta_l_bhad",
    #"Evt_M_MinDeltaRJets",
    #"MVA_max_dR_jj",
    #"Weight_GEN_nom",
    #"MVA_pt_all_jets_over_E_all_jets",
    #"N_LooseJets",
    #"N_BTagsL",
    #"N_BTagsM",
    #"N_BTagsT",
    #"MVA_delta_eta_blep_bhad",
    #"N_PrimaryVertices",
    #"MVA_h0",
    #"MVA_maxeta_jet_tag",
    #"Evt_Jet_MaxDeta_Jets",
    #"MVA_transverse_sphericity_jets",
    #"Evt_Eta_JetsAverage",
    #"MVA_MET",
    #"MVA_delta_phi_blep_bhad",
    #"MVA_lowest_btag",
    #"MVA_closest_tagged_dijet_mass",
    #"Evt_JetPtOverJetE",
    #"MVA_cos_theta_blep_bhad",
    #"MVA_MHT",
    #"MVA_Evt_Deta_TaggedJetsAverage",
    #"Evt_Dr_MinDeltaRTaggedJets",
    #"MVA_best_higgs_mass",
    #"MVA_sphericity",
    #"Weight_CSV",
    #"MVA_invariant_mass_of_everything",
    #"MVA_h3",
    #"MVA_h2",
    #"MVA_h1",
    #"MVA_avg_btag_disc_btags",
    #"MVA_aplanarity_tags",
    #"MVA_dev_from_avg_disc_btags",
    #"MVA_all_sum_pt_with_met",
    #"MVA_maxeta_tag_tag",
    #"Evt_Pt_MinDeltaRJets",
    #"Evt_Dr_JetsAverage",
    #"Evt_CSV_Average_Tagged",
    #"MVA_maxeta_jet_jet",
    #"Evt_Dr_MinDeltaRLeptonTaggedJet",
    #"Evt_Odd",
    #"MVA_avg_dr_tagged_jets",
    #"Evt_Dr_MinDeltaRLeptonJet",
    #"Evt_M_MedianTaggedJets",
    #"MVA_dr_between_lep_and_closest_jet",
    #"MVA_transverse_sphericity_tags",
    #"Evt_CSV_Min_Tagged",
    #"Evt_Eta_TaggedJetsAverage",
    #"MVA_transverse_sphericity",
    #"Evt_M_JetsAverage",
    #"Evt_CSV_Min",
    #"MVA_delta_phi_l_bhad",
    #"MVA_sphericity_jets",
    #"MVA_delta_eta_l_bhad",
    #"MVA_blr_transformed",
    #"Evt_Dr_TaggedJetsAverage",
    #"Weight_XS",
    #"MVA_dEta_fn",
    #"MVA_blr",
    #"MVA_pt_all_jets_over_E_all_jets_tags",
    #"MVA_HT",
    #"MVA_aplanarity_jets",
    #"MVA_Mlb",
    #"Evt_Pt_MET",
    #"MVA_aplanarity",
    #"Evt_M_MinDeltaRLeptonJet",
    #"MVA_sphericity_tags",
    #"Evt_M2_JetsAverage",
    #"LooseLepton_Pt[0]",
    #"Jet_Eta[2]",
    #"Jet_Eta[3]",
    #"Jet_Eta[4]",
    #"Jet_Eta[5]",
    #"Jet_Eta[0]",
    #"Jet_Eta[1]",
    #"LooseLepton_Eta[0]",
    #"Jet_E[4]",
    #"Jet_E[3]",
    #"Jet_E[5]",
    #"Jet_E[0]",
    #"Jet_E[2]",
    #"Jet_E[1]",
    #"LooseLepton_E[0]",
    #"Jet_Pt[2]",
    #"Jet_Pt[3]",
    #"Jet_Pt[4]",
    #"Jet_Pt[5]",
    #"Jet_Pt[1]",
    #"Jet_DeepJetCSV[5]",
    #"Jet_DeepJetCSV[3]",
    #"Jet_DeepJetCSV[4]",
    #"Jet_DeepJetCSV[1]",
    #"Jet_DeepJetCSV[2]",
    #"Jet_DeepJetCSV[0]",
    #"Jet_Phi[3]",
    #"Jet_Phi[5]",
    #"Jet_Phi[4]",
    #"Jet_Phi[0]",
    #"Jet_Phi[1]",
    #"Jet_Phi[2]",
    #"Jet_M[0]",
    #"Jet_M[2]",
    #"Jet_M[1]",
    #"Jet_M[5]",
    #"Jet_M[4]",
    #"Jet_M[3]",
    #"CSV[4]",
    #"CSV[5]",
    #"CSV[0]",
    #"CSV[1]",
    #"CSV[2]",
    #"LooseLepton_Phi[0]",
    #"class_label",
    ]




all_variables = list(set( [v for key in variables for v in variables[key] ] ))
