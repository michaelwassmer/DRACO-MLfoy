variables = {}

variables["leptonic"] = [
"Evt_Pt_MET",
"Evt_Phi_MET",
"M_W_transverse[0]",
"CaloMET",
"CaloMET_PFMET_ratio",
"N_TightMuons",
"Muon_Pt[0]",
"Muon_Eta[0]",
"Muon_E[0]",
"Muon_Phi[0]",
"N_TightElectrons",
"Electron_Pt[0]",
"Electron_Eta[0]",
"Electron_E[0]",
"Electron_Phi[0]",
"N_BTagsL",
"N_BTagsM",
"N_BTagsT",
"N_Jets",
"N_LooseJets",
"Jet_Pt[0]",
"Jet_Pt[1]",
"Jet_Pt[2]",
"Jet_Pt[3]",
"Jet_Eta[0]",
"Jet_Eta[1]",
"Jet_Eta[2]",
"Jet_Eta[3]",
"Jet_E[0]",
"Jet_E[1]",
"Jet_E[2]",
"Jet_E[3]",
"Jet_Phi[0]",
"Jet_Phi[1]",
"Jet_Phi[2]",
"Jet_Phi[3]",
"Jet_CSV[0]",
"Jet_CSV[1]",
"Jet_CSV[2]",
"Jet_CSV[3]",
"DeltaPhi_AK4Jet_MET[0]",
"DeltaPhi_AK4Jet_MET[1]",
"DeltaPhi_AK4Jet_MET[2]",
"DeltaPhi_AK4Jet_MET[3]",
]
all_variables = list(set( [v for key in variables for v in variables[key] ] ))

print all_variables
