variables = {}
variables["ge4j_ge3t"] = [
	#"N_Jets",
	#"N_BTags_M",
	"TopHad_B_Pt",
	"TopHad_B_Eta",
	"TopHad_B_Phi",
	"TopHad_B_E",
	"TopHad_B_CSV",
	"TopLep_B_Pt",
	"TopLep_B_Eta",
	"TopLep_B_Phi",
	"TopLep_B_E",
	"TopLep_B_CSV",
	"TopHad_Q1_Pt",
	"TopHad_Q1_Eta",
	"TopHad_Q1_Phi",
	"TopHad_Q1_E",
	"TopHad_Q1_CSV",
	"TopHad_Q2_Pt",
	"TopHad_Q2_Eta",
	"TopHad_Q2_Phi",
	"TopHad_Q2_E",
	"TopHad_Q2_CSV",
	"Muon_Pt[0]",
	"Muon_Eta[0]",
	"Muon_Phi[0]",
	"Muon_E[0]",
	"Electron_Pt[0]",
	"Electron_Eta[0]",
	"Electron_Phi[0]",
	"Electron_E[0]",
	"Evt_MET_Pt",
	"Evt_MET_Phi",
	"reco_TopHad_Pt",
	"reco_TopHad_Eta",
	"reco_TopHad_Phi",
	"reco_TopHad_M",
	"reco_TopHad_logM",
	"reco_TopLep_Pt",
	"reco_TopLep_Eta",
	"reco_TopLep_Phi",
	"reco_TopLep_M",
	"reco_TopLep_logM",
	"reco_WHad_Pt",
	"reco_WHad_Eta",
	"reco_WHad_Phi",
	"reco_WHad_M",
	"reco_WHad_logM",
	"reco_WLep_Pt",
	"reco_WLep_Eta",
	"reco_WLep_Phi",
	"reco_WLep_M",
	"reco_WLep_logM",
	"ttbar_phi",
	"ttbar_pt_div_ht_p_met"
	]


variables["ge4j_ge2t"] = [
        #"N_Jets",
        #"N_BTags_M",
        "TopHad_B_Pt",
        "TopHad_B_Eta",
        "TopHad_B_Phi",
        "TopHad_B_E",
        "TopHad_B_CSV",
        "TopLep_B_Pt",
        "TopLep_B_Eta",
        "TopLep_B_Phi",
        "TopLep_B_E",
        "TopLep_B_CSV",
        "TopHad_Q1_Pt",
        "TopHad_Q1_Eta",
        "TopHad_Q1_Phi",
        "TopHad_Q1_E",
        "TopHad_Q1_CSV",
        "TopHad_Q2_Pt",
        "TopHad_Q2_Eta",
        "TopHad_Q2_Phi",
        "TopHad_Q2_E",
        "TopHad_Q2_CSV",
        "Muon_Pt[0]",
        "Muon_Eta[0]",
        "Muon_Phi[0]",
        "Muon_E[0]",
        "Electron_Pt[0]",
        "Electron_Eta[0]",
        "Electron_Phi[0]",
        "Electron_E[0]",
        "Evt_MET_Pt",
        "Evt_MET_Phi",
        "reco_TopHad_Pt",
        "reco_TopHad_Eta",
        "reco_TopHad_Phi",
        "reco_TopHad_M",
        "reco_TopHad_logM",
        "reco_TopLep_Pt",
        "reco_TopLep_Eta",
        "reco_TopLep_Phi",
        "reco_TopLep_M",
        "reco_TopLep_logM",
        "reco_WHad_Pt",
        "reco_WHad_Eta",
        "reco_WHad_Phi",
        "reco_WHad_M",
        "reco_WHad_logM",
        "reco_WLep_Pt",
        "reco_WLep_Eta",
        "reco_WLep_Phi",
        "reco_WLep_M",
        "reco_WLep_logM",
        "ttbar_phi",
        "ttbar_pt_div_ht_p_met"
        ]


all_variables = list(set( [v for key in variables for v in variables[key] ] ))

