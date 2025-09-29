import awkward as ak
import numpy as np
from spritz.modules.MuonScaRe import pt_resol, pt_scale, pt_resol_var, pt_scale_var 
import correctionlib

def getMuonCorrectionJsons(cfg):
    muon_sr_file = cfg["rochester_file"]
    cset_rochester = correctionlib.CorrectionSet.from_file(muon_sr_file)
    return cset_rochester


def correctRochester(events, is_data, rochester):
    # muons = events.Muon[ak.mask(events.Lepton.muonIdx, mu_mask)]
    muons = events.Muon
    events[("Muon", "charge")] = events.Muon.pdgId / (-abs(events.Muon.pdgId))
    
    print("--->Correcting muon momentum")
    
    if is_data:
        # Data: only scale correction to gen Z peak
        events["Muon", "ptcorr"] = pt_scale(
            1, # 1 for data, 0 for mc 
            events.Muon.pt, 
            events.Muon.eta, 
            events.Muon.phi, 
            events.Muon.charge, 
            rochester, 
            nested=True # for awkward arrays. Set False for 1d arrays
        )
        
    else:
        # MC: both scale correction to gen Z peak AND resolution correction to Z width in data
        events["Muon", "ptscalecorr"] = pt_scale(
            0, 
            events.Muon.pt, 
            events.Muon.eta, 
            events.Muon.phi, 
            events.Muon.charge, 
            rochester, 
            nested=True
        )

        events["Muon", "ptcorr"] = pt_resol(
            events.Muon.ptscalecorr, 
            events.Muon.eta, 
            events.Muon.phi,
            events.Muon.nTrackerLayers, 
            events.event,
            events.luminosityBlock,
            rochester,
            nested=True
        )

        # uncertainties
        events["Muon", "ptscalecorr_up"] = pt_scale_var(
            events.Muon.ptcorr, 
            events.Muon.eta, 
            events.Muon.phi, 
            events.Muon.charge,
            "up",
            rochester, 
            nested=True
        )
        
        events["Muon", "ptscalecorr_dn"] = pt_scale_var(
            events.Muon.ptcorr, 
            events.Muon.eta, 
            events.Muon.phi, 
            events.Muon.charge,
            "dn",
            rochester, 
            nested=True
        )

        events["Muon", "ptcorr_resolup"] = pt_resol_var(
            events.Muon.ptscalecorr, 
            events.Muon.ptcorr, 
            events.Muon.eta, 
            "up",
            rochester, 
            nested=True
        )
        events["Muon", "ptcorr_resoldn"] = pt_resol_var(
            events.Muon.ptscalecorr, 
            events.Muon.ptcorr, 
            events.Muon.eta, 
            "dn",
            rochester, 
            nested=True
        )
        
    print("Difference 1")
    diff = events.Muon.pt - events.Muon.ptcorr
    diff = ak.flatten(diff[diff!=0])
    print(diff)
    
    # finally, set the corrected pt as the default pt for muons
    events["Muon", "pt_uncorr"] = events.Muon.pt
    events[("Muon", "pt")] = events.Muon.ptcorr
    
    
    # store old leptons 
    events["Lepton_uncorr"] = events.Lepton
    
    #set the same for all Leptons 
    mu_pt = events.Muon.pt  # corrected pt

    mu_idx = ak.to_packed(events.Lepton.muonIdx)  # flattened integer indices
    mu_pt = mu_pt[mu_idx]  # pick the corresponding pt for each muon in Lepton

    mu_mask = abs(events.Lepton.pdgId) == 13

    events[("Lepton", "pt")] = ak.where(mu_mask, mu_pt, events.Lepton.pt)
    
    return events
