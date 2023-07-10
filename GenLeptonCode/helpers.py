import ROOT
import os

ROOT.gInterpreter.Declare(f'#include "{os.getenv("RUN_PATH")}/GenLeptonCode/GenLepton.h"')

def get_GenLepton_rdf(rdf):
    rdf = rdf.Define("GenLeptons", "reco_tau::gen_truth::GenLepton::fromNanoAOD(GenPart_pt, GenPart_eta, GenPart_phi, GenPart_mass, GenPart_genPartIdxMother, GenPart_pdgId, GenPart_statusFlags)")
    # Get needed information on GenLepton
    rdf = rdf.Define("GenLepton_pt", "ROOT::VecOps::RVec<float> out; for(size_t n = 0; n < GenLeptons.size(); ++n) out.push_back(GenLeptons.at(n).visibleP4().pt()); return out")
    rdf = rdf.Define("GenLepton_eta", "ROOT::VecOps::RVec<float> out; for(size_t n = 0; n < GenLeptons.size(); ++n) out.push_back(GenLeptons.at(n).visibleP4().eta()); return out")
    rdf = rdf.Define("GenLepton_phi", "ROOT::VecOps::RVec<float> out; for(size_t n = 0; n < GenLeptons.size(); ++n) out.push_back(GenLeptons.at(n).visibleP4().phi()); return out")
    rdf = rdf.Define("GenLepton_kind", "ROOT::VecOps::RVec<int> out; for(size_t n = 0; n < GenLeptons.size(); ++n) out.push_back(static_cast<int>(GenLeptons.at(n).kind())); return out")
    #rdf = rdf.Define("GenLepton_nChargedHad", "ROOT::VecOps::RVec<int> out; for(size_t n = 0; n < GenLeptons.size(); ++n) out.push_back(static_cast<int>(GenLeptons.at(n).nChargedHadrons())); return out")
    #rdf = rdf.Define("GenLepton_nNeutralHad", "ROOT::VecOps::RVec<int> out; for(size_t n = 0; n < GenLeptons.size(); ++n) out.push_back(static_cast<int>(GenLeptons.at(n).nNeutralHadrons())); return out")
    
    return rdf