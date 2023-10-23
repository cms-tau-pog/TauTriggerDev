HLTnameDiTau = 'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1'
HLTnameSingleTau = 'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3'

rate_deepTau_DiTau = 56
rate_deepTau_SingleTau = 18.1

WP_params_SingleTau = {
    'Tight': {'t1': '1.0', 
              't2': '0.99',
              'rate': 12.9},
    'Medium': {'t1': '1.0', 
               't2': '0.95',
              'rate': 17.9},
    'Loose': {'t1': '0.94', 
              't2': '0.90',
              'rate': 28.3},
    'MaxEff': {'t1': '0.0', 
               't2': '0.0',
              'rate': 779.4},
}

WP_params_DiTau = {
    'Tight': {'t1': '0.60', 
              't2': '0.50',
              'rate': 45.79},
    'Medium': {'t1': '0.56', 
               't2': '0.47',
              'rate': 55.69},
    'Loose': {'t1': '0.52', 
              't2': '0.42',
              'rate': 69.87},     
    'MaxEff': {'t1': '0.0', 
               't2': '0.0',
              'rate': 4135.3},
}
