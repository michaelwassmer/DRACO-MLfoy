from keras import backend as K

def asimovLoss(y_true, y_pred):
    #Continuous version:
    #lumi             = 41.1  #luminosity in 1/fb at13TeV
    #eff              = 0.015
    systematic       = 0.1
    #cross_sec_signal = 293.4 # in fb
    #cross_sec_bkg    = 844000.

    expected_signal = 198.0 #16.9 # 1232 #lumi * cross_sec_signal * eff
    expected_bkg    = 19242.0 #382.9 # 582035 #lumi * cross_sec_bkg * eff

    scale_factor = 0.01
    signal_weight = scale_factor * expected_signal / K.sum(y_true)
    bkg_weight    = scale_factor * expected_bkg / K.sum(1 - y_true)

    s       = signal_weight * K.sum(y_pred * y_true)
    b       = bkg_weight * K.sum(y_pred * (1 - y_true))
    sys_bkg = systematic * b

    ###  approximations of statistical significance, s / sqrt(s + b)
    #return (s + b) / (s*s + K.epsilon())
    return - (s*s) / (s + b + K.epsilon())
    ### asimov with systematic bkg uncertainty
    #return 1./ (K.square(2 * ((s+b) * K.log((s+b) * (b + sys_bkg*sys_bkg) / (b*b + (s+b) * sys_bkg*sys_bkg + K.epsilon()) + K.epsilon()) //
    #       - b*b * K.log(1 + sys_bkg*sys_bkg * s / (b * (b + sys_bkg*sys_bkg) + K.epsilon())) / (sys_bkg*sys_bkg + K.epsilon()))))
    ### asimov without systematic bkg uncertainty
    #return 1. / K.square(2 * ((s+b) * K.log(1 + s/b) - s))
    #return - K.square(2 * ((s+b) * K.log(1 + s/b) - s))
