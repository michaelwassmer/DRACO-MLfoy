from keras import backend as K

def asimovLoss(y_true, y_pred, counter=[0]):
    #Continuous version:
    #lumi             = 41.1  #luminosity in 1/fb at13TeV
    #eff              = 0.015
    systematic       = 0.3
    #cross_sec_signal = 293.4 # in fb
    #cross_sec_bkg    = 844000.

    expected_signal = 16.9 #lumi * cross_sec_signal * eff
    expected_bkg    = 328.9 #lumi * cross_sec_bkg * eff

    signal_weight = expected_signal / K.sum(y_true)
    bkg_weight    = expected_bkg / K.sum(1 - y_true)

    s       = signal_weight * K.sum(y_pred * y_true)
    b       = bkg_weight * K.sum(y_pred * (1 - y_true))
    sys_bkg = systematic * b

    counter[0]+=1 # mutable variable get evaluated ONCE

    #if(counter[0]<10):
     #   return (s + b) / (s*s + K.epsilon())
    #else:
    return 1./ (K.square(2 * ((s+b) * K.log((s+b) * (b + sys_bkg*sys_bkg) / (b*b + (s+b) * sys_bkg*sys_bkg + K.epsilon()) + K.epsilon()) //
           - b*b * K.log(1 + sys_bkg*sys_bkg * s / (b * (b + sys_bkg*sys_bkg) + K.epsilon())) / (sys_bkg*sys_bkg + K.epsilon()))))
           #Add the epsilon to avoid dividing by 0
