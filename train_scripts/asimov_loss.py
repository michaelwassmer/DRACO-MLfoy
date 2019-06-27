from keras import backend as K
from keras import losses

expected_signal = 198.0   #lumi * cross_sec_signal * eff
expected_bkg    = 19242.0 #lumi * cross_sec_bkg * eff
scale_factor = 0.01

systematic       = 0.3

def asimovLossWithSys(y_true, y_pred):

    signal_weight = scale_factor * expected_signal / K.sum(y_true)
    bkg_weight    = scale_factor * expected_bkg / K.sum(1 - y_true)

    s       = signal_weight * K.sum(y_pred * y_true)
    b       = bkg_weight * K.sum(y_pred * (1 - y_true))
    sys_bkg = systematic * b

    ### asimov with systematic bkg uncertainty
    return -(K.square(2*((s+b)*K.log((s+b)*(b+sys_bkg*sys_bkg)/(b*b+(s+b)*sys_bkg*sys_bkg+K.epsilon())+K.epsilon())  //
           -b*b*K.log(1+sys_bkg*sys_bkg*s/(b*(b+sys_bkg*sys_bkg)+K.epsilon()))/(sys_bkg*sys_bkg+K.epsilon()))))

def asimovLossWithSysInv(y_true, y_pred):

    signal_weight = scale_factor * expected_signal / K.sum(y_true)
    bkg_weight    = scale_factor * expected_bkg / K.sum(1 - y_true)

    s       = signal_weight * K.sum(y_pred * y_true)
    b       = bkg_weight * K.sum(y_pred * (1 - y_true))
    sys_bkg = systematic * b

    ### inverted asimov with systematic bkg uncertainty
    return 1./(K.square(2*((s+b)*K.log((s+b)*(b+sys_bkg*sys_bkg)/(b*b+(s+b)*sys_bkg*sys_bkg+K.epsilon())+K.epsilon())  //
           -b*b*K.log(1+sys_bkg*sys_bkg*s/(b*(b+sys_bkg*sys_bkg)+K.epsilon()))/(sys_bkg*sys_bkg+K.epsilon()))))

def asimovLossWoutSys(y_true, y_pred):

    signal_weight = scale_factor * expected_signal / K.sum(y_true)
    bkg_weight    = scale_factor * expected_bkg / K.sum(1 - y_true)

    s = signal_weight * K.sum(y_pred * y_true)
    b = bkg_weight * K.sum(y_pred * (1 - y_true))

    ### asimov without systematic bkg uncertainty
    return -K.square(2*((s+b)*K.log(1+s/b)-s))

def asimovLossWoutSysInv(y_true, y_pred):

    signal_weight = scale_factor * expected_signal / K.sum(y_true)
    bkg_weight    = scale_factor * expected_bkg / K.sum(1 - y_true)

    s       = signal_weight * K.sum(y_pred * y_true)
    b       = bkg_weight * K.sum(y_pred * (1 - y_true))

    ### inverted asimov without systematic bkg uncertainty
    return 1./K.square(2*((s+b)*K.log(1+s/b)-s))

def significanceLoss(y_true, y_pred):

    signal_weight = scale_factor * expected_signal / K.sum(y_true)
    bkg_weight    = scale_factor * expected_bkg / K.sum(1 - y_true)

    s       = signal_weight * K.sum(y_pred * y_true)
    b       = bkg_weight * K.sum(y_pred * (1 - y_true))

    ###  approximations of statistical significance, s / sqrt(s + b)
    return -(s*s)/(s+b+K.epsilon())

def significanceLossInv(y_true, y_pred):

    signal_weight = scale_factor * expected_signal / K.sum(y_true)
    bkg_weight    = scale_factor * expected_bkg / K.sum(1 - y_true)

    s = signal_weight * K.sum(y_pred * y_true)
    b = bkg_weight * K.sum(y_pred * (1 - y_true))

    ### inverted approximations of statistical significance, s / sqrt(s + b)
    return (s+b)/(s*s+K.epsilon())


losses.asimov_with_sys     = asimovLossWithSys
losses.asimov_with_sys_inv = asimovLossWithSysInv
losses.asimov_wout_sys     = asimovLossWoutSys
losses.asimov_wout_sys_inv = asimovLossWoutSysInv
losses.sig_loss            = significanceLoss
losses.sig_inv_loss        = significanceLossInv
