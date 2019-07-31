from keras import backend as K
from keras import losses
import tensorflow as tf

expected_signal = 341.0 * (59.7 / 41.5)         # lumi * cross_sec_signal * eff
expected_bkg    = 19242.0 * (59.7 / 41.5)       # lumi * cross_sec_bkg * eff
scale_factor    = 0.01

systematic      = [0.3] #[0.04, 0.04, 0.025]  # pdf gg + QCD scale ttbar + Lumi 13TeV 2018

def asimovLossWithSys(y_true, y_pred):

    signal_weight = scale_factor * expected_signal / K.sum(y_true)
    bkg_weight    = scale_factor * expected_bkg / K.sum(1 - y_true)

    s       = signal_weight * K.sum(y_pred * y_true)
    b       = bkg_weight * K.sum(y_pred * (1 - y_true))
    sys_bkg = (sum([(sys * b)**2 for sys in systematic]))**(0.5)

    ### asimov with systematic bkg uncertainty
    return -(K.square(2*((s+b)*K.log((s+b)*(b+sys_bkg*sys_bkg)/(b*b+(s+b)*sys_bkg*sys_bkg+K.epsilon())+K.epsilon())  //
           -b*b*K.log(1+sys_bkg*sys_bkg*s/(b*(b+sys_bkg*sys_bkg)+K.epsilon()))/(sys_bkg*sys_bkg+K.epsilon()))))

def asimovLossWithSysInv(y_true, y_pred):

    signal_weight = scale_factor * expected_signal / K.sum(y_true)
    bkg_weight    = scale_factor * expected_bkg / K.sum(1 - y_true)

    s       = signal_weight * K.sum(y_pred * y_true)
    b       = bkg_weight * K.sum(y_pred * (1 - y_true))
    sys_b = (K.sum([(sys * b)**2 for sys in systematic]))**(0.5)

    s = tf.Print(s, [s, b], "s,b: ", summarize=1)
    ### inverted asimov with systematic bkg uncertainty
    Z = K.sqrt(2*((s+b)*K.log((s+b)*(b+sys_b*sys_b)/(b*b+(s+b)*sys_b*sys_b+1e-7)+1e-7)-b*b*K.log(1+sys_b*sys_b*s/(b*(b+sys_b*sys_b)+1e-7))/(sys_b*sys_b+1e-7)))
    return 1. / Z

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

def testLoss(y_true, y_pred):
    signal_weight = scale_factor * expected_signal / K.sum(y_true)
    bkg_weight    = scale_factor * expected_bkg / K.sum(1 - y_true)
    s = signal_weight * K.sum(y_pred * y_true)
    b = bkg_weight * K.sum(y_pred * (1 - y_true))
    sys_bkg = (sum([(sys * b)**2 for sys in systematic]))**(0.5)
    #s = tf.Print(s, [s, b], "s,b: ", summarize=1)
    return -s*s*s+b

value_range = [0.0, 1.0]
lam = 20.

def histLoss(y_true, y_pred):
    def WeightLoss(_, y_pred, CSV_weights):
        hist = tf.to_float(tf.histogram_fixed_width(y_pred, value_range, nbins=30))+K.epsilon()
        #hist = tf.Print(hist, [hist], first_n=10)
        #y_true, CSV_weights, up, down = tf.split(y_true, num_or_size_splits=4, axis=0)
        #y_true = tf.Print(y_true, [y_true])
        CSV_weights = tf.Print(CSV_weights, [CSV_weights])
        #y_pred = tf.Print(y_pred, [y_pred], first_n=10)
        sigma = 0.1 * hist
        dist = tf.distributions.Normal(loc=hist, scale=sigma)
        varHist = dist.sample()
        #varHist = tf.Print(varHist, [varHist], first_n=10)
        batchSize = 4000 #K.sum(y_true) + K.sum(1-y_true)
        #L = lam * K.sum(K.pow((hist-varHist)/hist, 2)) / batchSize
        L = lam * K.pow((hist-varHist)/hist, 2)
        L =  lam* K.sum(L)/batchSize #K.mean(L, axis=-1)
        #L = tf.Print(L, [L])
        return L

    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) + WeightLoss(None, y_pred, Weights)




losses.asimov_with_sys     = asimovLossWithSys
losses.asimov_with_sys_inv = asimovLossWithSysInv
losses.asimov_wout_sys     = asimovLossWoutSys
losses.asimov_wout_sys_inv = asimovLossWoutSysInv
losses.sig_loss            = significanceLoss
losses.sig_inv_loss        = significanceLossInv
losses.test                = testLoss
losses.hist_loss           = histLoss
