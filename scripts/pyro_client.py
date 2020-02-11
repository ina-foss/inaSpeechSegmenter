import Pyro4
import sys
import os
import socket
import time
#from inaSpeechSegmenter import Segmenter
#from speech_zic_male_female import do_segmentation_v3

from inaSpeechSegmenter import Segmenter, seg2csv

#def args2url(args):
#    return 'http://collgate.ina.fr:81/collgate.dlweb/get/%s/%s/%sT%s0000/%d?download&format=ts' % args

#def args2csvname(args):
#    return '/rex/store1/home/ddoukhan/genderface/' + '-'.join([str(e) for e in args]) + '.csv'

def myprocess(url, outname):
    if os.path.isfile(outname):
        return 'already done'
    results = g(url)
    seg2csv(results, outname)
    return 0

if __name__ == '__main__':
    dname = os.path.dirname(os.path.realpath(__file__))

    #segmenter = Segmenter()
    
    #sznn = keras.models.load_model(dname + '/keras_speech_music_cnn.hdf5')
#    sznn = keras.models.load_model(dname + '/kerasSZ21-0621-0.0333-0.9913.hdf5')
#    gendernn = keras.models.load_model(dname + '/keras_male_female_cnn.hdf5')

    
    hostname = socket.gethostname()

#    mountpoint = sys.argv[2]
#    assert os.access(mountpoint, os.W_OK)
    
    uri = sys.argv[1]
    jobserver = Pyro4.Proxy(uri)

    b = time.time()
    ret = -1
    outname = 'init'
    
    g = Segmenter()
    
    while True:
        url, outname = jobserver.get_job('%s %f %s %s' % (hostname, time.time() - b, ret, outname))
            
        b = time.time()
        # store = '/home/ddoukhan/rexstorenat/'
 #       store = '%s/%s/%s/%s/' % (mountpoint, media, channel, date[:4])
 #       os.makedirs(store, exist_ok=True)
        print(url, outname)
        
        try:
            ret =  myprocess(url, outname)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            ret = 'error'
        #do_segmentation_v3(media, channel, date, '%02d0000' % mtime, dur, segmenter, store)