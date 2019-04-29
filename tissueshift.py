import subprocess
import os

# setup command line parser to control execution
from optparse import OptionParser
parser = OptionParser()
parser.add_option( "--image",
                  action="store", dest="image", default=None,
                  help="anatomy image", metavar="FILE")
parser.add_option( "--gmm",
                  action="store", dest="gmm", default=None,
                  help="gmm image", metavar="FILE")
parser.add_option( "--output",
                  action="store", dest="output", default='output%02d.nii.gz',
                  help="output path image", metavar="FILE")
(options, args) = parser.parse_args()


if (options.image != None and options.gmm != None ):
    normalizedimage = options.image.replace('Ven.roi.nii.gz','Ven.normroi.nii.gz')
    getHeaderCmd = 'c3d %s %s -lstat | sed "s/^\s\+//g;s/\s\+/,/g;s/Vol(mm^3)/Vol.mm.3/g;s/Extent(Vox)/ExtentX,ExtentY,ExtentZ/g" ' % (options.image,options.gmm)
    print getHeaderCmd 
    try: 
      headerProcess = subprocess.Popen(getHeaderCmd ,shell=True,stdout=subprocess.PIPE ,stderr=subprocess.PIPE)
      while ( headerProcess.poll() == None ):
         pass
      headerinfo   =   headerProcess.stdout.readline().strip('\n')
      rawlstatinfo = [ dict(zip(headerinfo.split(','),map(float,line.strip('\n').split(','))))  for line in headerProcess.stdout.readlines()]
      labeldictionary = dict([ (int(datdic['LabelID']),{'Mean':datdic['Mean'],'StdD':datdic['StdD'],'Vol.mm.3':datdic['Vol.mm.3']}) for datdic in rawlstatinfo if int(datdic['LabelID']) > 0 ] )
    except (NameError, SyntaxError) as excp: 
      print excp
    #print labeldictionary 
    # equation for a line  y = y_0  + (y_1 - y_0)  * (x - x_0) /(x_1 - x_0)
    #  y(x) = y_0  + (y_1 - y_0)  * (x - x_csf) /(x_wm - x_csf)
    #     ==> y(x_csf) = y_0  
    #     ==> y(x_wm)  = y_1  
    xoffset = -labeldictionary[1]['Mean']
    shiftcmd = 'c3d %s -shift %12.5e  -type float -o %s ' % (options.image,xoffset , normalizedimage )
    print shiftcmd 
    os.system(shiftcmd )
    verifyrescalecmd = 'c3d %s %s -lstat  ' % (normalizedimage , options.gmm)
    print verifyrescalecmd 
    os.system( verifyrescalecmd  )
    ## IntensityMax  = 2.0
    ## IntensityMin  = 0.0
    ## MaxIter = 7
    ## tissuebins = { }
    ## for iditer in range(MaxIter):
    ##    ntissue = int(pow(2.,iditer))
    ##    binsize = (IntensityMax-  IntensityMin)/ntissue 
    ##    # open set on lb
    ##    tissuebins[ntissue] = dict([(idtissue,["%f" % ((idtissue-1)*binsize+.0001), "%f" % (idtissue*binsize)])for idtissue in range(1,ntissue+1)])
    ##    # set min to -inf
    ##    tissuebins[ntissue][1][0] = '-inf'
    ##    # set max to inf
    ##    tissuebins[ntissue][ntissue][-1] = 'inf'
    ##    for idtissue,binrange in tissuebins[ntissue].iteritems():
    ##      if (idtissue == 1 ):
    ##        binimagecmd = 'c3d %s -as anat -thresh %s %s %d 0 -as seg' % (normalizedimage, binrange[0], binrange[1], idtissue )
    ##      else:
    ##        binimagecmd = binimagecmd + ' -push anat -thresh %s %s %d 0 -push seg -add -as seg  ' % (binrange[0], binrange[1], idtissue )
    ##    binimagecmd = binimagecmd + ' %s -thresh 1 4 1 0 -multiply -o  %s  ' % ( options.gmm,options.output % ntissue )
    ##    print binimagecmd 
    ##    os.system( binimagecmd )
      
else:
  parser.print_help()
  print options
 
