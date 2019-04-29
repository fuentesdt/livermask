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
    normalizedimage = options.image.replace('Ven.raw.nii.gz','Ven.normalize.nii.gz')
    getHeaderCmd = 'c3d %s -replace 2 1 3 1 4 1 5 0 -centroid | grep CENTROID_VOX | sed "s/CENTROID_VOX\ //g"' % (options.gmm)
    print getHeaderCmd 
    try: 
      headerProcess = subprocess.Popen(getHeaderCmd ,shell=True,stdout=subprocess.PIPE ,stderr=subprocess.PIPE)
      while ( headerProcess.poll() == None ):
         pass
      headerinfo   =   headerProcess.stdout.readline().strip('\n')
      print headerinfo   
      centroid = eval( headerinfo   )
      print centroid 
    except (NameError, SyntaxError) as excp: 
      print excp
    npixel = 384
    roi = [ int(centroid[0]) - npixel/2. , int(centroid[0]) -1 + npixel/2. , 
            int(centroid[1]) - npixel/2. , int(centroid[1]) -1 + npixel/2. ] 
    print roi 
      
else:
  parser.print_help()
  print options
 
