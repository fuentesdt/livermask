import subprocess
import os
import numpy as np
import nibabel as nib  

# raw dicom data is usually short int (2bytes) datatype
# labels are usually uchar (1byte)
IMG_DTYPE = np.int16
SEG_DTYPE = np.uint8
_globalexpectedpixel=512

# setup command line parser to control execution
from optparse import OptionParser
parser = OptionParser()
parser.add_option( "--image",
                  action="store", dest="image", default=None,
                  help="anatomy image", metavar="FILE")
parser.add_option( "--gmm",
                  action="store", dest="gmm", default=None,
                  help="gmm image", metavar="FILE")
parser.add_option( "--outputdir",
                  action="store", dest="outputdir", default='.',
                  help="output path image", metavar="FILE")
(options, args) = parser.parse_args()


if (options.image != None and options.gmm != None ):
    imagedata = nib.load(options.image)
    numpyimage= imagedata.get_data().astype(IMG_DTYPE )
    print numpyimage.shape
    # error check
    assert numpyimage.shape[0:2] == (_globalexpectedpixel,_globalexpectedpixel)

    normalizedimage = options.image.replace('Ven.raw.nii.gz','Ven.normalize.nii.gz')
    getHeaderCmd = 'c3d %s -replace 2 1 3 1 4 0  -centroid | grep CENTROID_VOX | sed "s/CENTROID_VOX\ //g"' % (options.gmm)
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
    roi = np.array([ int(centroid[0]) - npixel/2. , int(centroid[0]) + npixel/2. , 
                     int(centroid[1]) - npixel/2. , int(centroid[1]) + npixel/2. ] )
    print roi 

    bndroi = np.array( [max(0, roi[0]) , 
                        min(_globalexpectedpixel, roi[1]) , 
                        max(0, roi[2]) , 
                        min(_globalexpectedpixel, roi[3]) ] )
    print bndroi 

    bnddiff  = roi - bndroi 
    print bnddiff 

    imgbndboxcmd = 'c3d %s -region %dx%dx0vox %dx%dx%dvox -pad %dx%dx0vox %dx%dx0vox 0 -info -o %s/Ven.roi.nii.gz ' % (options.image,bndroi[0],bndroi[2],bndroi[1]-bndroi[0],bndroi[3]-bndroi[2],numpyimage.shape[2],np.abs(bnddiff[0]),np.abs(bnddiff[1]),np.abs(bnddiff[2]),np.abs(bnddiff[3]),options.outputdir )
    print imgbndboxcmd 
    os.system( imgbndboxcmd )

    lblbndboxcmd = 'c3d %s -region %dx%dx0vox %dx%dx%dvox -pad %dx%dx0vox %dx%dx0vox 0 -info -type uchar -o %s/Truthroi.nii.gz ' % (options.gmm,bndroi[0],bndroi[2],bndroi[1]-bndroi[0],bndroi[3]-bndroi[2],numpyimage.shape[2],np.abs(bnddiff[0]),np.abs(bnddiff[1]),np.abs(bnddiff[2]),np.abs(bnddiff[3]),options.outputdir )
    print lblbndboxcmd 
    os.system( lblbndboxcmd )

else:
  parser.print_help()
  print options
 
