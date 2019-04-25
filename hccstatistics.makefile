SHELL := /bin/bash
ROOTDIR=/rsrch1/ip/dtfuentes/github/livermask
#-include $(ROOTDIR)/hcckfold005.makefile
#-include $(ROOTDIR)/hccnormkfold005.makefile
#-include $(ROOTDIR)/hccvolkfold005.makefile
#-include $(ROOTDIR)/hccvolnorm512kfold005.makefile
-include $(ROOTDIR)/hccvolnorm256kfold005.makefile
WORKDIR=$(TRAININGROOT)/ImageDatabase
DATADIR=$(TRAININGROOT)/datalocation/train
mask:        $(addprefix $(WORKDIR)/,$(addsuffix /unet/mask.nii.gz,$(UIDLIST)))
normalize:   $(addprefix $(WORKDIR)/,$(addsuffix /Ven.normalize.nii.gz,$(UIDLIST)))
combine:     $(addprefix $(DATADIR)/,$(addsuffix /TruthVen6.nii.gz,$(UIDLIST)))
labels:      $(addprefix $(WORKDIR)/,$(addsuffix /$(DATABASEID)/tumor.nii.gz,$(UIDLIST)))
labelsmrf:   $(addprefix $(WORKDIR)/,$(addsuffix /$(DATABASEID)/tumormrf.nii.gz,$(UIDLIST)))
labelsmedian:$(addprefix $(WORKDIR)/,$(addsuffix /$(DATABASEID)/tumormedian.nii.gz,$(UIDLIST)))
lstat:       $(addprefix    qastats/,$(addsuffix /lstat.sql,$(UIDLIST)))
overlap:     $(addprefix $(WORKDIR)/,$(addsuffix /$(DATABASEID)/overlap.sql,$(UIDLIST)))
overlappost: $(addprefix $(WORKDIR)/,$(addsuffix /$(DATABASEID)/overlapmrf.sql,$(UIDLIST))) $(addprefix $(WORKDIR)/,$(addsuffix /$(DATABASEID)/overlapmedian.sql,$(UIDLIST)))
reviewsoln:  $(addprefix $(WORKDIR)/,$(addsuffix /reviewsoln,$(UIDLIST)))
C3DEXE=/rsrch2/ip/dtfuentes/bin/c3d
# keep tmp files
.SECONDARY: 

$(WORKDIR)/%/unet/mask.nii.gz: 
	mkdir -p $(@D)
	python ./applymodel.py --predictimage=$(WORKDIR)/$*/Ven.raw.nii.gz --segmentation=$@

## intensity statistics
qastats/%/lstat.csv: 
	mkdir -p $(@D)
	$(C3DEXE) $(WORKDIR)/$*/Ven.raw.nii.gz  $(DATADIR)/$*/TruthVen1.nii.gz -lstat > $(@D)/lstat.txt &&  sed "s/^\s\+/$(subst /,\/,$*),TruthVen1.nii.gz,Ven.raw.nii.gz,/g;s/\s\+/,/g;s/LabelID/InstanceUID,SegmentationID,FeatureID,LabelID/g;s/Vol(mm^3)/Vol.mm.3/g;s/Extent(Vox)/ExtentX,ExtentY,ExtentZ/g" $(@D)/lstat.txt > $@

qastats/%/lstat.sql: qastats/%/lstat.csv
	-sqlite3 $(SQLITEDB)  -init .loadcsvsqliterc ".import $< lstat"

$(DATADIR)/%/TruthVen6.nii.gz:
	c3d -verbose $(@D)/TruthVen1.nii.gz -replace 3 2 4 3 5 4 -o $@
	
$(WORKDIR)/%/Ven.normalize.nii.gz:
	python ./tissueshift.py --image=$(@D)/Ven.raw.nii.gz --gmm=$(DATADIR)/$*/TruthVen1.nii.gz  

$(WORKDIR)/%/$(DATABASEID)/tumormrf.nii.gz:
	c3d -verbose $(@D)/tumor-1.nii.gz -scale .5 $(@D)/tumor-[2345].nii.gz -vote-mrf  VA .1 -o $@

$(WORKDIR)/%/$(DATABASEID)/tumormedian.nii.gz:
	c3d -verbose $(@D)/tumor.nii.gz -median 1x1x1 -o $@

$(WORKDIR)/%/$(DATABASEID)/overlapmrf.csv: $(WORKDIR)/%/$(DATABASEID)/tumormrf.nii.gz
	$(C3DEXE) $(DATADIR)/$*/TruthVen1.nii.gz  -as A $< -as B -overlap 1 -overlap 2 -overlap 3 -overlap 4  -overlap 5  > $(@D)/overlap.txt
	grep "^OVL" $(@D)/overlap.txt  |sed "s/OVL: \([0-9]\),/\1,$(subst /,\/,$*),/g;s/OVL: 1\([0-9]\),/1\1,$(subst /,\/,$*),/g;s/^/TruthVen1.nii.gz,tumormrf,/g;"  | sed "1 i FirstImage,SecondImage,LabelID,InstanceUID,MatchingFirst,MatchingSecond,SizeOverlap,DiceSimilarity,IntersectionRatio" > $@

$(WORKDIR)/%/overlapmrf.sql: $(WORKDIR)/%/overlapmrf.csv
	-sqlite3 $(SQLITEDB)  -init .loadcsvsqliterc ".import $< overlap"

## dice statistics
$(WORKDIR)/%/$(DATABASEID)/overlap.csv: $(WORKDIR)/%/$(DATABASEID)/tumor.nii.gz
	mkdir -p $(@D)
	$(C3DEXE) $<  -as A $(DATADIR)/$*/TruthVen1.nii.gz -as B -overlap 1 -overlap 2 -overlap 3 -overlap 4  -thresh 2 3 1 0 -comp -as C  -clear -push C -replace 0 255 -split -pop -foreach -push B -multiply -insert A 1 -overlap 1 -overlap 2 -overlap 3 -overlap 4 -pop -endfor
	grep "^OVL" $(@D)/overlap.txt  |sed "s/OVL: \([0-9]\),/\1,$(subst /,\/,$*),/g;s/OVL: 1\([0-9]\),/1\1,$(subst /,\/,$*),/g;s/^/TruthVen1.nii.gz,$(DATABASEID)\/tumor.nii.gz,/g;"  | sed "1 i FirstImage,SecondImage,LabelID,InstanceUID,MatchingFirst,MatchingSecond,SizeOverlap,DiceSimilarity,IntersectionRatio" > $@

$(WORKDIR)/%/overlap.sql: $(WORKDIR)/%/overlap.csv
	-sqlite3 $(SQLITEDB)  -init .loadcsvsqliterc ".import $< overlap"

$(WORKDIR)/%/reviewsoln: 
	vglrun itksnap -g $(WORKDIR)/$*/Ven.raw.nii.gz -s $(DATADIR)/$*/TruthVen1.nii.gz & vglrun itksnap -g $(WORKDIR)/$*/Ven.raw.nii.gz -s $(WORKDIR)/$*/$(DATABASEID)/tumor.nii.gz ;\
        pkill -9 ITK-SNAP



