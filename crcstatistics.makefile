SHELL := /bin/bash
ROOTDIR=/rsrch1/ip/dtfuentes/github/livermask
WORKDIR=ImageDatabase
DATADIR=$(TRAININGROOT)/datalocation/train
-include $(ROOTDIR)/crckfold005.makefile
setup:      $(addprefix $(WORKDIR)/,$(addsuffix /unetadadelta/setup,$(UIDLIST)))  $(addprefix $(WORKDIR)/,$(addsuffix /unetRMSprop/setup,$(UIDLIST4)))
mask:       $(addprefix $(WORKDIR)/,$(addsuffix /unetadadelta/mask.nii.gz,$(UIDLIST)))
labels:     $(addprefix $(WORKDIR)/,$(addsuffix /unetadadelta/tumor.nii.gz,$(UIDLIST)))
mrf:        $(addprefix $(WORKDIR)/,$(addsuffix /unetadadelta/tumormrf.nii.gz,$(UIDLIST)))
lstat:      $(addprefix $(WORKDIR)/,$(addsuffix /unetadadelta/lstat.sql,$(UIDLIST)))
overlap:    $(addprefix $(WORKDIR)/,$(addsuffix /unetadadelta/overlap.sql,$(UIDLIST)))   $(addprefix $(WORKDIR)/,$(addsuffix /unetRMSProp/overlap.sql,$(UIDLIST4)))
overlapmrf: $(addprefix $(WORKDIR)/,$(addsuffix /unetadadelta/overlapmrf.sql,$(UIDLIST)))
C3DEXE=/rsrch2/ip/dtfuentes/bin/c3d
# keep tmp files
.SECONDARY: 

$(WORKDIR)/%/mask.nii.gz: 
	mkdir -p $(@D)
	python ./applymodel.py --predictimage=$(WORKDIR)/$*/image.nii  --segmentation=$@

$(WORKDIR)/%/tumor.nii.gz: $(WORKDIR)/%/image.nii $(WORKDIR)/%/mask.nii.gz $(WORKDIR)/%/tumormodelunet.json
	python ./applymodel.py --predictimage=$< --modelpath=$(word 3, $^) --maskimage=$(word 2, $^) --segmentation=$@

## intensity statistics
$(WORKDIR)/%/lstat.csv: 
	mkdir -p $(@D)
	$(C3DEXE) $(WORKDIR)/$*/image.nii  $(WORKDIR)/$*/label.nii -lstat > $(@D)/lstat.txt &&  sed "s/^\s\+/$(subst /,\/,$*),TruthVen1.nii.gz,Ven.raw.nii.gz,/g;s/\s\+/,/g;s/LabelID/InstanceUID,SegmentationID,FeatureID,LabelID/g;s/Vol(mm^3)/Vol.mm.3/g;s/Extent(Vox)/ExtentX,ExtentY,ExtentZ/g" $(@D)/lstat.txt > $@

qastats/%/lstat.sql: qastats/%/lstat.csv
	-sqlite3 $(SQLITEDB)  -init .loadcsvsqliterc ".import $< lstat"

$(WORKDIR)/%/unethcc/tumormrf.nii.gz:
	c3d -verbose $(@D)/tumor-1.nii.gz -scale .5 $(@D)/tumor-[2345].nii.gz -vote-mrf  VA .1 -o $@

$(WORKDIR)/%/unethcc/overlapmrf.csv: $(WORKDIR)/%/unethcc/tumormrf.nii.gz
	$(C3DEXE) $(DATADIR)/$*/TruthVen1.nii.gz  -as A $< -as B -overlap 1 -overlap 2 -overlap 3 -overlap 4  -overlap 5  > $(@D)/overlap.txt
	grep "^OVL" $(@D)/overlap.txt  |sed "s/OVL: \([0-9]\),/\1,$(subst /,\/,$*),/g;s/OVL: 1\([0-9]\),/1\1,$(subst /,\/,$*),/g;s/^/TruthVen1.nii.gz,tumormrf,/g;"  | sed "1 i FirstImage,SecondImage,LabelID,InstanceUID,MatchingFirst,MatchingSecond,SizeOverlap,DiceSimilarity,IntersectionRatio" > $@

$(WORKDIR)/%/overlapmrf.sql: $(WORKDIR)/%/overlapmrf.csv
	-sqlite3 $(SQLITEDB)  -init .loadcsvsqliterc ".import $< overlap"

## dice statistics
$(WORKDIR)/%/overlap.csv: $(WORKDIR)/%/tumor.nii.gz
	mkdir -p $(@D)
	$(C3DEXE) $(WORKDIR)/$*/label.nii  -as A $< -as B -overlap 1 -overlap 2 -overlap 3 -overlap 4  -overlap 5  > $(@D)/overlap.txt
	grep "^OVL" $(@D)/overlap.txt  |sed "s/OVL: \([0-9]\),/\1,$(subst /,\/,$*),/g;s/OVL: 1\([0-9]\),/1\1,$(subst /,\/,$*),/g;s/^/label.nii,tumor.nii.gz,/g;"  | sed "1 i FirstImage,SecondImage,LabelID,InstanceUID,MatchingFirst,MatchingSecond,SizeOverlap,DiceSimilarity,IntersectionRatio" > $@

$(WORKDIR)/%/overlap.sql: $(WORKDIR)/%/overlap.csv
	-sqlite3 $(SQLITEDB)  -init .loadcsvsqliterc ".import $< overlap"
