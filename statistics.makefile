SHELL := /bin/bash
ROOTDIR=/rsrch1/ip/dtfuentes/github/livermask
-include $(ROOTDIR)/hcckfold005.makefile
WORKDIR=$(TRAININGROOT)/ImageDatabase
DATADIR=$(TRAININGROOT)/datalocation/train
mask:   $(addprefix $(WORKDIR)/,$(addsuffix /unet/mask.nii.gz,$(UIDLIST)))
labels: $(addprefix $(WORKDIR)/,$(addsuffix /unethcc/tumor.nii.gz,$(UIDLIST)))
overlap:  $(addprefix $(WORKDIR)/,$(addsuffix /unethcc/overlap.csv,$(UIDLIST)))
C3DEXE=/rsrch2/ip/dtfuentes/bin/c3d

$(WORKDIR)/%/unet/mask.nii.gz: 
	mkdir -p $(@D)
	python ./applymodel.py --predictimage=$(WORKDIR)/$*/Ven.raw.nii.gz --segmentation=$@

## dice statistics
$(WORKDIR)/%/unethcc/overlap.csv: $(WORKDIR)/%/unethcc/tumor.nii.gz
	mkdir -p $(@D)
	$(C3DEXE) $(DATADIR)/$*/TruthVen1.nii.gz  -as A $< -as B -overlap 1 -overlap 2 -overlap 3 -overlap 4  -overlap 5  > $(@D)/overlap.txt
	grep "^OVL" $(@D)/overlap.txt  |sed "s/OVL: \([0-9]\),/\1,$(subst /,\/,$*),/g;s/OVL: 1\([0-9]\),/1\1,$(subst /,\/,$*),/g;s/^/TruthVen1.nii.gz,unethcc\/tumor.nii.gz,/g;"  | sed "1 i FirstImage,SecondImage,LabelID,InstanceUID,MatchingFirst,MatchingSecond,SizeOverlap,DiceSimilarity,IntersectionRatio" > $@

$(WORKDIR)/%.sql: $(WORKDIR)/%.csv
	sqlite3 $(SQLITEDB)  -init .loadcsvsqliterc ".import $< overlap"
