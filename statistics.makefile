SHELL := /bin/bash
ROOTDIR=/rsrch1/ip/dtfuentes/github/livermask
-include $(ROOTDIR)/hcckfold005.makefile
WORKDIR=$(TRAININGROOT)/ImageDatabase
DATADIR=$(TRAININGROOT)/datalocation/train
mask:   $(addprefix $(WORKDIR)/,$(addsuffix /unet/mask.nii.gz,$(UIDLIST)))
labels: $(addprefix $(WORKDIR)/,$(addsuffix /unethcc/tumor.nii.gz,$(UIDLIST)))
overlap:  $(addprefix $(WORKDIR)/,$(addsuffix /unethcc/overlap.csv,$(UIDLIST)))

$(WORKDIR)/%/unet/mask.nii.gz: 
	mkdir -p $(@D)
	python ./applymodel.py --predictimage=$(WORKDIR)/$*/Ven.raw.nii.gz --segmentation=$@

## dice statistics
$(WORKDIR)/%/unethcc/overlap.csv: $(WORKDIR)/%/unethcc/tumor.nii.gz
	mkdir -p $(@D)
	$(C3DEXE) $(DATADIR)/$*/TruthVen1.nii.gz  -as A $< -as B -overlap 1 -overlap 2 -overlap 3 -overlap 4  -thresh 3 4 1 0 -comp -clip 1 10 -split -foreach -push B -multiply -insert A 1 -overlap 1 -overlap 2 -overlap 3 -overlap 4 -pop -endfor > $(@D)/overlap.txt
	grep "^OVL" $(@D)/overlap.txt  |sed "s/OVL: \([0-9]\),/\1,$(MODELID),/g;s/OVL: 1\([0-9]\),/1\1,$(MODELID),/g;s/^/$(subst /,,$*),Truth.nii.gz,LABELSLEGACY.nii.gz,/g;" | awk '{print int((ln++)/4)","  $$0 }'  | sed "1 i compid,InstanceUID,FirstImage,SecondImage,LabelID,SegmentationID,MatchingFirst,MatchingSecond,SizeOverlap,DiceSimilarity,IntersectionRatio" > $@

