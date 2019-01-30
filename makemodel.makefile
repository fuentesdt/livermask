# Segmentation commands

all : build train cross validate

build : 
	python3 liver2.py --builddb --dbfile=$(DBFILE)

train : 
	python3 liver2.py --dbfile=$(DBFILE) --numepochs=$(NUMEPOCHS) --kfolds=1 --idfold=0 --trainmodel --outdir=$(OUTDIR) 

cross : 
	number=0 ; while [[ $$number -lt $(KFOLDS) ]] ; do \
	       python3 liver2.py --dbfile=$(DBFILE) --numepochs=$(NUMEPOCHS) --kfolds=$(KFOLDS) --idfold=$$number --trainmodel --outdir=$(OUTDIR) ; \
		((number = number + 1)) ; \
	done

validate : 
	python3 liver2.py --dbfile=$(DBFILE) --numepochs=$(NUMEPOCHS) --kfolds=$(KFOLDS) --setuptestset --outdir=$(OUTDIR)
