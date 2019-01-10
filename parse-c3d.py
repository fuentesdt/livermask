import subprocess as sbp
import re
import csv
import json
import os.path

def display_command_result(r):
    print("\nResult args: \n", r.args)
    print("\nResult return code: \n", r.returncode)
    print("\nResult stdout: \n", r.stdout)
    print("\nResult stderr: \n", r.stderr)

regex0 = re.compile('\n+')
regex1 = re.compile(r', |: |\s')
regex2 = re.compile(r': |; ')
regex3 = re.compile(r'=')

imgname   = "/rsrch1/ip/dtfuentes/SegmentationTrainingData/LiTS2017/LITS/TrainingBatch2/volume-30.nii"
truthname = "/rsrch1/ip/dtfuentes/SegmentationTrainingData/LiTS2017/LITS/TrainingBatch2/segmentation-30.nii"
modelname = "tblog/dscimg/full/adadelta/256/run_a/004/005/000/tumormodelunet.json"
segname   = "out.nii.gz"
lblname   = "mylabel.nii.gz"

# command0 = ["python3", "liver.py", "--predictimage="+imgname, "--predictmodel="+modelname, "--segmentation="+segname]
# command1 = ["c3d", segname.replace(".nii.gz", "0.nii.gz"), "-info"]
# command2 = ["c3d", segname.replace(".nii.gz", "?.nii.gz"), "-vote", "-type", "uchar", "-o", lblname]
# command3 = ["c3d", "-verbose", lblname, truthname, "-overlap", "1", "-overlap", "2", "-overlap", "0"]
command1 = ["cat", "sample_info.txt"]
command3 = ["cat", "sample_out.txt"]

writechar = 'a' if os.path.isfile('stats.csv') else 'w+'
with open(r'stats.csv', writechar) as f:
    writer = csv.writer(f)
    if writechar == 'w+':
        writer.writerow(["imgname", "lblname", "truthname", "modelname", "dim", "bb", "vox", "range", "orient", "label", "matching_vox_img1", "matching_vox_img2", "size_overlap", "dice", "jaccard"])

    # result0 = sbp.run(command0)
    result1 = sbp.run(command1, stdout=sbp.PIPE, stderr=sbp.PIPE, universal_newlines=True)
    # result2 = sbp.run(command2)
    result3 = sbp.run(command3, stdout=sbp.PIPE, stderr=sbp.PIPE, universal_newlines=True)

    lines1 = regex0.split(result1.stdout)
    data = []
    for l in lines1:
        entries = regex2.split(l)
        for e in entries:
            e = e.strip()
            if re.search(regex3, e):
                values = regex3.split(e)
                v = values[1].strip()
                if v.startswith("["):
                    v_list = json.loads(v)
                    data = data + [v_list]
                if v.startswith("{"):
                    v = v.replace("[", "\"[")
                    v = v.replace("]", "]\"")
                    v = v.replace("{", "[")
                    v = v.replace("}", "]")
                    v = v.replace("\'", "\"")
                    vvals = json.loads(v)
                    v_array = []
                    for vv in vvals:
                        vv.strip()
                        vv = vv.replace(" ", ",")
                        vv = json.loads(vv)
                        v_array = v_array + [vv]
                    data = data + [v_array]
                if values[0].strip().startswith("orient"):
                    v_str = v
                    data = data + [v_str]

    lines2 = regex0.split(result3.stdout)
    for l in lines2:
        if l.startswith("OVL"):
            values = regex1.split(l)
            row = [imgname, lblname, truthname, modelname] + data + values[1:]
            writer.writerow(row)
