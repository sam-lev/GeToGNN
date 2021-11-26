import os

from localsetup import LocalSetup
LocalSetup = LocalSetup()

def write_model_scores(model, scoring, scoring_dict, out_folder, threshold=''):
    scoring_file = os.path.join(out_folder, scoring+threshold+'.txt')
    print("... Writing scoring file to:", scoring_file)
    scoring_file = open(scoring_file, "w+")
    for mode in scoring_dict.keys():
        score = scoring_dict[mode]
        scoring_file.write(mode+ ' ' + str(score) + "\n")
    scoring_file.close()