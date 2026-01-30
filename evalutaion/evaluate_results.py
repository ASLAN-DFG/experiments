import argparse
from extrinsic_metrics import *

def main(args):
    input_df = pd.read_csv(args.input)
    cm = contingency_matrix(input_df, "labels", "Score1")

    with open("out/contingency_matrix.txt", "w") as f:
        f.write(str(cm))

    with open("out/eval_scores.txt", "w") as f:
        purity = purity_score(cm)
        inv_purity = purity_score(cm, inverse=True)
        f.write(f"{purity = :.{3}} \n")
        f.write(f"{inv_purity = :.{3}} \n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="the clustering to be evaluated", default="out/clustering_result.csv")
    args = parser.parse_args()
    main(args)