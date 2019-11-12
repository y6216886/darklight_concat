from load_clock_label import *

def overWriteit(path):
    df = pd.read_csv(path, index_col='eyeId')
    print(df)
    print(df.loc["C2-001-5", "od_left"])
    df.loc["C2-001-5", "od_left"]= 99
    print(df)
if __name__ == '__main__':
    overWriteit("I:\octdata\\brightVsDark_label\\eyeId_label.csv")
