import pandas as pd 


def modify_isic_2016():
    df = pd.read_csv("isic2016/isic2016_train.csv")
    print(df.columns)

    df["label"] = df["label"].replace({"benign": 0, "malignant": 1})

    df.to_csv("isic2016/isic2016_train_modified.csv", index=False)


def modify_isic_2017():
    df = pd.read_csv("isic2017/isic2017_test.csv")
    new_df = pd.DataFrame(columns=["img", "label"])

    for _, line in df.iterrows():
        print(line)
        img, mel, sk = line
        img_id = img
        if mel == 1:
            label = 0
        elif sk == 1:
            label = 1
        else:
            label = 2
    
        new_df = new_df.append({"img": img_id, "label": label}, ignore_index=True)
    
    new_df.to_csv("isic2017/isic2017_test_modified.csv", index=False)


def modify_isic_2018():
    df = pd.read_csv("isic2019/isic2019_train.csv")
    new_df = pd.DataFrame(columns=["img", "label"])
    
    for _, line in df.iterrows():
        print(line)
        line = [item for item in line]
        img = line[0]
        one_hot = line[1:]
        label = one_hot.index(1)
        new_df = new_df.append({"img": img, "label": label}, ignore_index=True)
        
    new_df.to_csv("isic2019/isic2019_train_modified.csv", index=False)
    


modify_isic_2018()
    