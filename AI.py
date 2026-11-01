# imports
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas
import random
from collections import Counter

#print(sklearn.__version__)
#print(pandas.__version__)

# file handling
f = open ("training_data.csv", "r")
df = pandas.read_csv("training_data.csv")
# Switch over when needed
#f = open ("test_data.csv", "r")
#df = pandas.read_csv("test_data.csv")

# Brackets to define rules
df["AgeBracket"] = ["21-" if v < 22 else "22-44" if v < 44 else "45+"
                    for v in df["Age"]]
df["DurationBracket"] = ["12-" if v < 13 else "13-24" if v < 25 else "25+"
                    for v in df["Duration"]]
df["CreditBracket"] = ["Small" if v < 1351 else "Mid" if v < 4499 else "Large"
                    for v in df["Credit amount"]]
df["JobCount"] = ["Unemployed" if v < 1 else "Employed" if v < 1 else "Multiple"
                    for v in df["Age"]]

# Rules, trying to predict credit risk
# Most complex rules first due to FF con res
df_rules = [
    # 2 condition rules
    ({"Saving accounts=rich","Purpose=furniture/equipment"},"Risk=good"),
    ({"Saving accounts=rich","Housing=free"},"Risk=bad"),
    ({"Saving accounts=moderate","Housing=rent"},"Risk=bad"),
    ({"Saving accounts=moderate","Housing=own"},"Risk=good"),
    ({"Saving accounts=little","Housing=rent"},"Risk=bad"),
    ({"DurationBracket=13-24","CreditBracket=Mid"},"Risk=good"),
    ({"Purpose=radio/TV","JobCount=Multiple"},"Risk=good"),
    ({"DurationBracket=25+"},"Risk=bad"), 

    # 1 condition rules
    ({"AgeBracket=21-"},"Risk=bad"),
    ({"Saving accounts=quite rich"},"Risk=good"),
    ({"Saving accounts=rich"},"Risk=good"),
    ({"Checking account=rich"},"Risk=good"),
    ({"CreditBracket=Large"},"Risk=bad"),
    ({"Purpose=vacation/others"},"Risk=bad"),
    ({"Checking account=moderate"},"Risk=bad"),
    ({"Checking account=little"},"Risk=bad"),
    ({"DurationBracket=12-"},"Risk=good"),
    #({""},"Risk="), 
]

# Prediction
# Default does not seem to change accuracy with current rules
def predict_risk(row, rules, default="risk=good"):
    """First rule in the list that fires wins (conflict resolution)."""
    facts = {f"{a}={v}" for a, v in row.items()}
    for conditions, conclusion in rules:        
        if conditions <= facts:
            return conclusion
    return default

rows = df.to_dict("records")


# prints results, intended for use with test data, should work with both
#print([predict_risk(r,df_rules) for r in rows])

# Test prediction
#print(predict_risk({},df_rules) for r in rows)
# Test code to print given field data
#print(df.groupby("")[[""]].describe())

# Comment away all code below  when running test data, no risk field to use, crashes

# Find accuracy
accuracy = sum(predict_risk(r, df_rules) == f"Risk={r['Risk']}" for r in rows) / len(rows)
print(f"Whole set Accuracy: {accuracy:.2f}")

# Baseline for comparison
majority = Counter(r["Risk"] for r in rows).most_common(1)[0][0]
baseline = sum(r["Risk"] == majority for r in rows) / len(rows)
print(f"Baseline (always predict {majority!r}): {baseline:.2f}")

# Setting up evaluation
rows = [dict(r, **{"class": f"Risk={r['Risk']}"}) for r in rows]
classify = lambda r: predict_risk(r, df_rules)

# splits data for evaluation
random.seed(82)
shuffled = rows[:]
random.shuffle(shuffled)
cut = int(0.8 * len(shuffled))
train, test = shuffled[:cut], shuffled[cut:]

y_true = [r["class"] for r in test]
y_pred = [classify(r) for r in test]

# evaluation
print(f"Split set Accuracy: {accuracy_score(y_true, y_pred):.2f}")
print(classification_report(y_true, y_pred, zero_division=0))
print(confusion_matrix(y_true, y_pred, labels=sorted(set(y_true))))

# close files
f.close