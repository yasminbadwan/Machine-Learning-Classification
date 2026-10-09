import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.datasets import load_iris


# ============================================================
# Create PlayTennis Dataset
# ============================================================
def create_dataset():
    data = {
        'Outlook':  ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast',
                     'Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
        'Temp':     ['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                     'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
        'Humidity': ['High','High','High','High','Normal','Normal','Normal',
                     'High','Normal','Normal','Normal','High','Normal','High'],
        'Wind':     ['Weak','Strong','Weak','Weak','Weak','Strong','Strong',
                     'Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
        'Tennis':   ['No','No','Yes','Yes','Yes','No','Yes',
                     'No','Yes','Yes','Yes','Yes','Yes','No']
    }
    return pd.DataFrame(data)


# ============================================================
# General Printing Functions
# ============================================================
def print_title(title):
    print("\n" + "=" * 75)
    print(title.center(75))
    print("=" * 75)


def print_subtitle(title):
    print("\n" + "-" * 75)
    print(title.center(75))
    print("-" * 75)


def print_classifier_results(title, actual, predicted, labels):
    print_subtitle(title)

    accuracy = accuracy_score(actual, predicted)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )

    print(f"Accuracy: {accuracy:.3f}  ({accuracy * 100:.1f}%)")

    print("\nConfusion Matrix:\n")
    cm = confusion_matrix(actual, predicted, labels=labels)

    header = " " * 20
    for label in labels:
        header += f"Pred {label:<15}"
    print(header)

    for i, label in enumerate(labels):
        row = f"Actual {label:<13}"
        for value in cm[i]:
            row += f"{value:^20}"
        print(row)

    print("\nClassification Report:\n")
    print("              precision    recall    f1-score")
    for i, label in enumerate(labels):
        print(f"{label:<15}{precision[i]:>8.2f}{recall[i]:>10.2f}{f1[i]:>12.2f}")
    print(f"{'Accuracy':<35}{accuracy:>8.2f}")

    metrics = {
        'accuracy':          accuracy,
        'average_precision': sum(precision) / len(precision),
        'average_recall':    sum(recall)    / len(recall),
        'average_f1':        sum(f1)        / len(f1)
    }
    return metrics


# ============================================================
# LOOCV — Decision Tree
# ============================================================
def loocv_decision_tree(df):
    feature_cols = ['Outlook', 'Temp', 'Humidity', 'Wind']
    target_col   = 'Tennis'

    loo = LeaveOneOut()
    X   = df[feature_cols].values
    y   = df[target_col].values

    all_actual    = []
    all_predicted = []

    fold_details = []

    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
  
        train_df_fold = df.iloc[train_idx].copy()
        test_df_fold  = df.iloc[test_idx].copy()

       
        encoders = {}
        train_enc = train_df_fold.copy()
        test_enc  = test_df_fold.copy()

        for col in df.columns:
            le = LabelEncoder()
            train_enc[col] = le.fit_transform(train_df_fold[col])
            # handle unseen values in test gracefully
            test_enc[col]  = [
                le.transform([v])[0] if v in le.classes_
                else -1
                for v in test_df_fold[col]
            ]
            encoders[col] = le

        X_train = train_enc[feature_cols].values
        y_train = train_enc[target_col].values
        X_test  = test_enc[feature_cols].values
        y_test  = test_enc[target_col].values

       
        model = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=4,
            random_state=96
        )
        model.fit(X_train, y_train)
        pred_enc = model.predict(X_test)

     
        actual_label    = encoders[target_col].inverse_transform(y_test)[0]
        predicted_label = (
            encoders[target_col].inverse_transform(pred_enc)[0]
            if pred_enc[0] != -1
            else "Unknown"
        )

        all_actual.append(actual_label)
        all_predicted.append(predicted_label)

        correct = "Correct" if actual_label == predicted_label else "fail]"
        fold_details.append({
            'Fold':      fold + 1,
            'Test Row':  test_idx[0] + 1,
            'Actual':    actual_label,
            'Predicted': predicted_label,
            'Correct':   correct
        })

    return all_actual, all_predicted, fold_details


# ============================================================
# LOOCV — Naive Bayes
# ============================================================
def loocv_naive_bayes(df):

    feature_cols = ['Outlook', 'Temp', 'Humidity', 'Wind']
    target_col   = 'Tennis'

    loo = LeaveOneOut()
    X   = df[feature_cols].values
    y   = df[target_col].values

    all_actual    = []
    all_predicted = []

    fold_details = []

    for fold, (train_idx, test_idx) in enumerate(loo.split(X)):
        train_df_fold = df.iloc[train_idx]
        test_df_fold  = df.iloc[test_idx]

       
        ord_enc = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        X_train = ord_enc.fit_transform(train_df_fold[feature_cols])
        X_test  = ord_enc.transform(test_df_fold[feature_cols])

        le      = LabelEncoder()
        y_train = le.fit_transform(train_df_fold[target_col])
        y_test  = le.transform(test_df_fold[target_col])

      
        model = CategoricalNB()
        model.fit(X_train, y_train)
        pred_enc = model.predict(X_test)

        actual_label    = le.inverse_transform(y_test)[0]
        predicted_label = le.inverse_transform(pred_enc)[0]

        all_actual.append(actual_label)
        all_predicted.append(predicted_label)

        correct = "Correct" if actual_label == predicted_label else "fail"
        fold_details.append({
            'Fold':      fold + 1,
            'Test Row':  test_idx[0] + 1,
            'Actual':    actual_label,
            'Predicted': predicted_label,
            'Correct':   correct
        })

    return all_actual, all_predicted, fold_details




# ============================================================
# Visualize Comparison
# ============================================================
def visualize_model_comparison(dt_metrics, nb_metrics):
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']

    dt_scores = [
        dt_metrics['accuracy'],
        dt_metrics['average_precision'],
        dt_metrics['average_recall'],
        dt_metrics['average_f1']
    ]
    nb_scores = [
        nb_metrics['accuracy'],
        nb_metrics['average_precision'],
        nb_metrics['average_recall'],
        nb_metrics['average_f1']
    ]

    x         = range(len(metrics_names))
    bar_width  = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - bar_width/2 for i in x], dt_scores, width=bar_width,
            label='Decision Tree', color="#4E6FF1")
    plt.bar([i + bar_width/2 for i in x], nb_scores, width=bar_width,
            label='Naive Bayes',   color="#58AABE")

    plt.title("Decision Tree vs Naive Bayes — LOOCV Comparison", fontsize=14)
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.xticks(x, metrics_names)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    for i, (d, n) in enumerate(zip(dt_scores, nb_scores)):
        plt.text(i - bar_width/2, d + 0.02, f"{d:.2f}", ha='center', fontsize=9)
        plt.text(i + bar_width/2, n + 0.02, f"{n:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig("loocv_model_comparison_chart.png", dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Detailed Comparison
# ============================================================
def print_detailed_comparison(dt_metrics, nb_metrics):
    print_title("LOOCV — Classifier Comparison")

    print(f"{'Metric':<25}{'Decision Tree':<18}{'Naive Bayes':<18}{'Better Model'}")
    print("-" * 75)

    def better(a, b):
        return "Decision Tree" if a > b else ("Naive Bayes" if b > a else "Equal")

    rows = [
        ("Accuracy",      dt_metrics['accuracy'],           nb_metrics['accuracy']),
        ("Avg Precision", dt_metrics['average_precision'],  nb_metrics['average_precision']),
        ("Avg Recall",    dt_metrics['average_recall'],     nb_metrics['average_recall']),
        ("Avg F1-Score",  dt_metrics['average_f1'],         nb_metrics['average_f1']),
    ]
    for name, dv, nv in rows:
        print(f"{name:<25}{dv:<18.3f}{nv:<18.3f}{better(dv, nv)}")
    print("-" * 75)

    if dt_metrics['accuracy'] > nb_metrics['accuracy']:
        print("Overall Winner: Decision Tree")
    elif nb_metrics['accuracy'] > dt_metrics['accuracy']:
        print("Overall Winner: Naive Bayes")
    else:
        print("Overall Winner: Tied on Accuracy")

    visualize_model_comparison(dt_metrics, nb_metrics)


# ============================================================
# Part C: UCI Dataset — Iris (train/test split 80/20)
# ============================================================
def visualize_general_tree(model, feature_names, class_names, title, image_name):
    plt.figure(figsize=(20, 10), dpi=150)

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=False,
        rounded=True,
        fontsize=7,
        impurity=False,
        proportion=False,
        precision=2
    )

    ax = plt.gca()
    colors = {
        'setosa':     "#8689D6",
        'versicolor': "#4E6FF1",
        'virginica':  "#58AABE"
    }

    for node in ax.texts:
        text = node.get_text()
        box  = node.get_bbox_patch()
        if box is None:
            continue
        matched = False
        for cls, color in colors.items():
            if f"class = {cls}" in text:
                box.set_facecolor(color)
                matched = True
                break
        if not matched:
            box.set_facecolor("#F5F5F5")
        box.set_edgecolor("black")
        box.set_linewidth(1.5)

    plt.title(title, fontsize=20, pad=35)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)
    plt.savefig(image_name, dpi=300, bbox_inches="tight", pad_inches=0.6)
    plt.show()


# ============================================================
# PlayTennis Tree 
# ============================================================
def visualize_playtennis_tree(df):
    feature_cols = ['Outlook', 'Temp', 'Humidity', 'Wind']
    target_col   = 'Tennis'

    encoders  = {}
    df_enc    = df.copy()

    for col in df.columns:
        le          = LabelEncoder()
        df_enc[col] = le.fit_transform(df[col])
        encoders[col] = le

    X           = df_enc[feature_cols].values
    y           = df_enc[target_col].values
    class_names = list(encoders[target_col].classes_)

    model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=96)
    model.fit(X, y)

    plt.figure(figsize=(20, 10), dpi=150)

    plot_tree(
        model,
        feature_names=feature_cols,
        class_names=class_names,
        filled=False,
        rounded=True,
        fontsize=7,
        impurity=False,
        proportion=False,
        precision=2
    )

    ax = plt.gca()
    for node in ax.texts:
        text = node.get_text()
        box  = node.get_bbox_patch()
        if box is None:
            continue
        if "class = Yes" in text:
            box.set_facecolor("#4E6FF1")
        elif "class = No" in text:
            box.set_facecolor("#58AABE")
        else:
            box.set_facecolor("#F5F5F5")
        box.set_edgecolor("black")
        box.set_linewidth(1.5)

    plt.title("Decision Tree for PlayTennis Dataset", fontsize=20, pad=35)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)
    plt.savefig("playtennis_decision_tree_colored.png", dpi=300, bbox_inches="tight", pad_inches=0.6)
    plt.show()


def run_part_c_uci_dataset():
    print_title("Part C: UCI Dataset — Iris (80/20 Train/Test Split)")

    iris = load_iris()
    X, y = iris.data, iris.target

    clean_feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    target_names        = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=96, stratify=y
    )

    print(f"Number of Instances : {len(X)}")
    print(f"Number of Features  : {len(clean_feature_names)}")
    print(f"Features            : {', '.join(clean_feature_names)}")
    print(f"Target Classes      : {', '.join(target_names)}")
    print(f"Training Size       : {len(X_train)}")
    print(f"Testing Size        : {len(X_test)}")

    model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=96)
    model.fit(X_train, y_train)
    pred_encoded = model.predict(X_test)

    actual    = [target_names[i] for i in y_test]
    predicted = [target_names[i] for i in pred_encoded]

    visualize_general_tree(
        model=model,
        feature_names=clean_feature_names,
        class_names=target_names,
        title="Decision Tree for Iris Dataset",
        image_name="iris_decision_tree_colored.png"
    )

    iris_metrics = print_classifier_results(
        title="Part C — Decision Tree Results",
        actual=actual,
        predicted=predicted,
        labels=target_names
    )

    print_title("Part C Summary")
    print("Dataset    : Iris Dataset (UCI)")
    print("Classifier : Decision Tree")
    print("Split      : 80% Training / 20% Testing")
    print(f"Accuracy   : {iris_metrics['accuracy'] * 100:.1f}%")
    print(f"F1-Score   : {iris_metrics['average_f1']:.2f}")


# ============================================================
# Main
# ============================================================
def main():
    df             = create_dataset()
    target_classes = ['Yes', 'No']


    print_title("Part A: Decision Tree — PlayTennis")
    visualize_playtennis_tree(df)
    dt_actual, dt_predicted, dt_folds = loocv_decision_tree(df)
    dt_metrics = print_classifier_results(
        title="Decision Tree — Overall LOOCV Results",
        actual=dt_actual,
        predicted=dt_predicted,
        labels=target_classes
    )
    print_title("Part B: Naive Bayes on PlayTennis")
    nb_actual, nb_predicted, nb_folds = loocv_naive_bayes(df)

    nb_metrics = print_classifier_results(
        title="Naive Bayes — Overall LOOCV Results",
        actual=nb_actual,
        predicted=nb_predicted,
        labels=target_classes
    )

   
    print_detailed_comparison(dt_metrics, nb_metrics)

 
    run_part_c_uci_dataset()


main()