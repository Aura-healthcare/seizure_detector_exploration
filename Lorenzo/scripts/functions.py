from pathlib import Path
import pandas as pd
import json
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV


def load_data_res(path, features):
    '''
    features peut prendre la valeur pan-tompkins, swt, ou xqrs
    '''
    data = json.load(open(path, "r"))
    df = pd.DataFrame(columns = list(data[features]),
                        data = list(map(list, zip(*list(data[features].values())))))
    return df


def load_data(path):
    data = json.load(open(path, "r"))
    df = pd.DataFrame(columns = data['keys'], data = data['features'])
    return df


def load_and_concat_feats(directory, type):
    '''
    Directory : The path of the feats-v0_4 folder,
                could be relative or absolute path

    Type : Select which type of file to handle (pan, swt, xqrs)
    '''

    full_df = pd.DataFrame()

    for path in tqdm(Path(directory).rglob(f'*{type}.json')):
        data = json.load(open(path, "r"))
        df = pd.DataFrame(columns = data['keys'], data = data['features'])
        df['Set'] = path.parent.parent.parent.parent.parent.name
        df['Categorie Montage'] = path.parent.parent.parent.parent.name
        df['Dossier Patient'] = path.parent.parent.parent.name
        df['Patient'] = path.parent.parent.name
        df['Session'] = path.parent.name

        r = re.search(r't\d{3}', str(path.name))
        if r:
            df['File N°'] = r.group()

        full_df = full_df.append(df, ignore_index=True)

    cols = full_df.columns.tolist()
    cols = cols[-6:] + cols[:-6]
    full_df = full_df[cols]

    return full_df


def get_good_file(directory, threshold, error_log=False):
    '''
    Input : Path of res-v0_4 folder
            threshold for correl coefs
    .........................................................
    This function read all files from res-v0_4 folder and
    returns those with coefficients higher than the threshold
    .........................................................
    Return : 4 lists if error_log is True:
                -> One for the path of files with good coefs
                -> One for the key
                -> One for the coef
                -> One for the path of files with errors
             3 lists if error_log is False:
                -> path of files with good coefs
                -> One for the key
                -> One for the coef
    '''

    paths = []
    key = []
    coef = []
    error_file = []

    for path in tqdm(Path(directory).rglob(f'*.json')):
        try:
            data = json.load(open(path, "r"))

            df = pd.DataFrame(columns = ['Pan-Tompkins', 'SWT', 'XQRS'],
                                data = data['score']['corrcoefs'],
                                index=['Pan-Tompkins', 'SWT', 'XQRS'])
            if df.iloc[0, 1] >= threshold or df.iloc[0, 2] >= threshold or df.iloc[1, 2] >= threshold:
                # Collecting Paths
                paths.append(str(path))

                # Collecting Keys
                set_key = path.parent.parent.parent.parent.parent.name
                cat_key = path.parent.parent.parent.parent.name
                patient_key = str(path.parent.parent.name).lstrip("0")

                r1 = re.search(r's\d{3}', str(path.parent.name))
                if r1:
                    session_key = r1.group()

                r2 = re.search(r't\d{3}', str(path.name))
                if r2:
                    file_key = r2.group()

                key.append(set_key + '_' + cat_key + '_'
                            + patient_key + '_' + session_key + '_' + file_key)

                # Collecting coefs
                coef.append(data['score']['corrcoefs'])

        except ValueError:
            error_file.append(str(path))

    if error_log:
        return paths, key, coef, error_file
    else:
        return paths, key, coef


def load_and_concat_feats_v2(directory, method):
    '''
    Directory : The path of the feats-v0_4 folder,
                could be relative or absolute path

    method : Select which method of file to handle (pan, swt, xqrs)
    '''

    # On crée la DataFrame finale
    full_df = pd.DataFrame()

    # On crée une liste de chemins de tous les fichiers du dossier res-v0_4
    res_paths = [str(p) for p in Path('../res-v0_4').rglob(f'*.json')]

    # On boucle sur tous les fichiers du dossier feats-v0-4 (de la méthode choisie)
    for path in tqdm(Path(directory).rglob(f'*{method}.json')):
        data = json.load(open(path, "r"))
        df = pd.DataFrame(columns = data['keys'], data = data['features'])
        df['Set'] = path.parent.parent.parent.parent.parent.name
        df['Categorie Montage'] = path.parent.parent.parent.parent.name
        df['Dossier Patient'] = path.parent.parent.parent.name
        df['Patient'] = path.parent.parent.name
        df['Session'] = path.parent.name

        r = re.search(r't\d{3}', str(path.name))
        if r:
            df['File N°'] = r.group()

        # On vient chercher le nom du fichier
        r_res = re.search("([a-z0-9_]*)", str(path.name))
        # pour le retrouver dans la liste de chemins de tous les fichiers
        # du dossier res-v0_4
        if r_res:
            r = re.compile(f".*{r_res.group()}.*")
            res_path = list(filter(r.match, res_paths)) # On filtre la liste sur notre fichier

        # On vient ouvrir le fichier res correspondant
        data_res = json.load(open(res_path[0], "r"))

        # On vient récupérer les infos voulues
        df['exam_duration'] = data_res["infos"]["exam_duration"]

        if method == 'pan':
            df['Pan_vs_SWT'] = data_res["score"]["corrcoefs"][0][1]
            df['Pan_vs_XQRS'] = data_res["score"]["corrcoefs"][0][2]
        elif method == 'swt':
            df['Pan_vs_SWT'] = data_res["score"]["corrcoefs"][0][1]
            df['SWT_vs_XQRS'] = data_res["score"]["corrcoefs"][1][2]
        else:
            df['SWT_vs_XQRS'] = data_res["score"]["corrcoefs"][1][2]
            df['Pan_vs_XQRS'] = data_res["score"]["corrcoefs"][0][2]

        # On ajoute la DataFrame à la DataFrame golbale
        full_df = full_df.append(df, ignore_index=True)

    # On réarrange les colonnes
    cols = full_df.columns.tolist()
    cols = cols[-9:] + cols[:-9]
    full_df = full_df[cols]

    return full_df


def test_sk_models(classifiers, X_train, y_train, X_test, y_test, preprocessor):
    for classifier in classifiers:
        pipe = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', classifier)])
        pipe.fit(X_train, y_train)
        y_pred_test = pipe.predict(X_test)
        y_pred_train = pipe.predict(X_train)
        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_test = confusion_matrix(y_test, y_pred_test)
        print('-> ', classifier, ':\n------------------------------------------------------')
        print("F1 Score on Train Set: %.3f" % f1_score(y_train, y_pred_train))
        print("F1 Score on Test Set: %.3f" % f1_score(y_test, y_pred_test))
        print()
        print("Accuracy on Train Set: %.3f" % pipe.score(X_train, y_train))
        print("Accuracy on Test Set: %.3f" % pipe.score(X_test, y_test))
        print()
        print('Confusion Matrix on Train Set:')
        print(cm_train)
        print()
        print('Confusion Matrix on Test Set:')
        print(cm_test)
        print()
        print('Sensitivity on Train Set', cm_train[1][1] / (cm_train[1][0] + cm_train[1][1]))
        print('Sensitivity on Test Set', cm_test[1][1] / (cm_test[1][0] + cm_test[1][1]))
        print()
        print('Specificity on Train Set', cm_train[0][0] / (cm_train[0][0] + cm_train[0][1]))
        print('Specificity on Test Set', cm_test[0][0] / (cm_test[0][0] + cm_test[0][1]))
        print('\n------------------------------------------------------\n')


def predict_and_cm(models_fitted, X_train, y_train, X_test, y_test):
    y_pred_train = models_fitted.predict(X_train)
    y_pred_test = models_fitted.predict(X_test)

    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    score = {
        'accuracy train': models_fitted.score(X_train, y_train),
        'accuracy test': models_fitted.score(X_test, y_test),
        'f1_score train': f1_score(y_train, y_pred_train),
        'f1_score test': f1_score(y_test, y_pred_test),
        'cm train': cm_train,
        'cm test': cm_test,
        'sensitivity train': cm_train[1][1] / (cm_train[1][0] + cm_train[1][1]),
        'sensitivity test': cm_test[1][1] / (cm_test[1][0] + cm_test[1][1]),
        'specificity train': cm_train[0][0] / (cm_train[0][0] + cm_train[0][1]),
        'specificity test': cm_test[0][0] / (cm_test[0][0] + cm_test[0][1])
    }

    return score


def plot_cm(cm_train, cm_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    sns.heatmap(cm_train, annot=True, fmt='d', ax=ax1, cbar=False)
    sns.heatmap(cm_test, annot=True, fmt='d', ax=ax2, cbar=False)
    ax1.set_title('Train Set')
    ax2.set_title('Test Set')


def preprocessing(df, key_filter, impute_strategy='mean'):
    df = df[df['Key'].isin(key_filter)]
    df.drop(['Date', 'exam_duration', 'SWT_vs_XQRS', 'Pan_vs_XQRS',
            'interval_index', 'interval_start_time', 'Key'], axis=1, inplace=True)

    df.label = df.label.apply(lambda x: 1 if x!= 0 else 0)
    df = df.replace([np.inf, -np.inf], np.nan)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_strategy)),
        ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, X.columns.tolist())])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=0.2, random_state=42, stratify=y)

    return preprocessor, X_train, y_train, X_test, y_test


def model_comparison(X, y, models, scoring='f1', seed=7):
    print(scoring, '\n')
    # evaluate each model in turn
    results = []
    names = []
    for model in models:
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(models[model], X, y, cv=kfold, scoring=scoring, n_jobs=-1)
        results.append(cv_results)
        names.append(model)
        msg = "%s: %f (%f)" % (model, cv_results.mean(), cv_results.std())
        print(msg)

    # boxplot algorithm comparison
    fig = plt.figure(figsize=(15,5))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in df.columns]
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s(t)' % (j)) for j in df.columns]
        else:
            names += [('%s(t+%d)' % (j, i)) for j in df.columns]
	# put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
	# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def preprocessing_lagged(df, key_filter):
    df = df[df['Key'].isin(key_filter)]

    X = df.iloc[:,1:-1]
    y = df.iloc[:,-1]

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, X.columns.tolist())])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                            test_size=0.2, random_state=42, stratify=y)

    return preprocessor, X_train, y_train, X_test, y_test


def simple_ml(df, model, params=None, grid=False, scoring='f1'):
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1:]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    train_size = int(len(X) * 0.80)
    test_size = len(X) - train_size
    X_train, X_test = X.iloc[0:train_size], X.iloc[train_size:len(X)]
    y_train, y_test = y.iloc[0:train_size], y.iloc[train_size:len(X)]

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    if grid:
        rf = Pipeline(steps=[('preprocessor', preprocessor),
                ('classifier', model)])

        grid = GridSearchCV(rf, params, n_jobs=-1, scoring=scoring, verbose=1)
        grid.fit(X_train, y_train)

        y_pred_train = grid.best_estimator_.predict(X_train)
        y_pred_test = grid.best_estimator_.predict(X_test)

        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_test = confusion_matrix(y_test, y_pred_test)

        try:
            score = {
                'accuracy train': grid.best_estimator_.score(X_train, y_train),
                'accuracy test': grid.best_estimator_.score(X_test, y_test),
                'f1_score train': f1_score(y_train, y_pred_train),
                'f1_score test': f1_score(y_test, y_pred_test),
                'cm train': cm_train,
                'cm test': cm_test,
                'sensitivity train': cm_train[1][1] / (cm_train[1][0] + cm_train[1][1]),
                'sensitivity test': cm_test[1][1] / (cm_test[1][0] + cm_test[1][1]),
                'specificity train': cm_train[0][0] / (cm_train[0][0] + cm_train[0][1]),
                'specificity test': cm_test[0][0] / (cm_test[0][0] + cm_test[0][1])
            }
        except IndexError:
                score = {
                'accuracy train': rf.score(X_train, y_train),
                'accuracy test': rf.score(X_test, y_test),
                'f1_score train': f1_score(y_train, y_pred_train),
                'f1_score test': f1_score(y_test, y_pred_test),
                'cm train': cm_train,
                'cm test': cm_test
                }

        return score, grid

    else:
        rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', model)])

        rf.fit(X_train, y_train)

        y_pred_train = rf.predict(X_train)
        y_pred_test = rf.predict(X_test)

        cm_train = confusion_matrix(y_train, y_pred_train)
        cm_test = confusion_matrix(y_test, y_pred_test)

        try:
            score = {
                'accuracy train': rf.score(X_train, y_train),
                'accuracy test': rf.score(X_test, y_test),
                'f1_score train': f1_score(y_train, y_pred_train),
                'f1_score test': f1_score(y_test, y_pred_test),
                'cm train': cm_train,
                'cm test': cm_test,
                'sensitivity train': cm_train[1][1] / (cm_train[1][0] + cm_train[1][1]),
                'sensitivity test': cm_test[1][1] / (cm_test[1][0] + cm_test[1][1]),
                'specificity train': cm_train[0][0] / (cm_train[0][0] + cm_train[0][1]),
                'specificity test': cm_test[0][0] / (cm_test[0][0] + cm_test[0][1])
            }
        except IndexError:
                score = {
                'accuracy train': rf.score(X_train, y_train),
                'accuracy test': rf.score(X_test, y_test),
                'f1_score train': f1_score(y_train, y_pred_train),
                'f1_score test': f1_score(y_test, y_pred_test),
                'cm train': cm_train,
                'cm test': cm_test
                }

        return score, rf


def supervised_by_exam(df, lag_in=1, lag_out=1):
    dictionnary_by_key_xqrs = {k:df[df.Key == k]
                                for k in df.Key.unique()}

    result = pd.DataFrame()
    for key in dictionnary_by_key_xqrs.keys():
        result = result.append(series_to_supervised(dictionnary_by_key_xqrs[key], lag_in, lag_out),
                                                    ignore_index=True)

    label = result.iloc[:,-1:].rename(columns={'label(t)': 'label'})
    key = result.iloc[:,:1].rename(columns={'Key(t)': 'Key'})

    result = result[result.columns.drop(list(result.filter(regex='label|Key')))]

    result.insert(0, 'Key', key)
    result['label'] = label

    return result


def simple_ml_no_train(df, model):
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1:]

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder())])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])


    rf = Pipeline(steps=[('preprocessor', preprocessor),
                    ('classifier', model)])

    rf.fit(X, y)


    return rf