from pathlib import Path
import pandas as pd
import json
import re
from tqdm import tqdm


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