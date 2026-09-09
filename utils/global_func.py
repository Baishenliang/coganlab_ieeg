"""Shared utility functions used across analysis projects."""


def fill_missing_coords(df, reference_coords=None):
    import numpy as np
    import pandas as pd
    import re

    def split_label(label):
        match = re.match(r"([a-zA-Z]+)([0-9]+)", label)
        return (match.group(1), int(match.group(2))) if match else (None, None)

    if reference_coords:
        ref_list = []
        for label, position in reference_coords.items():
            if not np.isnan(position).any():
                prefix, number = split_label(label)
                if prefix:
                    ref_list.append({'prefix': prefix, 'num': number, 'pos': position})
        ref_df = pd.DataFrame(ref_list)

    for index, row in df.iterrows():
        if np.isnan([row['x'], row['y'], row['z']]).any():
            subject = row['subj']
            label = row['label']
            prefix, target_number = split_label(label)

            if not prefix:
                continue

            mask = ((df['subj'] == subject)
                    & df['label'].str.startswith(prefix)
                    & (~df['x'].isna()))
            current_known = df[mask].copy()
            current_known['prefix_num'] = current_known['label'].apply(
                lambda value: split_label(value)[1])

            indices = current_known['prefix_num'].values
            coords = current_known[['x', 'y', 'z']].values

            if len(indices) < 2 and reference_coords:
                sub_ref = ref_df[ref_df['prefix'] == prefix]
                indices = sub_ref['num'].values
                coords = np.array(list(sub_ref['pos'].values))

            if len(indices) >= 2:
                new_xyz = []
                for coordinate_index in range(3):
                    coefficients = np.polyfit(indices, coords[:, coordinate_index], 1)
                    new_xyz.append(np.polyval(coefficients, target_number))
                df.at[index, 'x'], df.at[index, 'y'], df.at[index, 'z'] = new_xyz

    return df


def get_coor(chs, method: str = 'individual', interpolate: bool = True,
             subjects_dir=None):
    """Return electrode coordinates for channels in individual or group space."""
    import os
    import re

    import mne
    import numpy as np
    import pandas as pd
    from ieeg.viz.mri import force2frame, get_sub_dir, subject_to_info

    parsed = {}
    for channel in chs:
        subject, label = channel.split('-')
        parsed.setdefault(subject, []).append(label)

    df_coords = pd.DataFrame(columns=['subj', 'label', 'x', 'y', 'z'])
    home = os.path.expanduser("~")

    for subject, labels in parsed.items():
        if subject == 'D107':
            subject_tag = 'D107B'
        elif subject == 'D139':
            subject_tag = 'D139A'
        else:
            subject_tag = subject

        if method == 'individual':
            trans = mne.transforms.Transform(fro='head', to='mri')
        elif method == 'group':
            to_fsaverage = mne.read_talxfm(
                subject_tag, get_sub_dir(subjects_dir))
            trans = mne.transforms.Transform(
                fro='head', to='mri', trans=to_fsaverage['trans'])
        elif method == 'ras':
            path = os.path.join(
                home, f"Box\\ECoG_Recon\\{subject_tag}\\elec_recon\\"
                f"{subject_tag}_elec_locations_RAS.txt")
            try:
                with open(path, 'r') as file:
                    lines = file.readlines()
            except FileNotFoundError:
                print(f"Warning: File not found for subject {subject}")
                continue

            coord_map = {}
            for line in lines:
                parts = line.strip().split()
                label = parts[0] + parts[1]
                try:
                    x, y, z = map(float, parts[2:5])
                    coord_map[label] = (x, y, z)
                except ValueError:
                    continue
        else:
            raise ValueError(f"Unknown coordinate method: {method}")

        if method in ('individual', 'group'):
            info = subject_to_info(subject_tag, subjects_dir=subjects_dir)
            montage = info.get_montage()
            force2frame(montage, trans.from_str)
            montage.apply_trans(trans)
            coord_map = {key: value for key, value in
                         montage.get_positions()['ch_pos'].items()}

        for label in labels:
            match = re.match(r"([a-zA-Z]+)([0-9]+)", label)
            prefix, suffix = match.groups() if match else (label, "")
            if subject == 'D128' and prefix == 'LAI':
                label_tag = 'LIA' + suffix
            elif subject == 'D128' and prefix == 'RAI':
                label_tag = 'RIA' + suffix
            else:
                label_tag = label

            if label_tag in coord_map:
                x, y, z = coord_map[label_tag]
                if method in ('individual', 'group'):
                    x, y, z = 1000 * x, 1000 * y, 1000 * z
                df_coords.loc[len(df_coords)] = [subject, label, x, y, z]
            else:
                df_coords.loc[len(df_coords)] = [subject, label,
                                                  np.nan, np.nan, np.nan]

    if interpolate:
        df_coords = fill_missing_coords(df_coords)

    return df_coords
