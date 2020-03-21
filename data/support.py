import functools
import json
import os
import shutil
from collections import defaultdict
from multiprocessing.pool import Pool, ThreadPool

from tqdm import tqdm

import pretorched.data.utils
from pretorched.runners import config as cfg


def generate_metadata(name, root=None):
    root = cfg.DATA_ROOT if root is None else root
    func = {
        'Moments': generate_moments_metadata,
        'MultiMoments': generate_multimoments_metadata,
        'Kinetics': generate_kinetics_metadata,
    }.get(name)
    func(os.path.join(root, name))


def generate_moments_categories_metadata(cat_file):
    cat2idx = {}
    categories = []
    cat_txt_file = cat_file + '.txt'
    cat_csv_file = cat_file + '.csv'
    cat2idx_csv_file = cat_file + '2idx.csv'

    if not os.path.exists(cat2idx_csv_file):
        shutil.copy(cat_txt_file, cat_csv_file)

    with open(cat_csv_file) as f:
        for line in f:
            cat, idx = line.strip().split(',')
            cat2idx[cat] = idx
            categories.append(cat)

    with open(cat_txt_file, 'w') as f:
        f.write('\n'.join(categories))

    with open(cat_file + '2idx.txt', 'w') as f:
        for cat in categories:
            f.write(f'{cat} {cat2idx[cat]}\n')

    with open(cat_file + '.json', 'w') as f:
        json.dump(categories, f)

    with open(cat_file + '2idx.json', 'w') as f:
        json.dump(cat2idx, f)

    return cat2idx


def generate_kinetics_categories_metadata(cat_file):
    cat2idx = {}
    categories = []
    cat_txt_file = cat_file + '.txt'
    cat_csv_file = cat_file + '.csv'

    with open(cat_txt_file) as f:
        categories = [x.strip() for x in f]
    cat2idx = {c: i for i, c in enumerate(categories)}

    with open(cat_csv_file, 'w') as f:
        for cat in categories:
            f.write(f'{cat},{cat2idx[cat]}\n')

    with open(cat_file + '2idx.txt', 'w') as f:
        for cat in categories:
            f.write(f'{cat} {cat2idx[cat]}\n')

    with open(cat_file + '.json', 'w') as f:
        json.dump(categories, f)

    with open(cat_file + '2idx.json', 'w') as f:
        json.dump(cat2idx, f)

    return cat2idx


def generate_multimoments_categories_metadata(cat_file):
    cat2idx = {}
    categories = []
    cat_txt_file = cat_file + '.txt'
    cat_csv_file = cat_file + '.csv'
    cat2idx_csv_file = cat_file + '2idx.csv'

    if not os.path.exists(cat2idx_csv_file):
        shutil.copy(cat_txt_file, cat_csv_file)

    with open(cat_csv_file) as f:
        for line in f:
            cat, idx = line.strip().split(',')
            cat2idx[cat] = idx
            categories.append(cat)

    with open(cat_txt_file, 'w') as f:
        f.write('\n'.join(categories))

    with open(cat_file + '2idx.txt', 'w') as f:
        for cat in categories:
            f.write(f'{cat} {cat2idx[cat]}\n')

    with open(cat_file + '.json', 'w') as f:
        json.dump(categories, f)

    with open(cat_file + '2idx.json', 'w') as f:
        json.dump(cat2idx, f)

    return cat2idx


def generate_moments_split_metadata(split, splitfile, cat2idx):
    with open(splitfile) as f:
        lines = [x.strip().split(',') for x in f]

    json_data = []
    with open(splitfile.replace('.csv', '.txt'), 'w') as f:
        for line in lines:
            path, cat, *tail = line
            f.write(f'{path} {cat2idx[cat]}\n')
            json_data.append({
                'path': path,
                'filename': os.path.basename(path),
                'label': cat2idx[cat],
                'category': cat,
            })

    with open(splitfile.replace('.csv', '.json'), 'w') as f:
        json.dump(json_data, f)


def generate_kinetics_split_metadata(splitfile, cat2idx):
    with open(splitfile) as f:
        lines = [x.strip().split(' ') for x in f]

    idx2cat = {idx: cat for cat, idx in cat2idx.items()}
    json_data = []
    for line in lines:
        path, label = line
        json_data.append({
            'path': path,
            'filename': os.path.basename(path),
            'label': label,
            'category': idx2cat[int(label)],
        })

    with open(splitfile.replace('.txt', '.json'), 'w') as f:
        json.dump(json_data, f)


def generate_multimoments_split_metadata(split, splitfile, cat2idx):
    with open(splitfile) as f:
        lines = [x.strip().split(',') for x in f]

    json_data = []
    idx2cat = {idx: cat for cat, idx in cat2idx.items()}
    with open(splitfile.replace('.csv', '.txt'), 'w') as f:
        for line in lines:
            path, *labels = line
            f.write(f'{path} {" ".join(labels)}\n')
            json_data.append({
                'path': path,
                'filename': os.path.basename(path),
                'labels': labels,
                'categories': [idx2cat[c] for c in labels],
            })

    with open(splitfile.replace('.csv', '.json'), 'w') as f:
        json.dump(json_data, f)


def generate_moments_metadata(root):
    cat_file = os.path.join(root, 'moments_categories')
    cat2idx = generate_moments_categories_metadata(cat_file)
    for split in ['training', 'validation']:
        splitfile = os.path.join(root, f'{split}Set.csv')
        generate_moments_split_metadata(split, splitfile, cat2idx)


def generate_multimoments_metadata(root):
    cat_file = os.path.join(root, 'moments_categories')
    cat2idx = generate_moments_categories_metadata(cat_file)
    for split in ['training', 'validation']:
        splitfile = os.path.join(root, f'{split}Set.csv')
        if not os.path.exists(splitfile):
            shutil.copy(splitfile.replace('.csv', '.txt'), splitfile)
        generate_multimoments_split_metadata(split, splitfile, cat2idx)


def unify_multi_moments(data_root=None):
    data_root = cfg.DATA_ROOT if data_root is None else data_root
    roots = {name: os.path.join(data_root, name) for name in ['Moments', 'MultiMoments']}
    data = defaultdict(dict)
    paths = defaultdict(dict)
    filenames = defaultdict(dict)
    for name, root in roots.items():
        for split in ['training', 'validation']:
            with open(os.path.join(root, f'{split}Set.json')) as f:
                d = json.load(f)
                data[name][split] = d
                paths[name][split] = set([x['path'] for x in d])
                filenames[name][split] = set([x['filename'] for x in d])
                print(f'{name} {split}:\n'
                      f'\tLen: {len(data[name][split])}')
    for split in ['training', 'validation']:
        common_filenames = set.intersection(*[filenames[n][split] for n in roots])
        common_paths = set.intersection(*[paths[n][split] for n in roots])
        print(f'Common {split} files: {len(common_filenames)}')
        print(f'Common {split} paths: {len(common_paths)}')

    all_filenames = {}
    all_paths = {}
    for name in roots:
        all_filenames[name] = set.union(*[filenames[name][s] for s in ['training', 'validation']])
        all_paths[name] = set.union(*[paths[name][s] for s in ['training', 'validation']])

    total_common_filenames = set.intersection(*all_filenames.values())
    total_common_paths = set.intersection(*all_paths.values())
    print(f'Total Common files: {len(total_common_filenames)}')
    print(f'Total Common paths: {len(total_common_paths)}')

    total_all_filenames = set.union(*all_filenames.values())
    total_all_paths = set.union(*all_paths.values())
    print(f'Total All files: {len(total_all_filenames)}')
    print(f'Total All paths: {len(total_all_paths)}')

    multi_missing = all_paths['Moments'] - all_paths['MultiMoments']
    print(len(multi_missing))
    with open('multi_missing.txt', 'w') as f:
        f.write('\n'.join(multi_missing))


def generate_kinetics_metadata(root):
    cat_file = os.path.join(root, 'kinetics_categories')
    cat2idx = generate_kinetics_categories_metadata(cat_file)
    for split in ['kinetics_test.txt', 'kinetics_train.txt', 'kinetics_val.txt']:
        splitfile = os.path.join(root, split)
        generate_kinetics_split_metadata(splitfile, cat2idx)


def get_all_data(data_root=None, splits=['training', 'validation']):
    roots = {name: os.path.join(data_root, 'Moments') for name in ['moments', 'multimoments']}
    data = defaultdict(dict)
    for name, root in roots.items():
        for split in splits:
            with open(os.path.join(root, f'{name.lower()}_{split}.json')) as f:
                data[name][split] = json.load(f)
    return data


def verify_kinetics(data_root=None, num_workers=24):
    data_root = cfg.DATA_ROOT if data_root is None else data_root
    all_paths = []
    for split in ['test', 'val', 'train']:
        with open(os.path.join(data_root, 'Kinetics', f'kinetics_{split}.json')) as f:
            data = json.load(f)
            all_paths += [d['path'] for d in data]

    # all_paths = [os.path.join(data_root, 'Kinetics', 'videos', f) for f in set(all_paths)]
    all_paths = [os.path.join(data_root, 'Kinetics', 'preproc_videos', f) for f in set(all_paths)]
    missing = find_missing(all_paths, num_workers=num_workers)
    return missing


def verify_full_moments(data_root=None, num_workers=24):
    data_root = cfg.DATA_ROOT if data_root is None else data_root
    data = get_all_data(data_root=data_root)
    all_paths = []
    for name, splits in data.items():
        for split, split_data in splits.items():
            all_paths += [x['path'] for x in split_data]

    # all_paths = [os.path.join(data_root, 'Moments', 'videos', f) for f in set(all_paths)]
    all_paths = [os.path.join(data_root, 'Moments', 'preproc_videos', f) for f in set(all_paths)]
    missing = find_missing(all_paths, num_workers=num_workers)
    return missing


def find_missing(all_paths, num_workers=24):
    num_proc, missing = 0, []
    with ThreadPool(num_workers) as pool:
        for miss in pool.imap_unordered(is_missing, all_paths):
            if miss is not None:
                print(f'Not found: {miss}')
                missing.append(miss)
            num_proc += 1
        pool.close()
        pool.join()
        print(f'Num_processed: {num_proc}')
        print(f'Num missing: {len(missing)}')
    return missing


def generate_extended_metadata(data_root=None, num_workers=48):
    data_root = cfg.DATA_ROOT if data_root is None else data_root
    roots = {name: os.path.join(data_root, 'Moments') for name in ['multimoments']}
    for name, root in roots.items():
        for split in ['training', 'validation']:
            fname = os.path.join(root, f'{name.lower()}_{split}.json')
            with open(fname) as f:
                d = json.load(f)
            data_out, bad_out = get_extended_metadata(d, os.path.join(data_root, 'Moments', 'videos'),
                                                      num_workers=num_workers)
            with open(fname.replace('.json', '_extended.json'), 'w') as f:
                json.dump(data_out, f)

            with open(fname.replace('.json', '_bad.json'), 'w') as f:
                json.dump(bad_out, f)


def generate_extended_metadata_kinetics(data_root=None, num_workers=48):
    name = 'kinetics'
    data_root = cfg.DATA_ROOT if data_root is None else data_root
    root = os.path.join(data_root, 'Kinetics')
    # for split in ['test', 'train', 'val']:
    for split in ['test']:
        fname = os.path.join(root, f'{name.lower()}_{split}.json')
        with open(fname) as f:
            d = json.load(f)
        data_out, bad_out = get_extended_metadata(d, os.path.join(data_root, 'Kinetics', 'videos'),
                                                  num_workers=num_workers)
        with open(fname.replace('.json', '_extended.json'), 'w') as f:
            json.dump(data_out, f)

        with open(fname.replace('.json', '_bad.json'), 'w') as f:
            json.dump(bad_out, f)


def get_info(data, root=''):
    path = os.path.join(root, data['path'])
    return {**data, **pretorched.data.utils.get_info(path)}


def get_extended_metadata(data, root, num_workers=100):
    num_proc = 0
    out, bad = [], []
    func = functools.partial(get_info, root=root)
    with Pool(num_workers) as pool:
        for d in pool.imap(func, data):
            if 'num_frames' not in d:
                bad.append(d)
            else:
                out.append(d)
            num_proc += 1
            if num_proc % 100 == 0:
                print(num_proc, d)
        pool.close()
        pool.join()
    return out, bad


def is_missing(filename):
    if not os.path.exists(filename):
        return filename


def downsample(args):
    input, output = args
    if not os.path.exists(output):
        pretorched.data.utils.downsample_video(input, output)


def downsample_full_moments(name='moments', split='training', data_root=None, num_workers=36):
    data_root = cfg.DATA_ROOT if data_root is None else data_root
    videos_root = os.path.join(data_root, 'Moments', 'videos')
    preproc_videos_root = os.path.join(data_root, 'Moments', 'preproc_videos')
    fname = os.path.join(data_root, 'Moments', f'{name}_{split}_extended.json')
    with open(fname) as f:
        data = json.load(f)

    paths = [d['path'] for d in data]
    args = [(os.path.join(videos_root, path), os.path.join(preproc_videos_root, path)) for path in paths]
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(downsample, args), total=len(args)))

    # for cat in os.listdir(videos_root):
        # cat_dir = os.path.join(videos_root, cat)
        # input_videos = [os.path.join(cat_dir, f) for f in os.listdir(cat_dir)]
        # output_videos = [os.path.join(preproc_videos_root, cat, f) for f in os.listdir(cat_dir)]
        # args = list(zip(input_videos, output_videos))


def downsample_kinetics(name='kinetics', split='train', data_root=None, num_workers=36):
    data_root = cfg.DATA_ROOT if data_root is None else data_root
    videos_root = os.path.join(data_root, 'Kinetics', 'videos')
    preproc_videos_root = os.path.join(data_root, 'Kinetics', 'preproc_videos')
    fname = os.path.join(data_root, 'Kinetics', f'{name}_{split}_extended.json')
    with open(fname) as f:
        data = json.load(f)

    paths = [d['path'] for d in data]
    args = [(os.path.join(videos_root, path), os.path.join(preproc_videos_root, path)) for path in paths]
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(downsample, args), total=len(args)))
#    for cat in os.listdir(videos_root):
#        if cat.endswith('.zip'):
#            continue
#        cat_dir = os.path.join(videos_root, cat)
#        args = [(os.path.join(cat_dir, f), os.path.join(preproc_videos_root, cat, f)) for f in os.listdir(cat_dir)]
#        with Pool(num_workers) as pool:
#            list(tqdm(pool.imap_unordered(downsample, args), total=len(args)))
