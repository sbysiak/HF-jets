import uproot


def calc_njets(input_files, n_b=None, trainset_frac_b=None):
    if n_b and trainset_frac_b:
        raise ValueError('One cannot provide both parameters: `n_b` and `trainset_frac_b`')
    if not n_b and not trainset_frac_b:
        raise ValueError('Provide one out of these two parameters: `n_b` and `trainset_frac_b`')
        
    n_avail_b, n_avail_c, n_avail_udsg = 0, 0, 0
    for f in input_files:
        try:
            froot = uproot.open(f)
        except FileNotFoundError as e:
            print(f'WARNING: file {f} not found!')
            continue
        n_avail_b += froot[tree_name_core+'bJets'].numentries
        n_avail_c += froot[tree_name_core+'cJets'].numentries
        n_avail_udsg += froot[tree_name_core+'udsgJets'].numentries
#     print(n_avail_b, n_avail_c, n_avail_udsg)

    if not n_b:
        n_b = int(n_avail_b * trainset_frac_b)
    else:
        trainset_frac_b = n_b / n_avail_b # round_down(n_b / n_avail_b, 4)
    n_c    = int(0.1 * n_b)
    n_udsg = int(0.9 * n_b)

    trainset_frac_c    = n_c / n_avail_c # round_down(n_c / n_avail_c, 4)
    trainset_frac_udsg = n_udsg / n_avail_udsg # round_down(n_udsg / n_avail_udsg, 4)
    
    d = {'b'   :[n_avail_b, n_b, trainset_frac_b], 
         'c'   :[n_avail_c, n_c, trainset_frac_c], 
         'udsg':[n_avail_udsg, n_udsg, trainset_frac_udsg]}
    df = pd.DataFrame(d)
    df.index = ['n available', 'n trainset', 'trainset fraction']
    if any(df.loc['trainset fraction'] > 1):
        print('ERROR: not enough jets')
    return df
