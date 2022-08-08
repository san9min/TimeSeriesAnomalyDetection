from datasets.KPI import build_kpi
from datasets.Yahoo import build_yahoo

def load_data(args):

    if 'Yahoo' in args.datasets:
        train,test = build_yahoo(args)

    elif args.datasets == 'KPI':
        train, test = build_kpi(args)


    return train, test