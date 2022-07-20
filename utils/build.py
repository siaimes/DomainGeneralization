from data import data_pacs, data_oh, data_vlcs, data_dn
def get_inter_config(args):
    if args.suffix == 'PACS':
        args.root = '../datasets/PACS/'
        args.source = ['art_painting', 'cartoon', 'photo', 'sketch']
        args.n_classes = 7
        args.single_model = ['../exps/PACS/intra_adr_single/art_painting.tar',
                             '../exps/PACS/intra_adr_single/cartoon.tar',
                             '../exps/PACS/intra_adr_single/photo.tar',
                             '../exps/PACS/intra_adr_single/sketch.tar']
        args.intra_model = ['../exps/PACS/intra_adr/art_painting.tar',
                            '../exps/PACS/intra_adr/cartoon.tar',
                            '../exps/PACS/intra_adr/photo.tar',
                            '../exps/PACS/intra_adr/sketch.tar']
    if args.suffix == 'officehome':
        args.root = '../datasets/officehome/'
        args.source = ['art', 'clipart', 'product', 'real_world']
        args.n_classes = 65
        args.single_model = ['../exps/officehome/intra_adr_single/art.tar',
                             '../exps/officehome/intra_adr_single/clipart.tar',
                             '../exps/officehome/intra_adr_single/product.tar',
                             '../exps/officehome/intra_adr_single/real_world.tar']
        args.intra_model = ['../exps/officehome/intra_adr/art.tar',
                            '../exps/officehome/intra_adr/clipart.tar',
                            '../exps/officehome/intra_adr/product.tar',
                            '../exps/officehome/intra_adr/real_world.tar']
    if args.suffix == 'VLCS':
        args.root = '../datasets/VLCS/'
        args.source = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
        args.n_classes = 5
        args.single_model = ['../exps/VLCS/intra_adr_single/CALTECH.tar',
                             '../exps/VLCS/intra_adr_single/LABELME.tar',
                             '../exps/VLCS/intra_adr_single/PASCAL.tar',
                             '../exps/VLCS/intra_adr_single/SUN.tar']
        args.intra_model = ['../exps/VLCS/intra_adr/CALTECH.tar',
                            '../exps/VLCS/intra_adr/LABELME.tar',
                            '../exps/VLCS/intra_adr/PASCAL.tar',
                            '../exps/VLCS/intra_adr/SUN.tar']
    if args.suffix == 'Domainnet':
        args.root = '../datasets/Domainnet/'
        args.source = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.n_classes = 345
        args.single_model = ['../exps/Domainnet/intra_adr_single/art_painting.tar',
                             '../exps/Domainnet/intra_adr_single/cartoon.tar',
                             '../exps/Domainnet/intra_adr_single/photo.tar',
                             '../exps/Domainnet/intra_adr_single/sketch.tar']
        args.intra_model = ['../exps/Domainnet/intra_adr/art_painting.tar',
                            '../exps/Domainnet/intra_adr/cartoon.tar',
                            '../exps/Domainnet/intra_adr/photo.tar',
                            '../exps/Domainnet/intra_adr/sketch.tar']
    return args

def get_intra_config(args):
    if args.suffix == 'PACS':
        args.root = '../datasets/PACS/'
        args.source = ['art_painting', 'cartoon', 'photo', 'sketch']
        args.n_classes = 7
    if args.suffix == 'officehome':
        args.root = '../datasets/officehome/'
        args.source = ['art', 'clipart', 'product', 'real_world']
        args.n_classes = 65
    if args.suffix == 'VLCS':
        args.root = '../datasets/VLCS/'
        args.source = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
        args.n_classes = 5
    if args.suffix == 'Domainnet':
        args.root = '../datasets/Domainnet/'
        args.source = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.n_classes = 345
    return args

def get_inter_loader(args):
    if args.suffix == 'PACS':
        s_loader = data_pacs.get_kd_single_dataloader(args)
        source_loader, val_loader = data_pacs.get_train_dataloader(args)
        target_loader = data_pacs.get_val_dataloader(args)
    elif args.suffix == 'officehome':
        s_loader = data_oh.get_kd_single_dataloader(args)
        source_loader, val_loader = data_oh.get_train_dataloader(args)
        target_loader = data_oh.get_val_dataloader(args)
    elif args.suffix == 'VLCS':
        s_loader = data_vlcs.get_kd_single_dataloader(args)
        source_loader, val_loader = data_vlcs.get_train_dataloader(args)
        target_loader = data_vlcs.get_val_dataloader(args)
    else:
        s_loader = data_dn.get_kd_single_dataloader(args)
        source_loader, val_loader = data_dn.get_train_dataloader(args)
        target_loader = data_dn.get_val_dataloader(args)
    return s_loader, source_loader, val_loader, target_loader

def get_intra_loader(args):
    if args.suffix == 'PACS':
        source_loader, val_loader = data_pacs.get_train_dataloader(args)
        target_loader = data_pacs.get_val_dataloader(args)
    elif args.suffix == 'officehome':
        source_loader, val_loader = data_oh.get_train_dataloader(args)
        target_loader = data_oh.get_val_dataloader(args)
    elif args.suffix == 'VLCS':
        source_loader, val_loader = data_vlcs.get_train_dataloader(args)
        target_loader = data_vlcs.get_val_dataloader(args)
    else:
        source_loader, val_loader = data_dn.get_train_dataloader(args)
        target_loader = data_dn.get_val_dataloader(args)
    return source_loader, val_loader, target_loader

def get_intra_single_loader(args):
    if args.suffix == 'PACS':
        source_loader = data_pacs.single_train_dataloader(args)
        val_loader = data_pacs.get_val_dataloader(args)
        target_loader = data_pacs.get_val_dataloader(args)
    elif args.suffix == 'officehome':
        source_loader = data_oh.single_train_dataloader(args)
        val_loader = data_oh.get_val_dataloader(args)
        target_loader = data_oh.get_val_dataloader(args)
    elif args.suffix == 'VLCS':
        source_loader = data_vlcs.single_train_dataloader(args)
        val_loader = data_vlcs.get_val_dataloader(args)
        self.target_loader = data_vlcs.get_val_dataloader(args)
    else:
        source_loader = data_dn.single_train_dataloader(args)
        val_loader = data_dn.get_val_dataloader(args)
        target_loader = data_dn.get_val_dataloader(args)
    return source_loader, val_loader, target_loader