config = {'seed': 2021,
          'trainer': {
              'max_epochs': 1000,
              # 'gpus': 1,
              'accumulate_grad_batches': 1,
              # 'progress_bar_refresh_rate': 1,
              # 'fast_dev_run': False,
              # 'num_sanity_val_steps': 0,
              # 'resume_from_checkpoint': None,
            #   'default_root_dir': "/home/jarartur/Workspaces/HAAv2/temp/pl_logs",
              'log_every_n_steps': 50
          },
          'data': {
              'dataset_name': 'haav2',
              'batch_size': 256,
              'img_size': [224, 224],
              'num_workers': 14,
            #   'train_data_csv': '/home/jarartur/Workspaces/HAAv2/classification/tests.csv',
              'val_data_csv': '/net/pr2/projects/plgrid/plggmiadl/arjurgas/Datasets/HAAv2/ResNet_classification/bp_annot_val.csv',
              # 'val_data_csv': '/net/pr2/projects/plgrid/plggmiadl/arjurgas/Datasets/HAAv2/ResNet_classification/acr_val.csv',
              # 'test_data_csv': '/net/pr2/projects/plgrid/plggmiadl/arjurgas/Datasets/HAAv2/ResNet_classification/anh_annot.csv',
              # 'test_data_csv': '/net/pr2/projects/plgrid/plggmiadl/arjurgas/Datasets/HAAv2/ResNet_classification/acr_test_annot.csv',
              'test_data_csv': '/net/pr2/projects/plgrid/plggmiadl/arjurgas/Datasets/HAAv2/ResNet_classification/bp_annot_test.csv',
              # 'test_data_csv': '/net/pr2/projects/plgrid/plggmiadl/arjurgas/Datasets/HAAv2/ResNet_classification/acr_test.csv',
          },
          'model':{
                # 'backbone_init': {
                #     'model': 'efficientnet_v2_s_in21k',
                #     'nclass': 0, # do not change this
                #     'pretrained': True,
                #     },
                'optimizer_init':{
                    # 'class_path': 'torch.optim.SGD',
                    'class_path': 'torch.optim.AdamW',
                    'init_args': {
                        'lr': 0.01,
                        # 'momentum': 0.95,
                        # 'weight_decay': 0.0005
                        }
                    },
                'lr_scheduler_init':{
                    # 'class_path': 'torch.optim.lr_scheduler.CosineAnnealingLR',
                    'class_path': 'torch.optim.lr_scheduler.ExponentialLR',
                    'init_args':{
                        # 'T_max': 0 # no need to change this
                        'gamma': 0.97
                        },
                    'step': 'epoch'
                    }
            }
}