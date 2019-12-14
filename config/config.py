class Config(object):
    env = 'default'
    classify = 'softmax'
    num_classes = 10006
    metric = 'arc_margin'
    easy_margin = False
    use_se = True
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = ''
    train_list = '/home/lihebeizi/data/FaceRegDataset/train_enhanced/train.meta'
    val_list = '/home/lihebeizi/data/FaceRegDataset/train_enhanced/val.meta'
    test_list = '/home/lihebeizi/data/FaceRegDataset/test/test.meta'

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'

    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/resnet18_32_v2_70.pth'
    save_interval = 10
    save_name = "resnet18_32_v3"

    train_batch_size = 32  # batch size
#    test_batch_size = 60

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    num_workers = 4
    print_freq = 100

    max_epoch = 100
    lr = 1e-1  # initial learning rate
    lr_step = 20
    lr_gamma = 0.1
    weight_decay = 5e-4


    step2_load_pth = 'checkpoints/resnet18_32_90.pth'
    step2_lr = 0.0001
    step2_lr_step = 25
    step2_save_name = 'resnet18_32_step2'
    step2_optimizer = 'Adam'