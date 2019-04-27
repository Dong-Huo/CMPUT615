
class Config():

    training_dir = './data/train_vot_Archive/'
    training_dir_list = ['./data/trans_scale_vot/', './data/trans_scale_vot1/','./data/trans_vot/','./data/trans_vot1/', './data/scale_vot/','./data/org_vot/']
    train_batch_size = 64
    train_number_epochs = 100
    lr = 0.0001

    testing_dir = './data/train_vot_Archive/'
    test_model_dir = './b64_0.00001_alex_model/'

    warp_choice = ''    # 'inv', None

    reg_coord = True    # True: regress on coords. False: regress on warp
    warp_sty = 'affine' # "homo", "affine", "simi", None. Cannot take none when reg_coord==True

    train_folder = ['bag',       'book',         'fish4',       'handball1',     'nature',      'singer2',
        'ball1',       'butterfly',    'girl',         'handball2',    'octopus',     'singer3',
        'ball2',       'car1',         'glove',        'helicopter',   'pedestrian1', 'soccer1',
        'basketball',   'car2',         'godfather',    'iceskater1',   'pedestrian2',  'soccer2',
        'birds1',       'crossing',     'graduate',     'iceskater2',   'rabbit',       'soldier',
        'birds2',       'dinosaur',     'gymnastics1',  'leaves',       'racing',       'sphere',
        'blanket',      'fernando',     'gymnastics2',  'marching',     'road',         'tiger',
        'bmx',          'fish1',        'gymnastics3',  'matrix',       'shaking',      'traffic',
        'bolt1',        'fish2',        'gymnastics4',  'motocross1',   'sheep',        'tunnel',
        'bolt2',        'fish3',        'hand',         'motocross2',   'singer1',      'wiper']

    test_folder = ['ball', 'david', 'fish1',    'polarbear',  'sunshade',  'trellis',
        'bicycle',      'diving',   'hand1',    'skating',    'surfing',   'tunnel',
        'car',          'drunk',    'jogging',  'sphere',     'torus',     'woman']

    spec_folder = ['car1']