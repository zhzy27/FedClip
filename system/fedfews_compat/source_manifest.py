COMMIT = '1b29dec70b66d89b956d90e0da024897f698897a'
URL = 'https://github.com/pgg3/FedFew'
FILES = {'PFLlib/system/flcore/clients/clientbase.py': {'sha256': 'a5d3c15bef54a0061ea11b8875160f8b9ab1b53bc70a376a69b7d69428e769d7',
                                                'adaptation': 'isolated imports; common DataLoader factory '
                                                              'with original phase options and injectable '
                                                              'class; whitespace normalized'},
 'PFLlib/system/flcore/clients/clientfedfews.py': {'sha256': 'ede179dce0275145da8bacbe5c3bedfb9096b1845cdad378737f831b76a7bd87',
                                                   'adaptation': 'isolated imports; trailing whitespace '
                                                                 'normalized'},
 'PFLlib/system/flcore/servers/serverbase.py': {'sha256': 'acb8b13bc392feaff699e6607c0d20698b37c6b6861a2b50996e1e42d4f21b70',
                                                'adaptation': 'isolated imports; trailing whitespace '
                                                              'normalized'},
 'PFLlib/system/flcore/servers/serverfedfews.py': {'sha256': '49e2ae03c583ea5fadea9e8d5e523d34a82ff3ee633ba5fda041014774dda53d',
                                                   'adaptation': 'isolated imports; trailing whitespace '
                                                                 'normalized'},
 'PFLlib/system/utils/data_utils.py': {'sha256': '619a763a1af3026747b166fd88d6cfa26966162d5c91b9ef4e23e9f1d13d3a52',
                                       'adaptation': 'root/routing/few-shot adapter; explicit pkl-first '
                                                     'common file selection/read/image decode; original '
                                                     'numpy-list policy'},
 'PFLlib/system/flcore/trainmodel/models.py': {'sha256': 'd90e11282e877b00cc9ac6be73c0630466abbeb8796b4de8554bf64cb9114956',
                                               'adaptation': 'single Torch-only class in common.models; '
                                                             'vendor and flcore compatibility exports; '
                                                             'identical constructor/forward statements'},
 'PFLlib/system/main.py': {'sha256': '4b7e0b51ed431a416668b611feac6179a4c2d038a53e76710271f0437468c0f4',
                           'adaptation': 'verbatim str2bool and argparse declarations only; no unrelated '
                                         'imports'},
 'configs/cifar100/noniid_dir_20_a0p5/base.yaml': {'sha256': '3aee134bd242990a994bcf9abc1f987587d7723a8c2060e065c637f5716f7f3d',
                                                   'adaptation': 'none'},
 'configs/cifar100/noniid_dir_20_a0p5/algorithms/fedfews.yaml': {'sha256': '32db4f84da46e9e478a9f81f443b3d23058a6107568c6ae3c9e1bda349c9b4e5',
                                                                 'adaptation': 'none'},
 'configs/cifar100/noniid_pat_10/base.yaml': {'sha256': 'e2b89464f098a1be2d3d907e9a9643e3c1104479bb7895d1455bc39f6baa4886',
                                              'adaptation': 'none'},
 'configs/cifar100/noniid_pat_10/algorithms/fedfews.yaml': {'sha256': 'c6ca4ab2de45698b1ee733a373af62d307393c023b61fe0daf677cebb40e83f8',
                                                            'adaptation': 'none'},
 'scripts/run_pfllib.py': {'sha256': '5fa9afcb47cb80a2e570233ce0232d69ac4cb5c88875927786b62086898a56f3',
                           'adaptation': 'reference config merge/command mapping'},
 'LICENSE': {'sha256': '981483e3fd5e3a77598b4c9d864add884e5330a98e7a4ecc4bb5099b3f346eb5',
             'adaptation': 'reference only; license copied verbatim'}}
