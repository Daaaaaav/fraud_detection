[33mcommit df5c816ab62a961e530f88985a39f6b2a7b7e912[m[33m ([m[1;36mHEAD[m[33m -> [m[1;32mmain[m[33m)[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Sat May 17 20:53:05 2025 +0700

    fixed dataset picker

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m9786[m -> [32m9933[m bytes
 __pycache__/isolationforest.cpython-312.pyc     | Bin [31m4923[m -> [32m5987[m bytes
 __pycache__/preprocessing.cpython-312.pyc       | Bin [31m3995[m -> [32m4748[m bytes
 __pycache__/randomforest.cpython-312.pyc        | Bin [31m3178[m -> [32m6257[m bytes
 app.py                                          |  12 [32m+[m[31m-[m
 autoencoder_backend.py                          |  51 [32m++++[m[31m-----[m
 isolationforest.py                              |  79 [32m+++++++++[m[31m----[m
 models/autoencoder.h5                           | Bin [31m166072[m -> [32m166072[m bytes
 models/encoder.h5                               | Bin [31m45120[m -> [32m45120[m bytes
 models/isolation_forest_model.pkl               | Bin [31m0[m -> [32m1019833[m bytes
 models/models/random_forest.pkl                 | Bin [31m0[m -> [32m12168793[m bytes
 models/random_forest.pkl                        | Bin [31m13056633[m -> [32m20577097[m bytes
 models/scaler.pkl                               | Bin [31m1863[m -> [32m1863[m bytes
 models/threshold.json                           |   1 [32m+[m
 preprocessing.py                                |  37 [32m++++[m[31m--[m
 randomforest.py                                 | 145 [32m++++++++++++++++++[m[31m------[m
 static/scripts.js                               |   5 [32m+[m[31m-[m
 17 files changed, 229 insertions(+), 101 deletions(-)

[33mcommit f21fbeaba311f18a3894490f6fadaf09da8d7615[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Fri May 16 17:45:41 2025 +0700

    New added

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m10214[m -> [32m9786[m bytes
 __pycache__/isolationforest.cpython-312.pyc     | Bin [31m6364[m -> [32m4923[m bytes
 __pycache__/preprocessing.cpython-312.pyc       | Bin [31m4107[m -> [32m3995[m bytes
 __pycache__/randomforest.cpython-312.pyc        | Bin [31m3162[m -> [32m3178[m bytes
 app.py                                          | 101 [32m++++++[m[31m--------[m
 autoencoder_backend.py                          | 170 [32m+++++++++++++[m[31m-----------[m
 isolationforest.py                              | 129 [32m++++++[m[31m------------[m
 models/autoencoder.h5                           | Bin [31m166072[m -> [32m166072[m bytes
 preprocessing.py                                | 144 [32m+++++++++[m[31m-----------[m
 randomforest.py                                 |  43 [32m+++[m[31m---[m
 rf_model.pkl                                    | Bin [31m2696057[m -> [32m2695881[m bytes
 11 files changed, 258 insertions(+), 329 deletions(-)

[33mcommit c75c73a8ddcdc5a64681c8bc89d9f1fb2a6f387e[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Fri May 16 16:09:55 2025 +0700

    Fixed Autoencoders

 __pycache__/autoencoder_backend.cpython-312.pyc |   Bin [31m8708[m -> [32m10214[m bytes
 autoencoder_backend.py                          |    78 [32m+[m[31m-[m
 autoencoder_loss_analysis.csv                   | 85447 [32m+[m[31m---------------------[m
 model_predictions.csv                           | 56966 [32m+[m[31m--------------[m
 models/autoencoder.h5                           |   Bin [31m460144[m -> [32m166072[m bytes
 models/encoder.h5                               |   Bin [31m93816[m -> [32m45120[m bytes
 models/random_forest.pkl                        |   Bin [31m6127369[m -> [32m13056633[m bytes
 static/scripts.js                               |     6 [32m+[m[31m-[m
 xgb_thresholds_analysis.csv                     | 55863 [32m+[m[31m-------------[m
 9 files changed, 64 insertions(+), 198296 deletions(-)

[33mcommit 14a083f11eac98690768d324b2b26745f9d1dfcf[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Thu May 15 18:12:13 2025 +0700

    Track large files with Git LFS

 .gitattributes      |      2 [32m+[m
 creditcard_2023.csv | 568634 [32m+[m[31m------------------------------------------------[m
 2 files changed, 5 insertions(+), 568631 deletions(-)

[33mcommit 46dbd28630f1dded91b78b25640adf993c542e4d[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Thu May 15 17:07:59 2025 +0700

    user input dataset

 __pycache__/autoencoder_backend.cpython-312.pyc |    Bin [31m8708[m -> [32m8708[m bytes
 __pycache__/preprocessing.cpython-312.pyc       |    Bin [31m3394[m -> [32m4107[m bytes
 __pycache__/randomforest.cpython-312.pyc        |    Bin [31m3084[m -> [32m3162[m bytes
 app.py                                          |     71 [32m+[m[31m-[m
 creditcard_2023.csv                             | 568631 [32m+++++++++++++++++++++[m
 feature_names_creditcard.npy                    |    Bin [31m0[m -> [32m452[m bytes
 feature_names_creditcard_2023.npy               |    Bin [31m0[m -> [32m450[m bytes
 models/autoencoder.h5                           |    Bin [31m460144[m -> [32m460144[m bytes
 models/encoder.h5                               |    Bin [31m93816[m -> [32m93816[m bytes
 models/random_forest.pkl                        |    Bin [31m6279017[m -> [32m6127369[m bytes
 models/rf_on_bottleneck.pkl                     |    Bin [31m0[m -> [32m6989801[m bytes
 preprocessing.py                                |     39 [32m+[m[31m-[m
 randomforest.py                                 |      3 [32m+[m[31m-[m
 rf_model.pkl                                    |    Bin [31m2695881[m -> [32m2696057[m bytes
 scaler_creditcard.pkl                           |    Bin [31m0[m -> [32m1927[m bytes
 scaler_creditcard_2023.pkl                      |    Bin [31m0[m -> [32m1927[m bytes
 static/scripts.js                               |     40 [32m+[m[31m-[m
 templates/index.html                            |     11 [32m+[m[31m-[m
 18 files changed, 568737 insertions(+), 58 deletions(-)

[33mcommit c849f5bd928cf47e7b97363af6987a20d914ab00[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue May 13 22:23:06 2025 +0700

    charts and autoencoder pred edit

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m6793[m -> [32m8708[m bytes
 app.py                                          |  48 [32m+++++[m[31m------[m
 autoencoder_backend.py                          | 106 [32m+++++++++++++++[m[31m---------[m
 models/autoencoder.h5                           | Bin [31m453856[m -> [32m460144[m bytes
 models/encoder.h5                               | Bin [31m92792[m -> [32m93816[m bytes
 models/random_forest.pkl                        | Bin [31m7875833[m -> [32m6279017[m bytes
 models/scaler.pkl                               | Bin [31m2455[m -> [32m1863[m bytes
 static/scripts.js                               |  31 [32m+++++[m[31m--[m
 templates/index.html                            |   4 [32m+[m[31m-[m
 9 files changed, 114 insertions(+), 75 deletions(-)

[33mcommit 95f086d5a5e2f0b82da51ee4edb86e151b8417dc[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Mon May 12 23:38:10 2025 +0700

    Charts Update (autoencoder needs fix)

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m6459[m -> [32m6793[m bytes
 __pycache__/isolationforest.cpython-312.pyc     | Bin [31m3324[m -> [32m6364[m bytes
 __pycache__/randomforest.cpython-312.pyc        | Bin [31m3085[m -> [32m3084[m bytes
 app.py                                          |  73 [32m++[m[31m----[m
 autoencoder_backend.py                          | 139 [32m+++++[m[31m-----[m
 isolation_forest_model.pkl                      | Bin [31m984281[m -> [32m35903881[m bytes
 isolationforest.py                              | 141 [32m+++++++[m[31m---[m
 models/autoencoder.h5                           | Bin [31m159448[m -> [32m453856[m bytes
 models/encoder.h5                               | Bin [31m0[m -> [32m92792[m bytes
 models/random_forest.pkl                        | Bin [31m0[m -> [32m7875833[m bytes
 models/threshold.txt                            |   2 [32m+[m[31m-[m
 models/threshold.txt.npy                        | Bin [31m136[m -> [32m136[m bytes
 randomforest.py                                 |  19 [32m+[m[31m-[m
 rf_model.pkl                                    | Bin [31m16786617[m -> [32m2695881[m bytes
 scaler.pkl                                      | Bin [31m1927[m -> [32m1927[m bytes
 static/scripts.js                               | 335 [32m+++++++++[m[31m---------------[m
 templates/index.html                            |  26 [32m+[m[31m-[m
 threshold.json                                  |   1 [32m+[m
 18 files changed, 359 insertions(+), 377 deletions(-)

[33mcommit f0e5485dcb1be1681a989b9254c536a844aa13ed[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Sat May 10 23:22:16 2025 +0700

    fraud charts check

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m6490[m -> [32m6459[m bytes
 __pycache__/isolationforest.cpython-312.pyc     | Bin [31m3324[m -> [32m3324[m bytes
 __pycache__/randomforest.cpython-312.pyc        | Bin [31m3085[m -> [32m3085[m bytes
 app.py                                          |  54 [32m+++[m[31m-[m
 autoencoder_backend.py                          |   4 [32m+[m[31m-[m
 models/autoencoder.h5                           | Bin [31m159448[m -> [32m159448[m bytes
 models/threshold.txt.npy                        | Bin [31m0[m -> [32m136[m bytes
 rf_model.pkl                                    | Bin [31m16786441[m -> [32m16786617[m bytes
 static/scripts.js                               | 321 [32m++++++++++++++[m[31m----------[m
 templates/index.html                            |   1 [32m+[m
 10 files changed, 249 insertions(+), 131 deletions(-)

[33mcommit 034f87e6a97eb1b598467fc18b1d363a04c642ef[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Thu May 8 14:31:45 2025 +0700

    scripts edit for prediction

 app.py                 |  94 [32m+++++++++++++++++[m[31m-------------------------------[m
 best_autoencoder.h5    | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy | Bin [31m136[m -> [32m136[m bytes
 static/scripts.js      |  96 [32m++++++++++++++++++++++++[m[31m-------------------------[m
 4 files changed, 80 insertions(+), 110 deletions(-)

[33mcommit 03edefcfc4c10fadb110fdc68ebcae601cb3dbb9[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed May 7 23:40:54 2025 +0700

    fraudulent example data added

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m6673[m -> [32m6490[m bytes
 autoencoder_backend.py                          |   5 [32m+[m[31m--[m
 static/scripts.js                               |  39 [32m++++++++++++++++++++++++[m
 templates/index.html                            |   1 [32m+[m
 4 files changed, 41 insertions(+), 4 deletions(-)

[33mcommit fc8e7106147fcd2b457d4467f266c23f3e405fa6[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed May 7 17:22:11 2025 +0700

    autoencoder predict update

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m5625[m -> [32m6673[m bytes
 autoencoder_backend.py                          |  25 [32m+++++++++++++++++++[m[31m-----[m
 best_autoencoder.h5                             | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy                          | Bin [31m136[m -> [32m136[m bytes
 static/scripts.js                               |  22 [32m+++++++++[m[31m------------[m
 5 files changed, 29 insertions(+), 18 deletions(-)

[33mcommit 36526d9c0c1cf5b35ee8173a830d35324a568fd1[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed May 7 11:48:29 2025 +0700

    toggle update

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m5625[m -> [32m5625[m bytes
 autoencoder_backend.py                          |   1 [31m-[m
 best_autoencoder.h5                             | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy                          | Bin [31m136[m -> [32m136[m bytes
 static/scripts.js                               |  25 [32m+++++++++++++[m[31m-----------[m
 5 files changed, 14 insertions(+), 12 deletions(-)

[33mcommit a092ccbbbab36065f3391157a377f0ae33528052[m
Merge: 5ef227b 4943673
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed May 7 11:27:40 2025 +0700

    fixed autoencoder training

[33mcommit 49436732a63fdb7bf77af30a6911a65f4f79890d[m
Author: nava <nava.simanjuntak@student.president.ac.id>
Date:   Tue May 6 15:05:02 2025 +0700

    fixed charts

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m5590[m -> [32m5581[m bytes
 __pycache__/isolationforest.cpython-312.pyc     | Bin [31m3296[m -> [32m3287[m bytes
 __pycache__/preprocessing.cpython-312.pyc       | Bin [31m3394[m -> [32m3385[m bytes
 __pycache__/randomforest.cpython-312.pyc        | Bin [31m3019[m -> [32m3010[m bytes
 app.py                                          |  48 [32m++[m[31m-[m
 feature_names.npy                               | Bin [31m452[m -> [32m577[m bytes
 isolation_forest_model.pkl                      | Bin [31m984281[m -> [32m973080[m bytes
 scaler.pkl                                      | Bin [31m1927[m -> [32m1927[m bytes
 static/scripts.js                               | 518 [32m+++++++++++++++[m[31m---------[m
 templates/index.html                            |  18 [32m+[m[31m-[m
 10 files changed, 360 insertions(+), 224 deletions(-)

[33mcommit 5ef227b84c8f5b6d17624a5872bea83e94901117[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue May 6 13:19:25 2025 +0700

    fixed evaluate

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m5590[m -> [32m6482[m bytes
 __pycache__/isolationforest.cpython-312.pyc     | Bin [31m3296[m -> [32m3324[m bytes
 __pycache__/randomforest.cpython-312.pyc        | Bin [31m3019[m -> [32m3085[m bytes
 app.py                                          |   2 [32m+[m
 autoencoder_backend.py                          |  34 [32m++++++++++[m[31m--[m
 best_autoencoder.h5                             | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy                          | Bin [31m136[m -> [32m136[m bytes
 isolationforest.py                              |   2 [32m+[m
 randomforest.py                                 |  16 [32m+++[m[31m---[m
 rf_model.pkl                                    | Bin [31m16786617[m -> [32m16786441[m bytes
 static/scripts.js                               |  68 [32m+++++++++++++++++++++[m[31m---[m
 11 files changed, 105 insertions(+), 17 deletions(-)

[33mcommit 4a6c2d6497ed83fdb0750b41c117df356fdff0d3[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Mon May 5 15:29:26 2025 +0700

    fixed evaluate example data

 best_autoencoder.h5    | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy | Bin [31m136[m -> [32m136[m bytes
 static/scripts.js      |  57 [32m++++++++++++++++++++++++++[m[31m-----------------------[m
 3 files changed, 30 insertions(+), 27 deletions(-)

[33mcommit 7d335c40e096be9f5320e4fad6ce9ce21d5c0117[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Mon May 5 15:00:42 2025 +0700

    fixed evaluate example data

 app.py                 |  46 [32m++++++++++++++++++++++++++++++++++++[m[31m----------[m
 best_autoencoder.h5    | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy | Bin [31m136[m -> [32m136[m bytes
 rf_model.pkl           | Bin [31m16786441[m -> [32m16786617[m bytes
 static/scripts.js      |  43 [32m+++++++++++++[m[31m------------------------------[m
 5 files changed, 49 insertions(+), 40 deletions(-)

[33mcommit bb9037c7ed438600f1ca93ea56668efed6520b0e[m
Author: Emilia Adinda Putri <emilsamsung596@gmail.com>
Date:   Wed Apr 30 21:16:10 2025 +0700

    app.py

 app.py | 78 [32m++++++++++++[m[31m------------------------------------------------------[m
 1 file changed, 14 insertions(+), 64 deletions(-)

[33mcommit 207d891f34375e50d38daa48888cd5af15295b54[m
Author: Emilia Adinda Putri <emilsamsung596@gmail.com>
Date:   Wed Apr 30 20:43:46 2025 +0700

     app.py

 app.py | 78 [32m++++++++++++++++++++++++++++++++++++++++++++++++++++++[m[31m------------[m
 1 file changed, 64 insertions(+), 14 deletions(-)

[33mcommit 43296b1864d29acc5501e2f7eaaa53d7e181d4b0[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 30 15:59:01 2025 +0700

    fixed autoencoder eval

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m6058[m -> [32m5590[m bytes
 app.py                                          |  29 [32m++++[m[31m--------------------[m
 autoencoder_backend.py                          |  27 [32m++++[m[31m------------------[m
 best_autoencoder.h5                             | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy                          | Bin [31m136[m -> [32m136[m bytes
 static/scripts.js                               |  17 [32m+++++[m[31m---------[m
 6 files changed, 15 insertions(+), 58 deletions(-)

[33mcommit 46b672316df617e5d662a9ff196e810e98f224a8[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 30 14:36:54 2025 +0700

    fixed autoencoder

 app.py               |   5 [32m+[m[31m----[m
 best_autoencoder.h5  | Bin [31m159448[m -> [32m159448[m bytes
 static/scripts.js    |  58 [32m++++++++++++++++++++++++++++++++++++++++++++++++[m[31m---[m
 templates/index.html |   4 [32m++[m[31m--[m
 4 files changed, 58 insertions(+), 9 deletions(-)

[33mcommit de161f40b15b9ee42ce4463f49be1fba12837348[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 30 13:59:54 2025 +0700

    added input

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m6030[m -> [32m6058[m bytes
 __pycache__/isolationforest.cpython-312.pyc     | Bin [31m3268[m -> [32m3296[m bytes
 __pycache__/preprocessing.cpython-312.pyc       | Bin [31m3366[m -> [32m3394[m bytes
 __pycache__/randomforest.cpython-312.pyc        | Bin [31m2991[m -> [32m3019[m bytes
 app.py                                          |  84 [32m+++[m[31m-[m
 static/scripts.js                               | 496 [32m++++++++++++++[m[31m----------[m
 templates/index.html                            |  18 [32m+[m
 7 files changed, 383 insertions(+), 215 deletions(-)

[33mcommit 69d2f6c5c795e2790728057872509ab7a1c89170[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Thu Apr 24 16:30:21 2025 +0700

    fraud chart update

 __pycache__/isolationforest.cpython-312.pyc | Bin [31m2904[m -> [32m3268[m bytes
 isolationforest.py                          |  23 [32m+++++[m[31m--[m
 static/scripts.js                           |  93 [32m+++++[m[31m-----------------------[m
 3 files changed, 36 insertions(+), 80 deletions(-)

[33mcommit b538e2c082318c3cdd206cdb4f06e02827ea0696[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Thu Apr 24 16:02:58 2025 +0700

    charts edit

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m6030[m -> [32m6030[m bytes
 __pycache__/isolationforest.cpython-312.pyc     | Bin [31m2686[m -> [32m2904[m bytes
 app.py                                          |   4 [32m++++[m
 best_autoencoder.h5                             | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy                          | Bin [31m136[m -> [32m136[m bytes
 isolationforest.py                              |  29 [32m+++++++++++++++[m[31m---------[m
 static/scripts.js                               |  26 [32m+++++++++++++[m[31m--------[m
 static/styles.css                               |  13 [32m+++++++++++[m
 8 files changed, 52 insertions(+), 20 deletions(-)

[33mcommit fd68cd9f2c2e79719c6bc5356e85681d47ac4ecd[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Thu Apr 24 12:51:14 2025 +0700

    UI and fetch edits

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m6034[m -> [32m6030[m bytes
 __pycache__/isolationforest.cpython-312.pyc     | Bin [31m1806[m -> [32m2686[m bytes
 __pycache__/preprocessing.cpython-312.pyc       | Bin [31m2935[m -> [32m3366[m bytes
 __pycache__/randomforest.cpython-312.pyc        | Bin [31m2020[m -> [32m2991[m bytes
 app.py                                          |  77 [32m++++++[m[31m--[m
 autoencoder_backend.py                          |   8 [32m+[m[31m-[m
 best_autoencoder.h5                             | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy                          | Bin [31m136[m -> [32m136[m bytes
 feature_names.npy                               | Bin [31m0[m -> [32m452[m bytes
 isolationforest.py                              |  32 [32m+++[m[31m-[m
 preprocessing.py                                |  32 [32m++[m[31m--[m
 randomforest.py                                 |  43 [32m+++[m[31m--[m
 rf_model.pkl                                    | Bin [31m16786393[m -> [32m16786441[m bytes
 scaler.pkl                                      | Bin [31m0[m -> [32m1927[m bytes
 static/scripts.js                               | 227 [32m+++++++++++++[m[31m-----------[m
 static/styles.css                               | 139 [32m++++++++++++[m[31m---[m
 templates/index.html                            |  32 [32m++[m[31m--[m
 17 files changed, 402 insertions(+), 188 deletions(-)

[33mcommit 4f39d87030e4fa2f640a435077e040bef16af757[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 23 15:28:27 2025 +0700

    fixed Dockerfile 4

 Dockerfile | 1 [32m+[m
 app.py     | 2 [32m+[m[31m-[m
 2 files changed, 2 insertions(+), 1 deletion(-)

[33mcommit e9541aa5d2abf6d92b735d223d7cbddd7e669267[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 23 15:13:37 2025 +0700

    fixed Dockerfile 3

 Dockerfile | 10 [32m+++++++[m[31m---[m
 1 file changed, 7 insertions(+), 3 deletions(-)

[33mcommit e8ee5fec7651e5677296f8757b3398237a51143e[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 23 15:09:01 2025 +0700

    fixed Dockerfile 2

 Dockerfile | 2 [32m+[m[31m-[m
 1 file changed, 1 insertion(+), 1 deletion(-)

[33mcommit 78db1196c3f568fdb6f0da24c036092ca7c56576[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 23 14:55:18 2025 +0700

    fixed Dockerfile

 Dockerfile | 8 [32m++++[m[31m----[m
 1 file changed, 4 insertions(+), 4 deletions(-)

[33mcommit 000eb92c0219056b4805707410466088bf7e8036[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 23 14:50:04 2025 +0700

    fixed dependencies 2

 best_autoencoder.h5    | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy | Bin [31m136[m -> [32m136[m bytes
 2 files changed, 0 insertions(+), 0 deletions(-)

[33mcommit 663f5924e6ca379ee9aa6d12a2520542f3b6790c[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 23 14:37:29 2025 +0700

    fixed dependencies

 Dockerfile | 2 [32m++[m
 1 file changed, 2 insertions(+)

[33mcommit 9f540e95cad0ba6e0ac803227f1df40ce6af1f5f[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 23 14:33:21 2025 +0700

    added Dockerfile

 Dockerfile | 17 [32m+++++++++++++++++[m
 1 file changed, 17 insertions(+)

[33mcommit 63bf4a9c45524339898f018e714c5ecd84591537[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 23 14:18:48 2025 +0700

    added imblearn library

 requirements.txt | Bin [31m12224[m -> [32m12260[m bytes
 1 file changed, 0 insertions(+), 0 deletions(-)

[33mcommit b7517a4be73bc9688a8ec4a2bf378418669e9d81[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 23 14:14:01 2025 +0700

    fixed chart max

 static/scripts.js | 2 [32m+[m[31m-[m
 1 file changed, 1 insertion(+), 1 deletion(-)

[33mcommit 6d08a91ab1bebe079ca23ef987c950369c3deb84[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue Apr 22 16:30:07 2025 +0700

    fixed eval

 best_autoencoder.h5    | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy | Bin [31m136[m -> [32m136[m bytes
 static/scripts.js      |   7 [32m++++++[m[31m-[m
 3 files changed, 6 insertions(+), 1 deletion(-)

[33mcommit 4711f9033ce23fd6bf64d51016b921ca371323f4[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue Apr 22 15:48:42 2025 +0700

    fixed randomforest

 __pycache__/randomforest.cpython-312.pyc | Bin [31m2020[m -> [32m2020[m bytes
 app.py                                   |  67 [32m+++++++++++++++[m[31m-----------[m
 best_autoencoder.h5                      | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy                   | Bin [31m136[m -> [32m136[m bytes
 static/scripts.js                        |  78 [32m+++++++++++++++++[m[31m--------------[m
 templates/index.html                     |   4 [32m+[m[31m-[m
 6 files changed, 85 insertions(+), 64 deletions(-)

[33mcommit 54821372e5c6ecac3892f4cbe3a7a750d90fa4fc[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue Apr 22 14:16:55 2025 +0700

    deployment setup 3

 requirements.txt | Bin [31m94[m -> [32m12224[m bytes
 1 file changed, 0 insertions(+), 0 deletions(-)

[33mcommit 2002fe6277031d059a98b4da1ec7e036c2c87fd1[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue Apr 22 14:14:44 2025 +0700

    deployment setup 2

 requirements.txt | 7 [32m++++[m[31m---[m
 1 file changed, 4 insertions(+), 3 deletions(-)

[33mcommit ef25a61a0fd3a058f6d8313e49ad3a6d86939ca3[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue Apr 22 14:09:20 2025 +0700

    deployment setup

 Procfile                                        |   1 [32m+[m
 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m5466[m -> [32m6034[m bytes
 autoencoder_backend.py                          |  25 [32m++++[m[31m-[m
 best_autoencoder.h5                             | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy                          | Bin [31m136[m -> [32m136[m bytes
 requirements.txt                                |   9 [32m++[m
 static/scripts.js                               | 133 [32m+++++++++++++++[m[31m---------[m
 7 files changed, 116 insertions(+), 52 deletions(-)

[33mcommit 36f81c18dda1a524a785f2f16671fdb0f9f9628d[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Mon Apr 21 16:29:33 2025 +0700

    autoencoder page fixed

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m5605[m -> [32m5466[m bytes
 app.py                                          |  31 [32m+++++++++[m[31m---------[m
 autoencoder_backend.py                          |  40 [32m+++++++++++++[m[31m-----------[m
 best_autoencoder.h5                             | Bin [31m159448[m -> [32m159448[m bytes
 best_threshold.txt.npy                          | Bin [31m0[m -> [32m136[m bytes
 models/autoencoder.h5                           | Bin [31m0[m -> [32m159448[m bytes
 models/scaler.pkl                               | Bin [31m0[m -> [32m2455[m bytes
 models/threshold.txt                            |   1 [32m+[m
 static/scripts.js                               |  12 [32m+++++[m[31m--[m
 9 files changed, 46 insertions(+), 38 deletions(-)

[33mcommit 3507c4963f4797e0ae8cc7ee8fdbe0c587f230c1[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Sun Apr 20 23:00:29 2025 +0700

    added autoencoder page

 __pycache__/autoencoder_backend.cpython-312.pyc | Bin [31m0[m -> [32m5605[m bytes
 __pycache__/isolationforest.cpython-312.pyc     | Bin [31m1806[m -> [32m1806[m bytes
 __pycache__/randomforest.cpython-312.pyc        | Bin [31m1998[m -> [32m2020[m bytes
 app.py                                          |  25 [32m+++[m[31m-[m
 autoencoder_backend.py                          |  93 [32m+++++++++++++[m
 autoencoders.py                                 |   8 [32m+[m[31m-[m
 best_autoencoder.h5                             | Bin [31m159448[m -> [32m159448[m bytes
 static/scripts.js                               | 173 [32m+++++++++++++[m[31m-----------[m
 templates/index.html                            |  11 [32m++[m
 9 files changed, 223 insertions(+), 87 deletions(-)

[33mcommit 3711e64078aeb45afede15850cdc60b4deb10886[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Sat Apr 19 19:02:14 2025 +0700

    autoencoders fixed

 autoencoder_loss_analysis.csv                      | 85444 [32m+++++++++++++++++++[m
 autoencoder_tuning/creditcard_fraud/oracle.json    |     1 [32m+[m
 .../creditcard_fraud/trial_0000/build_config.json  |     1 [32m+[m
 .../trial_0000/checkpoint.weights.h5               |   Bin [31m0[m -> [32m211432[m bytes
 .../creditcard_fraud/trial_0000/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0001/build_config.json  |     1 [32m+[m
 .../trial_0001/checkpoint.weights.h5               |   Bin [31m0[m -> [32m119728[m bytes
 .../creditcard_fraud/trial_0001/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0002/build_config.json  |     1 [32m+[m
 .../trial_0002/checkpoint.weights.h5               |   Bin [31m0[m -> [32m118256[m bytes
 .../creditcard_fraud/trial_0002/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0003/build_config.json  |     1 [32m+[m
 .../trial_0003/checkpoint.weights.h5               |   Bin [31m0[m -> [32m214952[m bytes
 .../creditcard_fraud/trial_0003/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0004/build_config.json  |     1 [32m+[m
 .../trial_0004/checkpoint.weights.h5               |   Bin [31m0[m -> [32m154736[m bytes
 .../creditcard_fraud/trial_0004/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0005/build_config.json  |     1 [32m+[m
 .../trial_0005/checkpoint.weights.h5               |   Bin [31m0[m -> [32m136040[m bytes
 .../creditcard_fraud/trial_0005/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0006/build_config.json  |     1 [32m+[m
 .../trial_0006/checkpoint.weights.h5               |   Bin [31m0[m -> [32m167912[m bytes
 .../creditcard_fraud/trial_0006/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0007/build_config.json  |     1 [32m+[m
 .../trial_0007/checkpoint.weights.h5               |   Bin [31m0[m -> [32m173936[m bytes
 .../creditcard_fraud/trial_0007/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0008/build_config.json  |     1 [32m+[m
 .../trial_0008/checkpoint.weights.h5               |   Bin [31m0[m -> [32m199432[m bytes
 .../creditcard_fraud/trial_0008/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0009/build_config.json  |     1 [32m+[m
 .../trial_0009/checkpoint.weights.h5               |   Bin [31m0[m -> [32m174944[m bytes
 .../creditcard_fraud/trial_0009/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0010/build_config.json  |     1 [32m+[m
 .../trial_0010/checkpoint.weights.h5               |   Bin [31m0[m -> [32m167400[m bytes
 .../creditcard_fraud/trial_0010/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0011/build_config.json  |     1 [32m+[m
 .../trial_0011/checkpoint.weights.h5               |   Bin [31m0[m -> [32m126416[m bytes
 .../creditcard_fraud/trial_0011/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0012/build_config.json  |     1 [32m+[m
 .../trial_0012/checkpoint.weights.h5               |   Bin [31m0[m -> [32m98544[m bytes
 .../creditcard_fraud/trial_0012/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0013/build_config.json  |     1 [32m+[m
 .../trial_0013/checkpoint.weights.h5               |   Bin [31m0[m -> [32m194056[m bytes
 .../creditcard_fraud/trial_0013/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0014/build_config.json  |     1 [32m+[m
 .../trial_0014/checkpoint.weights.h5               |   Bin [31m0[m -> [32m145512[m bytes
 .../creditcard_fraud/trial_0014/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0015/build_config.json  |     1 [32m+[m
 .../trial_0015/checkpoint.weights.h5               |   Bin [31m0[m -> [32m248944[m bytes
 .../creditcard_fraud/trial_0015/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0016/build_config.json  |     1 [32m+[m
 .../trial_0016/checkpoint.weights.h5               |   Bin [31m0[m -> [32m90608[m bytes
 .../creditcard_fraud/trial_0016/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0017/build_config.json  |     1 [32m+[m
 .../trial_0017/checkpoint.weights.h5               |   Bin [31m0[m -> [32m118376[m bytes
 .../creditcard_fraud/trial_0017/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0018/build_config.json  |     1 [32m+[m
 .../trial_0018/checkpoint.weights.h5               |   Bin [31m0[m -> [32m130976[m bytes
 .../creditcard_fraud/trial_0018/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0019/build_config.json  |     1 [32m+[m
 .../trial_0019/checkpoint.weights.h5               |   Bin [31m0[m -> [32m109944[m bytes
 .../creditcard_fraud/trial_0019/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0020/build_config.json  |     1 [32m+[m
 .../trial_0020/checkpoint.weights.h5               |   Bin [31m0[m -> [32m136432[m bytes
 .../creditcard_fraud/trial_0020/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0021/build_config.json  |     1 [32m+[m
 .../trial_0021/checkpoint.weights.h5               |   Bin [31m0[m -> [32m145512[m bytes
 .../creditcard_fraud/trial_0021/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0022/build_config.json  |     1 [32m+[m
 .../trial_0022/checkpoint.weights.h5               |   Bin [31m0[m -> [32m90864[m bytes
 .../creditcard_fraud/trial_0022/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0023/build_config.json  |     1 [32m+[m
 .../trial_0023/checkpoint.weights.h5               |   Bin [31m0[m -> [32m193184[m bytes
 .../creditcard_fraud/trial_0023/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0024/build_config.json  |     1 [32m+[m
 .../trial_0024/checkpoint.weights.h5               |   Bin [31m0[m -> [32m184168[m bytes
 .../creditcard_fraud/trial_0024/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0025/build_config.json  |     1 [32m+[m
 .../trial_0025/checkpoint.weights.h5               |   Bin [31m0[m -> [32m136040[m bytes
 .../creditcard_fraud/trial_0025/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0026/build_config.json  |     1 [32m+[m
 .../trial_0026/checkpoint.weights.h5               |   Bin [31m0[m -> [32m142880[m bytes
 .../creditcard_fraud/trial_0026/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0027/build_config.json  |     1 [32m+[m
 .../trial_0027/checkpoint.weights.h5               |   Bin [31m0[m -> [32m117360[m bytes
 .../creditcard_fraud/trial_0027/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0028/build_config.json  |     1 [32m+[m
 .../trial_0028/checkpoint.weights.h5               |   Bin [31m0[m -> [32m250752[m bytes
 .../creditcard_fraud/trial_0028/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0029/build_config.json  |     1 [32m+[m
 .../trial_0029/checkpoint.weights.h5               |   Bin [31m0[m -> [32m105696[m bytes
 .../creditcard_fraud/trial_0029/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0030/build_config.json  |     1 [32m+[m
 .../trial_0030/checkpoint.weights.h5               |   Bin [31m0[m -> [32m80880[m bytes
 .../creditcard_fraud/trial_0030/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0031/build_config.json  |     1 [32m+[m
 .../trial_0031/checkpoint.weights.h5               |   Bin [31m0[m -> [32m98784[m bytes
 .../creditcard_fraud/trial_0031/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0032/build_config.json  |     1 [32m+[m
 .../trial_0032/checkpoint.weights.h5               |   Bin [31m0[m -> [32m230128[m bytes
 .../creditcard_fraud/trial_0032/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0033/build_config.json  |     1 [32m+[m
 .../trial_0033/checkpoint.weights.h5               |   Bin [31m0[m -> [32m174440[m bytes
 .../creditcard_fraud/trial_0033/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0034/build_config.json  |     1 [32m+[m
 .../trial_0034/checkpoint.weights.h5               |   Bin [31m0[m -> [32m193184[m bytes
 .../creditcard_fraud/trial_0034/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0035/build_config.json  |     1 [32m+[m
 .../trial_0035/checkpoint.weights.h5               |   Bin [31m0[m -> [32m250752[m bytes
 .../creditcard_fraud/trial_0035/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0036/build_config.json  |     1 [32m+[m
 .../trial_0036/checkpoint.weights.h5               |   Bin [31m0[m -> [32m248944[m bytes
 .../creditcard_fraud/trial_0036/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0037/build_config.json  |     1 [32m+[m
 .../trial_0037/checkpoint.weights.h5               |   Bin [31m0[m -> [32m167400[m bytes
 .../creditcard_fraud/trial_0037/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0038/build_config.json  |     1 [32m+[m
 .../trial_0038/checkpoint.weights.h5               |   Bin [31m0[m -> [32m136040[m bytes
 .../creditcard_fraud/trial_0038/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0039/build_config.json  |     1 [32m+[m
 .../trial_0039/checkpoint.weights.h5               |   Bin [31m0[m -> [32m199432[m bytes
 .../creditcard_fraud/trial_0039/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0040/build_config.json  |     1 [32m+[m
 .../trial_0040/checkpoint.weights.h5               |   Bin [31m0[m -> [32m174440[m bytes
 .../creditcard_fraud/trial_0040/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0041/build_config.json  |     1 [32m+[m
 .../trial_0041/checkpoint.weights.h5               |   Bin [31m0[m -> [32m174944[m bytes
 .../creditcard_fraud/trial_0041/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0042/build_config.json  |     1 [32m+[m
 .../trial_0042/checkpoint.weights.h5               |   Bin [31m0[m -> [32m136040[m bytes
 .../creditcard_fraud/trial_0042/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0043/build_config.json  |     1 [32m+[m
 .../trial_0043/checkpoint.weights.h5               |   Bin [31m0[m -> [32m167912[m bytes
 .../creditcard_fraud/trial_0043/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0044/build_config.json  |     1 [32m+[m
 .../trial_0044/checkpoint.weights.h5               |   Bin [31m0[m -> [32m145512[m bytes
 .../creditcard_fraud/trial_0044/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0045/build_config.json  |     1 [32m+[m
 .../trial_0045/checkpoint.weights.h5               |   Bin [31m0[m -> [32m126416[m bytes
 .../creditcard_fraud/trial_0045/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0046/build_config.json  |     1 [32m+[m
 .../trial_0046/checkpoint.weights.h5               |   Bin [31m0[m -> [32m248944[m bytes
 .../creditcard_fraud/trial_0046/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0047/build_config.json  |     1 [32m+[m
 .../trial_0047/checkpoint.weights.h5               |   Bin [31m0[m -> [32m250752[m bytes
 .../creditcard_fraud/trial_0047/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0048/build_config.json  |     1 [32m+[m
 .../trial_0048/checkpoint.weights.h5               |   Bin [31m0[m -> [32m193184[m bytes
 .../creditcard_fraud/trial_0048/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0049/build_config.json  |     1 [32m+[m
 .../trial_0049/checkpoint.weights.h5               |   Bin [31m0[m -> [32m167400[m bytes
 .../creditcard_fraud/trial_0049/trial.json         |     1 [32m+[m
 .../creditcard_fraud/trial_0050/build_config.json  |     1 [32m+[m
 .../trial_0050/checkpoint.weights.h5               |   Bin [31m0[m -> [32m248944[m bytes
 .../creditcard_fraud/trial_0050/trial.json         |     1 [32m+[m
 .../trial_0051/checkpoint.weights.h5               |   Bin [31m0[m -> [32m250752[m bytes
 .../creditcard_fraud/trial_0051/trial.json         |     1 [32m+[m
 autoencoder_tuning/creditcard_fraud/tuner0.json    |     1 [32m+[m
 autoencoders.py                                    |   255 [32m+[m[31m-[m
 best_autoencoder.h5                                |   Bin [31m0[m -> [32m159448[m bytes
 best_threshold.txt                                 |     1 [32m+[m
 f1_vs_threshold.png                                |   Bin [31m0[m -> [32m41517[m bytes
 precision_recall_vs_threshold.png                  |   Bin [31m0[m -> [32m40855[m bytes
 threshold_distribution.png                         |   Bin [31m0[m -> [32m29382[m bytes
 training_loss.png                                  |   Bin [31m0[m -> [32m27694[m bytes
 165 files changed, 85631 insertions(+), 174 deletions(-)

[33mcommit 57cd153eb69e2dda6b82c39c7cbcb59dfe6a6c90[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Fri Apr 18 14:24:36 2025 +0700

    random forest eval fixed

 __pycache__/randomforest.cpython-312.pyc | Bin [31m1975[m -> [32m1998[m bytes
 app.py                                   |  18 [32m+++++++++++++[m[31m-----[m
 randomforest.py                          |  18 [32m+++++++++++++[m[31m-----[m
 rf_model.pkl                             | Bin [31m0[m -> [32m16786393[m bytes
 4 files changed, 26 insertions(+), 10 deletions(-)

[33mcommit 7177335c70cc58ce99b2ecd9e749bb3eca75dc7c[m
Author: ragilmi <ragilmaulanaa875@gmail.com>
Date:   Wed Apr 16 17:52:46 2025 +0700

    Add files via upload

 randomforest.py      |   7 [32m+[m[31m--[m
 static/scripts.js    | 140 [32m+++++++++++++++++++++++++++++[m[31m----------------------[m
 templates/index.html |  12 [32m+++[m[31m--[m
 3 files changed, 92 insertions(+), 67 deletions(-)

[33mcommit a5d3150f92597ca56f5ec870f502ceb31a0df175[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 16 15:49:58 2025 +0700

    modified training and eval

 __pycache__/isolationforest.cpython-312.pyc | Bin [31m1197[m -> [32m1806[m bytes
 __pycache__/randomforest.cpython-312.pyc    | Bin [31m1411[m -> [32m1975[m bytes
 isolationforest.py                          |  20 [32m+++[m[31m-[m
 randomforest.py                             |  21 [32m+++[m[31m-[m
 static/scripts.js                           | 146 [32m++++++++++++++++++++++[m[31m------[m
 5 files changed, 152 insertions(+), 35 deletions(-)

[33mcommit f504ffc2d00aedc6eb009ab46284cd7b688173b4[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Wed Apr 16 15:33:35 2025 +0700

    fixed training and eval design

 __pycache__/isolationforest.cpython-312.pyc | Bin [31m5230[m -> [32m1197[m bytes
 __pycache__/preprocessing.cpython-312.pyc   | Bin [31m2360[m -> [32m2935[m bytes
 __pycache__/randomforest.cpython-312.pyc    | Bin [31m1326[m -> [32m1411[m bytes
 app.py                                      |  50 [32m+++[m[31m---[m
 isolation_forest_model.pkl                  | Bin [31m853885[m -> [32m984281[m bytes
 static/scripts.js                           | 214 [32m+++++++++++++++[m[31m-------[m
 static/styles.css                           | 269 [32m+++++++++++++++++++++++++[m[31m---[m
 templates/index.html                        |  97 [32m+++++++[m[31m---[m
 8 files changed, 477 insertions(+), 153 deletions(-)

[33mcommit 2b56c89a8d42408fff17d6cd2800d57c620f4160[m
Author: ragilmi <ragilmaulanaa875@gmail.com>
Date:   Tue Apr 15 17:23:40 2025 +0700

    Add files via upload

 app.py               |  43 [32m+++++++++++++[m[31m----[m
 isolationforest.py   | 132 [32m+++++[m[31m----------------------------------------------[m
 preprocessing.py     |  65 [32m+++++++++++++[m[31m------------[m
 randomforest.py      |  27 [32m++++++[m[31m-----[m
 static/scripts.js    |  77 [32m++++++++++++++++++++++++++++++[m
 static/styles.css    |  30 [32m++++++++++++[m
 templates/index.html |  40 [32m++++++++++++++++[m
 7 files changed, 241 insertions(+), 173 deletions(-)

[33mcommit 1aea713d5f9127a20fbe412b1ff3259395cecbbc[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue Apr 15 13:50:57 2025 +0700

    backend ports update

 __pycache__/isolationforest.cpython-312.pyc | Bin [31m5090[m -> [32m5230[m bytes
 app.py                                      |  33 [32m++[m[31m--[m
 isolationforest.py                          | 248 [32m++++++++++++++[m[31m--------------[m
 3 files changed, 140 insertions(+), 141 deletions(-)

[33mcommit ba0dbc680e5a3d891f2b09282633f8ddad3da129[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue Apr 15 13:34:02 2025 +0700

    add backend

 __pycache__/isolationforest.cpython-312.pyc | Bin [31m0[m -> [32m5090[m bytes
 __pycache__/preprocessing.cpython-312.pyc   | Bin [31m2606[m -> [32m2360[m bytes
 __pycache__/randomforest.cpython-312.pyc    | Bin [31m0[m -> [32m1326[m bytes
 app.py                                      |  70 [32m+++++++++++++++++++[m
 preprocessing.py                            | 105 [32m++++++++++++++[m[31m--------------[m
 randomforest.py                             |  73 [32m++++[m[31m---------------[m
 6 files changed, 136 insertions(+), 112 deletions(-)

[33mcommit c6d382418e0418265c9417ff40c49720c84d3256[m
Merge: cfa6b3b 1dcfa3f
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue Apr 15 12:05:20 2025 +0700

    Merge branch 'master'
    yes

[33mcommit cfa6b3b12ad6905491ab3ccb29dcd4166f4b11d1[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue Apr 15 12:04:05 2025 +0700

    first be

 .gitignore         |    2 [32m+[m
 git-filter-repo.py | 4984 [32m++++++++++++++++++++++++++++++++++++++++++++++++++++[m
 2 files changed, 4986 insertions(+)

[33mcommit 1dcfa3ff18c6487b10711754becba0da4916f411[m
Author: Daaaaaav <scarletstormsubs@gmail.com>
Date:   Tue Apr 15 11:07:03 2025 +0700

    first

 __pycache__/preprocessing.cpython-312.pyc |   Bin [31m0[m -> [32m2606[m bytes
 autoencoders.py                           |   188 [32m+[m
 isolation_forest_evaluation.json          |    40 [32m+[m
 isolation_forest_model.pkl                |   Bin [31m0[m -> [32m853885[m bytes
 isolationforest.py                        |   136 [32m+[m
 model_predictions.csv                     | 56963 [32m++++++++++++++++++++++++++++[m
 preprocessing.py                          |    59 [32m+[m
 randomforest.py                           |    67 [32m+[m
 sagemaker.py                              |    51 [32m+[m
 xgb_thresholds_analysis.csv               | 55860 [32m+++++++++++++++++++++++++++[m
 xgboost_evaluation.json                   |    41 [32m+[m
 xgboost_model.pkl                         |   Bin [31m0[m -> [32m330492[m bytes
 12 files changed, 113405 insertions(+)

[33mcommit a0e703c3ceb4374b27e18d0e69edf1582e7f6bce[m
Author: Davina Ritzky Amarina <144467455+Daaaaaav@users.noreply.github.com>
Date:   Tue Apr 15 11:03:59 2025 +0700

    Initial commit

 README.md | 1 [32m+[m
 1 file changed, 1 insertion(+)
