# Comprehensive model workflow report

## Dataset and Metrics
- Source protocol: root `train.csv` (3875 windows) for fitting, `val.csv` (1045 windows) for selection, `train_val_combined.csv` (4921 windows) for retraining, `test.csv` (1045 windows) for one saved-artifact evaluation.
- Features: `['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz', 'Speed']`; window=156, stride=78.
- Metrics: accuracy, macro-F1, class-wise precision/recall/F1, and deployment latency.

## Model Explanations and Hyperparameters
Tabular models use configured flattened-window builders; neural, transformer, convolutional, recurrent, and state-space models use the existing project builders and are exported through their registered deployment adapters.

## Metrics for Every Model
               model    status error  search_candidates                                                                   best_config                                            artifact  val_accuracy  val_macro_f1  test_accuracy  test_macro_f1  test_latency_mean_ms  test_latency_p95_ms  test_windows_evaluated_once
histgradientboosting evaluated                        2 {"learning_rate": 0.05, "max_bins": 32, "max_iter": 40, "max_leaf_nodes": 63} models\workflow_deployment\histgradientboosting.pkl      0.846890      0.650993       0.704306       0.683755              0.087929             0.087929                         1045
            lightgbm evaluated                        2                {"learning_rate": 0.05, "n_estimators": 120, "num_leaves": 31}             models\workflow_deployment\lightgbm.pkl      0.878469      0.759565       0.745455       0.729397              0.045794             0.045794                         1045
       random_forest evaluated                        2                 {"max_depth": 24, "min_samples_leaf": 2, "n_estimators": 100}        models\workflow_deployment\random_forest.pkl      0.848804      0.627725       0.656459       0.607820              0.159836             0.159836                         1045
             xgboost evaluated                        2                  {"learning_rate": 0.03, "max_depth": 6, "n_estimators": 140}             models\workflow_deployment\xgboost.json      0.864115      0.689990       0.705263       0.680131              0.011138             0.011138                         1045
 logistic_regression evaluated                        2                                                                    {"C": 0.5}  models\workflow_deployment\logistic_regression.pkl      0.728230      0.568326       0.675598       0.666963              0.017454             0.017454                         1045
          linear_svm evaluated                        2                                                                    {"C": 0.5}           models\workflow_deployment\linear_svm.pkl      0.711962      0.530938       0.666986       0.653580              0.008553             0.008553                         1045
         mlp_tabular evaluated                        2                                                                 {"epochs": 2}        models\workflow_deployment\mlp_tabular.keras      0.768421      0.524386       0.637321       0.570322              0.767223             0.767223                         1045
    conv_transformer evaluated                        2                                                                 {"epochs": 2}  models\workflow_deployment\conv_transformer.tflite      0.816268      0.487834       0.623923       0.529276              0.852688             2.266520                         1045
                 gru evaluated                        2                                                                 {"epochs": 2}               models\workflow_deployment\gru.tflite      0.788517      0.381265       0.620096       0.515502              3.553209             3.553209                         1045
                lstm evaluated                        2                                                                 {"epochs": 2}              models\workflow_deployment\lstm.tflite      0.785646      0.340620       0.577033       0.420221              2.529173             2.529173                         1045
           fusformer evaluated                        2                                                                 {"epochs": 2}         models\workflow_deployment\fusformer.tflite      0.822010      0.514199       0.635407       0.557779              0.456116             1.380140                         1045
                mega evaluated                        2                                                                 {"epochs": 2}              models\workflow_deployment\mega.tflite      0.836364      0.578265       0.638278       0.553962              0.466156             1.396880                         1045
         convnet_har evaluated                        2                                                                 {"epochs": 2}       models\workflow_deployment\convnet_har.tflite      0.806699      0.499294       0.700478       0.669705              0.164055             0.380900                         1045
            tcnca_v3 evaluated                        2                                                                 {"epochs": 2}          models\workflow_deployment\tcnca_v3.tflite      0.735885      0.293208       0.592344       0.542232              0.151222             0.578360                         1045
             rbf_svm evaluated                        2                                                  {"C": 8.0, "gamma": "scale"}              models\workflow_deployment\rbf_svm.pkl      0.310048      0.434048       0.599043       0.620380              1.135314             1.135314                         1045
            tcnca_v2 evaluated                        2                                                                 {"epochs": 2}          models\workflow_deployment\tcnca_v2.tflite      0.738756      0.283110       0.616268       0.557738              0.167771             0.604100                         1045
            tcnca_v1 evaluated                        2                                                                 {"epochs": 2}          models\workflow_deployment\tcnca_v1.tflite      0.811483      0.495864       0.625837       0.538009              0.139428             0.393700                         1045
                 tst evaluated                        2                                                                 {"epochs": 2}               models\workflow_deployment\tst.tflite      0.798086      0.366017       0.570335       0.366518              0.724554             2.042620                         1045
                 s4d evaluated                        2                                                                 {"epochs": 2}                 models\workflow_deployment\s4d.onnx      0.356938      0.224761       0.288995       0.235492              0.117034             0.117034                         1045
             s5_onnx evaluated                        2                                                                 {"epochs": 2}             models\workflow_deployment\s5_onnx.onnx      0.356938      0.224761       0.288995       0.235492              0.178452             0.178452                         1045
                 lru evaluated                        2                                                                 {"epochs": 2}                 models\workflow_deployment\lru.onnx      0.356938      0.224761       0.288995       0.235492              0.119175             0.119175                         1045
              sgconv evaluated                        2                                                                 {"epochs": 2}              models\workflow_deployment\sgconv.onnx      0.109091      0.185018       0.327273       0.261494              0.153690             0.153690                         1045
            resnet1d evaluated                        2                                                                 {"epochs": 2}            models\workflow_deployment\resnet1d.onnx      0.641148      0.157001       0.274641       0.092955              0.039720             0.039720                         1045

## Benchmark Graphs
Generated: logs\figures\comprehensive_test_macro_f1.png, logs\figures\comprehensive_test_latency_mean_ms.png, logs\figures\comprehensive_robustness.png

## Confusion Matrices and Class-Wise Results
Class-wise metrics are in `logs/comprehensive_classwise.csv`. Confusion matrices are generated only for models with a saved deployment artifact.

## Robustness Channel Dropout, Clock Jitter, Missing, Noise, Window Size
               model       condition  accuracy  macro_f1  requested_window_size
histgradientboosting channel_dropout     0.888  0.709439                    NaN
histgradientboosting    clock_jitter     0.894  0.714377                    NaN
histgradientboosting         missing     0.874  0.665756                    NaN
histgradientboosting           noise     0.886  0.685104                    NaN
histgradientboosting window_size_100     0.886  0.689811                  100.0
histgradientboosting window_size_128     0.906  0.751337                  128.0
histgradientboosting window_size_156     0.884  0.692325                  156.0
histgradientboosting window_size_200     0.892  0.709650                  200.0
            lightgbm channel_dropout     0.894  0.740036                    NaN
            lightgbm    clock_jitter     0.914  0.773777                    NaN
            lightgbm         missing     0.906  0.752490                    NaN
            lightgbm           noise     0.908  0.782585                    NaN
            lightgbm window_size_100     0.876  0.685516                  100.0
            lightgbm window_size_128     0.912  0.761320                  128.0
            lightgbm window_size_156     0.908  0.762847                  156.0
            lightgbm window_size_200     0.912  0.794647                  200.0
       random_forest channel_dropout     0.894  0.708391                    NaN
       random_forest    clock_jitter     0.898  0.713452                    NaN
       random_forest         missing     0.892  0.671651                    NaN
       random_forest           noise     0.894  0.681377                    NaN
       random_forest window_size_100     0.884  0.664424                  100.0
       random_forest window_size_128     0.894  0.674898                  128.0
       random_forest window_size_156     0.892  0.693180                  156.0
       random_forest window_size_200     0.894  0.698996                  200.0
             xgboost channel_dropout     0.902  0.775478                    NaN
             xgboost    clock_jitter     0.900  0.731949                    NaN
             xgboost         missing     0.892  0.704397                    NaN
             xgboost           noise     0.896  0.711774                    NaN
             xgboost window_size_100     0.884  0.677813                  100.0
             xgboost window_size_128     0.892  0.678473                  128.0
             xgboost window_size_156     0.896  0.700989                  156.0
             xgboost window_size_200     0.900  0.739401                  200.0
 logistic_regression channel_dropout     0.626  0.472827                    NaN
 logistic_regression    clock_jitter     0.740  0.497547                    NaN
 logistic_regression         missing     0.590  0.410992                    NaN
 logistic_regression           noise     0.736  0.497067                    NaN
 logistic_regression window_size_100     0.698  0.477001                  100.0
 logistic_regression window_size_128     0.746  0.513436                  128.0
 logistic_regression window_size_156     0.738  0.491597                  156.0
 logistic_regression window_size_200     0.738  0.498746                  200.0
          linear_svm channel_dropout     0.634  0.476347                    NaN
          linear_svm    clock_jitter     0.718  0.462521                    NaN
          linear_svm         missing     0.580  0.388203                    NaN
          linear_svm           noise     0.718  0.469939                    NaN
          linear_svm window_size_100     0.684  0.457181                  100.0
          linear_svm window_size_128     0.748  0.511764                  128.0
          linear_svm window_size_156     0.726  0.465285                  156.0
          linear_svm window_size_200     0.732  0.484874                  200.0
         mlp_tabular channel_dropout     0.866  0.605463                    NaN
         mlp_tabular    clock_jitter     0.874  0.646653                    NaN
         mlp_tabular         missing     0.850  0.563986                    NaN
         mlp_tabular           noise     0.866  0.606369                    NaN
         mlp_tabular window_size_100     0.832  0.547597                  100.0
         mlp_tabular window_size_128     0.864  0.634896                  128.0
         mlp_tabular window_size_156     0.866  0.604350                  156.0
         mlp_tabular window_size_200     0.846  0.584994                  200.0
    conv_transformer channel_dropout     0.884  0.617847                    NaN
    conv_transformer    clock_jitter     0.882  0.615955                    NaN
    conv_transformer         missing     0.880  0.599055                    NaN
    conv_transformer           noise     0.884  0.617847                    NaN
    conv_transformer window_size_100     0.858  0.557436                  100.0
    conv_transformer window_size_128     0.870  0.597719                  128.0
    conv_transformer window_size_156     0.882  0.615955                  156.0
    conv_transformer window_size_200     0.878  0.641771                  200.0
                 gru channel_dropout     0.868  0.614581                    NaN
                 gru    clock_jitter     0.862  0.606424                    NaN
                 gru         missing     0.862  0.591493                    NaN
                 gru           noise     0.868  0.611039                    NaN
                 gru window_size_100     0.834  0.520487                  100.0
                 gru window_size_128     0.858  0.584656                  128.0
                 gru window_size_156     0.868  0.614581                  156.0
                 gru window_size_200     0.858  0.598672                  200.0
                lstm channel_dropout     0.852  0.511256                    NaN
                lstm    clock_jitter     0.850  0.498828                    NaN
                lstm         missing     0.850  0.519815                    NaN
                lstm           noise     0.848  0.508115                    NaN
                lstm window_size_100     0.822  0.401332                  100.0
                lstm window_size_128     0.832  0.436941                  128.0
                lstm window_size_156     0.850  0.510092                  156.0
                lstm window_size_200     0.840  0.502271                  200.0
           fusformer channel_dropout     0.884  0.616457                    NaN
           fusformer    clock_jitter     0.884  0.616457                    NaN
           fusformer         missing     0.884  0.616457                    NaN
           fusformer           noise     0.884  0.617847                    NaN
           fusformer window_size_100     0.872  0.573780                  100.0
           fusformer window_size_128     0.880  0.606890                  128.0
           fusformer window_size_156     0.884  0.616457                  156.0
           fusformer window_size_200     0.892  0.665862                  200.0
                mega channel_dropout     0.892  0.667909                    NaN
                mega    clock_jitter     0.884  0.629472                    NaN
                mega         missing     0.884  0.625633                    NaN
                mega           noise     0.884  0.629472                    NaN
                mega window_size_100     0.872  0.616469                  100.0
                mega window_size_128     0.884  0.634359                  128.0
                mega window_size_156     0.886  0.631500                  156.0
                mega window_size_200     0.888  0.654807                  200.0
         convnet_har channel_dropout     0.876  0.668463                    NaN
         convnet_har    clock_jitter     0.884  0.685233                    NaN
         convnet_har         missing     0.856  0.646017                    NaN
         convnet_har           noise     0.872  0.671015                    NaN
         convnet_har window_size_100     0.862  0.645935                  100.0
         convnet_har window_size_128     0.874  0.655733                  128.0
         convnet_har window_size_156     0.884  0.685233                  156.0
         convnet_har window_size_200     0.884  0.702105                  200.0
            tcnca_v3 channel_dropout     0.730  0.486126                    NaN
            tcnca_v3    clock_jitter     0.722  0.487078                    NaN
            tcnca_v3         missing     0.724  0.451266                    NaN
            tcnca_v3           noise     0.718  0.458389                    NaN
            tcnca_v3 window_size_100     0.706  0.466365                  100.0
            tcnca_v3 window_size_128     0.750  0.500197                  128.0
            tcnca_v3 window_size_156     0.738  0.483189                  156.0
            tcnca_v3 window_size_200     0.752  0.513141                  200.0
             rbf_svm channel_dropout     0.400  0.430355                    NaN
             rbf_svm    clock_jitter     0.430  0.448559                    NaN
             rbf_svm         missing     0.388  0.426115                    NaN
             rbf_svm           noise     0.410  0.431171                    NaN
             rbf_svm window_size_100     0.392  0.421229                  100.0
             rbf_svm window_size_128     0.416  0.439375                  128.0
             rbf_svm window_size_156     0.410  0.433344                  156.0
             rbf_svm window_size_200     0.410  0.432244                  200.0
            tcnca_v2 channel_dropout     0.770  0.495802                    NaN
            tcnca_v2    clock_jitter     0.800  0.561993                    NaN
            tcnca_v2         missing     0.760  0.475231                    NaN
            tcnca_v2           noise     0.772  0.492330                    NaN
            tcnca_v2 window_size_100     0.766  0.525564                  100.0
            tcnca_v2 window_size_128     0.816  0.551792                  128.0
            tcnca_v2 window_size_156     0.774  0.492962                  156.0
            tcnca_v2 window_size_200     0.794  0.544345                  200.0
            tcnca_v1 channel_dropout     0.884  0.633120                    NaN
            tcnca_v1    clock_jitter     0.884  0.627117                    NaN
            tcnca_v1         missing     0.878  0.593241                    NaN
            tcnca_v1           noise     0.882  0.625091                    NaN
            tcnca_v1 window_size_100     0.860  0.581233                  100.0
            tcnca_v1 window_size_128     0.870  0.606969                  128.0
            tcnca_v1 window_size_156     0.884  0.627117                  156.0
            tcnca_v1 window_size_200     0.884  0.627688                  200.0
                 tst channel_dropout     0.856  0.405818                    NaN
                 tst    clock_jitter     0.856  0.405818                    NaN
                 tst         missing     0.856  0.405818                    NaN
                 tst           noise     0.854  0.403701                    NaN
                 tst window_size_100     0.856  0.424322                  100.0
                 tst window_size_128     0.858  0.427521                  128.0
                 tst window_size_156     0.856  0.405818                  156.0
                 tst window_size_200     0.862  0.439548                  200.0
                 s4d channel_dropout     0.406  0.245259                    NaN
                 s4d    clock_jitter     0.390  0.224256                    NaN
                 s4d         missing     0.374  0.238814                    NaN
                 s4d           noise     0.416  0.273865                    NaN
                 s4d window_size_100     0.386  0.263854                  100.0
                 s4d window_size_128     0.388  0.245638                  128.0
                 s4d window_size_156     0.396  0.247981                  156.0
                 s4d window_size_200     0.394  0.242778                  200.0
             s5_onnx channel_dropout     0.406  0.245259                    NaN
             s5_onnx    clock_jitter     0.390  0.224256                    NaN
             s5_onnx         missing     0.376  0.238917                    NaN
             s5_onnx           noise     0.400  0.250949                    NaN
             s5_onnx window_size_100     0.386  0.263854                  100.0
             s5_onnx window_size_128     0.388  0.245638                  128.0
             s5_onnx window_size_156     0.396  0.247981                  156.0
             s5_onnx window_size_200     0.394  0.242778                  200.0
                 lru channel_dropout     0.406  0.245259                    NaN
                 lru    clock_jitter     0.390  0.224256                    NaN
                 lru         missing     0.370  0.247954                    NaN
                 lru           noise     0.388  0.238788                    NaN
                 lru window_size_100     0.386  0.263854                  100.0
                 lru window_size_128     0.388  0.245638                  128.0
                 lru window_size_156     0.396  0.247981                  156.0
                 lru window_size_200     0.394  0.242778                  200.0
              sgconv channel_dropout     0.168  0.213257                    NaN
              sgconv    clock_jitter     0.166  0.208725                    NaN
              sgconv         missing     0.166  0.214263                    NaN
              sgconv           noise     0.166  0.210176                    NaN
              sgconv window_size_100     0.162  0.206919                  100.0
              sgconv window_size_128     0.166  0.210070                  128.0
              sgconv window_size_156     0.166  0.208725                  156.0
              sgconv window_size_200     0.160  0.204531                  200.0
            resnet1d channel_dropout     0.582  0.148280                    NaN
            resnet1d    clock_jitter     0.586  0.148920                    NaN
            resnet1d         missing     0.642  0.156777                    NaN
            resnet1d           noise     0.586  0.148920                    NaN
            resnet1d window_size_100     0.596  0.150886                  100.0
            resnet1d window_size_128     0.594  0.150380                  128.0
            resnet1d window_size_156     0.586  0.148920                  156.0
            resnet1d window_size_200     0.602  0.151447                  200.0

## Final Comparison and Recommendation
Validation-selected recommendation: **lightgbm**. Test metrics are reported only for freshly trained workflow artifacts; unavailable architectures are not replaced by legacy artifacts and have no claimed test metrics.