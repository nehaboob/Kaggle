	lgb_params = {
            'learning_rate': 0.03,
            'n_estimators': 1000,
            'num_leaves': 64,
            'subsample': 0.2217,
            'colsample_bytree': 0.6810,
            'min_split_gain': np.power(10.0, -4.9380),
            'reg_alpha': np.power(10.0, -3.2454),
            'reg_lambda': np.power(10.0, -4.8571),
            'min_child_weight': np.power(10.0, 2),
            'silent': True }

	params_clf = {
	        'n_estimators': 1000,
		    "max_bin": 512,
		    "learning_rate": 0.02,
		    "boosting_type": "gbdt",
		    "objective": "binary",
		    "metric": "binary_logloss",
		    "num_leaves": 10,
		    "min_data": 100,
		    "boost_from_average": True
		}


1. results

[{'loss': 0.32102009787299357, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.32960751629908513, 'learning_rate': 0.14128507590354147, 'min_child_weight': 30, 'min_split_gain': 0.19194075068070393, 'n_estimators': 1600, 'num_leaves': 70, 'reg_alpha': 0.5774258307732933, 'reg_lambda': 0.055973041573782835, 'subsample': 1.0}, 'iteration': 17, 'train_time': 7.686584781004058, 'status': 'ok'}, 
 {'loss': 0.3211014653483526, 'params': {'boosting_type': 'gbdt', 'colsample_bytree': 0.950571641661073, 'learning_rate': 0.04496627341545186, 'min_child_weight': 85, 'min_split_gain': 0.3506869275383543, 'n_estimators': 1300, 'num_leaves': 88, 'reg_alpha': 0.3577154252478001, 'reg_lambda': 0.5739525832016432, 'subsample': 0.6497130141096576}, 'iteration': 106, 'train_time': 11.515355486997578, 'status': 'ok'}]

2. 
[{'loss': 0.3212220173723687, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.48439915100002195, 'learning_rate': 0.030422200909737244, 'min_child_weight': 60, 'min_split_gain': 0.486490134919048, 'n_estimators': 1300, 'num_leaves': 38, 'reg_alpha': 0.05867750540078395, 'reg_lambda': 0.6923497326649881, 'subsample_for_bin': 120000.0, 'subsample_freq': 45, 'subsample': 0.7456371561843139}, 'iteration': 40, 'train_time': 99.5313191249952, 'status': 'ok'}, 
{'loss': 0.3212269879525816, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.9194609357223081, 'learning_rate': 0.04160769283913887, 'min_child_weight': 50, 'min_split_gain': 0.24338233284916647, 'n_estimators': 500, 'num_leaves': 105, 'reg_alpha': 0.5041934836439877, 'reg_lambda': 0.06069006816725053, 'subsample_for_bin': 80000.0, 'subsample_freq': 90, 'subsample': 1.0}, 'iteration': 24, 'train_time': 9.575309501997253, 'status': 'ok'}]


3. with new columns

[{'loss': 0.3213236712368899, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.9674138437780953, 'learning_rate': 0.09166145526761348, 'min_child_weight': 45, 'min_split_gain': 0.36788956549524854, 'n_estimators': 2000, 'num_leaves': 103, 'reg_alpha': 0.4532630141999002, 'reg_lambda': 0.6127492034871995, 'subsample_for_bin': 260000.0, 'subsample_freq': 20, 'subsample': 0.7168841834693274}, 'iteration': 21, 'train_time': 33.44375436200062, 'status': 'ok'}, {'loss': 0.32137299458054763, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.9777368653884159, 'learning_rate': 0.09484371178896216, 'min_child_weight': 50, 'min_split_gain': 0.3462054390848908, 'n_estimators': 1000, 'num_leaves': 101, 'reg_alpha': 0.43839366253427203, 'reg_lambda': 0.6434235048679875, 'subsample_for_bin': 260000.0, 'subsample_freq': 20, 'subsample': 0.7388870272863974}, 'iteration': 22, 'train_time': 40.973779141000705, 'status': 'ok'}]

4. remvoing less important columns
[{'loss': 0.32122201711184417, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.48439915100002195, 'learning_rate': 0.030422200909737244, 'min_child_weight': 60, 'min_split_gain': 0.486490134919048, 'n_estimators': 1300, 'num_leaves': 38, 'reg_alpha': 0.05867750540078395, 'reg_lambda': 0.6923497326649881, 'subsample_for_bin': 120000.0, 'subsample_freq': 45, 'subsample': 0.7456371561843139}, 'iteration': 40, 'train_time': 100.04122068399738, 'status': 'ok'}, {'loss': 0.3212269879525816, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.9194609357223081, 'learning_rate': 0.04160769283913887, 'min_child_weight': 50, 'min_split_gain': 0.24338233284916647, 'n_estimators': 500, 'num_leaves': 105, 'reg_alpha': 0.5041934836439877, 'reg_lambda': 0.06069006816725053, 'subsample_for_bin': 80000.0, 'subsample_freq': 90, 'subsample': 1.0}, 'iteration': 24, 'train_time': 9.846729944008985, 'status': 'ok'}]


5. with recent columns

[{'loss': 0.3210305318597825, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.845598132327906, 'learning_rate': 0.10953432710262367, 'min_child_weight': 50, 'min_split_gain': 0.134434391813417, 'n_estimators': 500, 'num_leaves': 40, 'reg_alpha': 0.8392080820384631, 'reg_lambda': 0.3725434212598154, 'subsample_for_bin': 220000.0, 'subsample': 0.7091915385738168}, 'iteration': 33, 'train_time': 29.36616769900138, 'status': 'ok'}, 
{'loss': 0.3210633024224764, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.4710970164512299, 'learning_rate': 0.031642623234827805, 'min_child_weight': 165, 'min_split_gain': 0.4251945969203937, 'n_estimators': 600, 'num_leaves': 138, 'reg_alpha': 0.5005519711106031, 'reg_lambda': 0.45127104117580963, 'subsample_for_bin': 260000.0, 'subsample': 0.3738991074449508}, 'iteration': 2, 'train_time': 128.8927383600094, 'status': 'ok'}]

5 with out less imp columns
[{'loss': 0.32126104318480525, 'params': {'boosting_type': 'gbdt', 'colsample_bytree': 0.9675674948719577, 'learning_rate': 0.0354824873463615, 'min_child_weight': 175, 'min_split_gain': 0.5683746495081688, 'n_estimators': 1100, 'num_leaves': 100, 'reg_alpha': 0.8609332600487669, 'reg_lambda': 0.24548591032980271, 'subsample': 0.498787829990324}, 'iteration': 26, 'train_time': 10.131754011992598, 'status': 'ok'}, 
{'loss': 0.32133525317859674, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.6184187096333772, 'learning_rate': 0.052832213875717564, 'min_child_weight': 170, 'min_split_gain': 0.6103179124473367, 'n_estimators': 1700, 'num_leaves': 74, 'reg_alpha': 0.7407787235284571, 'reg_lambda': 0.11035398441869984, 'subsample': 0.4315192653214378}, 'iteration': 40, 'train_time': 46.94639874299173, 'status': 'ok'}]


tune all with all columns

######
20160801
#######
[{'loss': 0.35480471384176465, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.38906808235361906, 'learning_rate': 0.0986491306631427, 'min_child_weight': 140, 'min_split_gain': 0.9272258002922322, 'n_estimators': 1900, 'num_leaves': 101, 'reg_alpha': 0.8773394768276057, 'reg_lambda': 0.2659905184062262, 'subsample_for_bin': 120000.0, 'subsample': 0.9959970933407915}, 'iteration': 26, 'train_time': 296.038039947, 'status': 'ok'}, 
{'loss': 0.355063363984611, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.4219856803437042, 'learning_rate': 0.05872794311423726, 'min_child_weight': 35, 'min_split_gain': 0.1874254087583043, 'n_estimators': 1800, 'num_leaves': 34, 'reg_alpha': 0.787416292126707, 'reg_lambda': 0.5521275770632086, 'subsample_for_bin': 220000.0, 'subsample': 1.0}, 'iteration': 10, 'train_time': 12.617700743000114, 'status': 'ok'}]

20160901
[{'loss': 0.355402094519487, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.38906808235361906, 'learning_rate': 0.0986491306631427, 'min_child_weight': 140, 'min_split_gain': 0.9272258002922322, 'n_estimators': 1900, 'num_leaves': 101, 'reg_alpha': 0.8773394768276057, 'reg_lambda': 0.2659905184062262, 'subsample_for_bin': 120000.0, 'subsample': 0.9959970933407915}, 'iteration': 26, 'train_time': 332.5058747940002, 'status': 'ok'},
 {'loss': 0.35567225443374256, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.22992235638198413, 'learning_rate': 0.12254101704277647, 'min_child_weight': 145, 'min_split_gain': 0.9146168059274657, 'n_estimators': 1600, 'num_leaves': 121, 'reg_alpha': 0.5721757853934379, 'reg_lambda': 0.307140101332562, 'subsample_for_bin': 120000.0, 'subsample': 0.8380398104017898}, 'iteration': 29, 'train_time': 161.352761225, 'status': 'ok'}]

20161001
[{'loss': 0.348106871625851, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.7593999930761446, 'learning_rate': 0.113297983315749, 'min_child_weight': 130, 'min_split_gain': 0.9502933004096531, 'n_estimators': 2000, 'num_leaves': 117, 'reg_alpha': 0.010697367051874905, 'reg_lambda': 0.7063958644366212, 'subsample_for_bin': 120000.0, 'subsample': 0.8954542796528193}, 'iteration': 16, 'train_time': 98.72506623799927, 'status': 'ok'}, 
{'loss': 0.3484355073540583, 'params': {'boosting_type': 'gbdt', 'colsample_bytree': 0.8356448167248348, 'learning_rate': 0.09442470801660842, 'min_child_weight': 130, 'min_split_gain': 0.8887636824404529, 'n_estimators': 2000, 'num_leaves': 96, 'reg_alpha': 0.10073760678825669, 'reg_lambda': 0.16512472969319203, 'subsample_for_bin': 140000.0, 'subsample': 0.9905141354778699}, 'iteration': 24, 'train_time': 12.506759676998627, 'status': 'ok'}]

20161101

[{'loss': 0.3254708935195675, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.47279793426586203, 'learning_rate': 0.08820875723920915, 'min_child_weight': 30, 'min_split_gain': 0.7444197632763998, 'n_estimators': 1000, 'num_leaves': 66, 'reg_alpha': 0.34803313714432865, 'reg_lambda': 0.779293013881024, 'subsample_for_bin': 100000.0, 'subsample': 1.0}, 'iteration': 8, 'train_time': 9.09294577000037, 'status': 'ok'}, 
{'loss': 0.32547794999267193, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.4219856803437042, 'learning_rate': 0.05872794311423726, 'min_child_weight': 35, 'min_split_gain': 0.1874254087583043, 'n_estimators': 180

20161201
[{'loss': 0.3958202962207971, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.7593999930761446, 'learning_rate': 0.113297983315749, 'min_child_weight': 130, 'min_split_gain': 0.9502933004096531, 'n_estimators': 2000, 'num_leaves': 117, 'reg_alpha': 0.010697367051874905, 'reg_lambda': 0.7063958644366212, 'subsample_for_bin': 120000.0, 'subsample': 0.8954542796528193}, 'iteration': 16, 'train_time': 78.21927871499793, 'status': 'ok'}, 
{'loss': 0.39584887419973896, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.5885884181036832, 'learning_rate': 0.05349824970384282, 'min_child_weight': 45, 'min_split_gain': 0.1281611540990728, 'n_estimators': 1900, 'num_leaves': 79, 'reg_alpha': 0.09325735572996519, 'reg_lambda': 0.6664606685493126, 'subsample_for_bin': 180000.0, 'subsample': 0.9087974554962668}, 'iteration': 28, 'train_time': 45.47517937000157, 'status': 'ok'}]

20170101
[{'loss': 0.41640351785129737, 'params': {'boosting_type': 'gbdt', 'colsample_bytree': 0.29752728330965655, 'learning_rate': 0.023351371456881882, 'min_child_weight': 75, 'min_split_gain': 0.9232038075218142, 'n_estimators': 1300, 'num_leaves': 125, 'reg_alpha': 0.9457444660601183, 'reg_lambda': 0.2929723397227686, 'subsample_for_bin': 60000.0, 'subsample': 0.6463802645169471}, 'iteration': 30, 'train_time': 8.555438126997615, 'status': 'ok'}, 
{'loss': 0.41652084151293567, 'params': {'boosting_type': 'gbdt', 'colsample_bytree': 0.3275476460271277, 'learning_rate': 0.017147598530481763, 'min_child_weight': 65, 'min_split_gain': 0.34631141265087784, 'n_estimators': 1400, 'num_leaves': 147, 'reg_alpha': 0.9686153493563557, 'reg_lambda': 0.2091392723732014, 'subsample_for_bin': 80000.0, 'subsample': 0.7472969426747409}, 'iteration': 24, 'train_time': 10.544272009996348, 'status': 'ok'}]

20170201
[{'loss': 0.31481413211516296, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.6362766173441813, 'learning_rate': 0.013376483790442711, 'min_child_weight': 115, 'min_split_gain': 0.9924238880823565, 'n_estimators': 1800, 'num_leaves': 45, 'reg_alpha': 0.28795596659285755, 'reg_lambda': 0.017184862664919285, 'subsample_for_bin': 240000.0, 'subsample': 1.0}, 'iteration': 12, 'train_time': 11.8065770010071, 'status': 'ok'}, 
{'loss': 0.31490684231156674, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.8298901965345191, 'learning_rate': 0.014187725618348735, 'min_child_weight': 110, 'min_split_gain': 0.3416908200991302, 'n_estimators': 700, 'num_leaves': 81, 'reg_alpha': 0.5439376383063149, 'reg_lambda': 0.3206070549051747, 'subsample_for_bin': 300000.0, 'subsample': 1.0}, 'iteration': 25, 'train_time': 13.654469988003257, 'status': 'ok'}]

20170301
[{'loss': 0.30457551886711276, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.6998864796582646, 'learning_rate': 0.06554053682012086, 'min_child_weight': 20, 'min_split_gain': 0.6466457209384664, 'n_estimators': 800, 'num_leaves': 93, 'reg_alpha': 0.6814237430096464, 'reg_lambda': 0.6337919424510632, 'subsample_for_bin': 140000.0, 'subsample': 1.0}, 'iteration': 29, 'train_time': 8.737943669999368, 'status': 'ok'}, 
{'loss': 0.30465770296441674, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.7045887377627815, 'learning_rate': 0.0669997267598685, 'min_child_weight': 20, 'min_split_gain': 0.6122041711732973, 'n_estimators': 500, 'num_leaves': 95, 'reg_alpha': 0.535475356652016, 'reg_lambda': 0.315187861690093, 'subsample_for_bin': 40000.0, 'subsample': 1.0}, 'iteration': 26, 'train_time': 7.989077292993898, 'status': 'ok'}]

20170401
[{'loss': 0.313132911202661, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.4219856803437042, 'learning_rate': 0.05872794311423726, 'min_child_weight': 35, 'min_split_gain': 0.1874254087583043, 'n_estimators': 1800, 'num_leaves': 34, 'reg_alpha': 0.787416292126707, 'reg_lambda': 0.5521275770632086, 'subsample_for_bin': 220000.0, 'subsample': 1.0}, 'iteration': 10, 'train_time': 9.296649975000037, 'status': 'ok'}, 
{'loss': 0.3131841253354818, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.6362766173441813, 'learning_rate': 0.013376483790442711, 'min_child_weight': 115, 'min_split_gain': 0.9924238880823565, 'n_estimators': 1800, 'num_leaves': 45, 'reg_alpha': 0.28795596659285755, 'reg_lambda': 0.017184862664919285, 'subsample_for_bin': 240000.0, 'subsample': 1.0}, 'iteration': 12, 'train_time': 16.411307351000005, 'status': 'ok'}]

20170501
[{'loss': 0.3210633024224764, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.4710970164512299, 'learning_rate': 0.031642623234827805, 'min_child_weight': 165, 'min_split_gain': 0.4251945969203937, 'n_estimators': 600, 'num_leaves': 138, 'reg_alpha': 0.5005519711106031, 'reg_lambda': 0.45127104117580963, 'subsample_for_bin': 260000.0, 'subsample': 0.3738991074449508}, 'iteration': 2, 'train_time': 126.3899686950001, 'status': 'ok'}, 
{'loss': 0.32112015170928215, 'params': {'boosting_type': 'gbdt', 'colsample_bytree': 0.48524550809843103, 'learning_rate': 0.15956582765442182, 'min_child_weight': 180, 'min_split_gain': 0.5717653294444226, 'n_estimators': 600, 'num_leaves': 61, 'reg_alpha': 0.9748327622464755, 'reg_lambda': 0.2339154313198487, 'subsample_for_bin': 40000.0, 'subsample': 0.757884693108577}, 'iteration': 13, 'train_time': 7.289694731000054, 'status': 'ok'}]

20170601
[{'loss': 0.31878948466361456, 'params': {'boosting_type': 'gbdt', 'colsample_bytree': 0.9597211908291912, 'learning_rate': 0.13851059302284938, 'min_child_weight': 100, 'min_split_gain': 0.21775367540850343, 'n_estimators': 1100, 'num_leaves': 102, 'reg_alpha': 0.19172431700514758, 'reg_lambda': 0.6897250931579948, 'subsample_for_bin': 200000.0, 'subsample': 0.3916684297756816}, 'iteration': 14, 'train_time': 9.781408103000103, 'status': 'ok'}, 
{'loss': 0.31894966540238157, 'params': {'boosting_type': 'dart', 'colsample_bytree': 0.8599470824550043, 'learning_rate': 0.18176278032562698, 'min_child_weight': 175, 'min_split_gain': 0.4637526828496205, 'n_estimators': 1900, 'num_leaves': 70, 'reg_alpha': 0.8695230563444872, 'reg_lambda': 0.44082259662739115, 'subsample_for_bin': 160000.0, 'subsample': 0.789736348878856}, 'iteration': 9, 'train_time': 23.266933489000166, 'status': 'ok'}]

20170701
[{'loss': 0.3305498698847994, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.6362766173441813, 'learning_rate': 0.013376483790442711, 'min_child_weight': 115, 'min_split_gain': 0.9924238880823565, 'n_estimators': 1800, 'num_leaves': 45, 'reg_alpha': 0.28795596659285755, 'reg_lambda': 0.017184862664919285, 'subsample_for_bin': 240000.0, 'subsample': 1.0}, 'iteration': 12, 'train_time': 12.943018230000234, 'status': 'ok'}, 
{'loss': 0.3305869968710812, 'params': {'boosting_type': 'goss', 'colsample_bytree': 0.4219856803437042, 'learning_rate': 0.05872794311423726, 'min_child_weight': 35, 'min_split_gain': 0.1874254087583043, 'n_estimators': 1800, 'num_leaves': 34, 'reg_alpha': 0.787416292126707, 'reg_lambda': 0.5521275770632086, 'subsample_for_bin': 220000.0, 'subsample': 1.0}, 'iteration': 10, 'train_time': 10.434055661000457, 'status': 'ok'}]

20170801
[{'loss': 0.31776943169448885, 'params': {'boosting_type': 'gbdt', 'colsample_bytree': 0.7010272783085567, 'learning_rate': 0.04094919304252701, 'min_child_weight': 150, 'min_split_gain': 0.331242596811015, 'n_estimators': 1400, 'num_leaves': 150, 'reg_alpha': 0.08056103718787506, 'reg_lambda': 0.8603239608548007, 'subsample_for_bin': 80000.0, 'subsample': 0.5549475251562849}, 'iteration': 24, 'train_time': 10.302412173999983, 'status': 'ok'}, 
{'loss': 0.3177952773115766, 'params': {'boosting_type': 'gbdt', 'colsample_bytree': 0.2732695028583563, 'learning_rate': 0.047089838395288325, 'min_child_weight': 160, 'min_split_gain': 0.7966439678474594, 'n_estimators': 1600, 'num_leaves': 112, 'reg_alpha': 0.0025361424319769927, 'reg_lambda': 0.09469342010644329, 'subsample_for_bin': 180000.0, 'subsample': 0.7265238319186288}, 'iteration': 19, 'train_time': 8.684013563999542, 'status': 'ok'}]
