import os 
import json 
import numpy as np 

def compute_mean_res(dest_json_path, json_folder):
    mean_json_dict = {}
    json_files = os.listdir(json_folder)
    for json_name in json_files:
        json_path = os.path.join(json_folder, json_name)

        json_data = json.load(open(json_path, 'r'))
        for k in json_data:
            if k not in mean_json_dict:
                mean_json_dict[k] = []
            mean_json_dict[k].append(json_data[k])

    res_json_dict = {}
    for k in mean_json_dict:
        res_json_dict[k] = np.asarray(mean_json_dict[k]).mean()

    json.dump(res_json_dict, open(dest_json_path, 'w'))

def get_mean_val_seen_object_wo_guidance():
    data_dict = {}
    data_dict['frl_apartment_0'] = {}
    # data_dict['frl_apartment_1'] = {}
    data_dict['frl_apartment_4'] = {} 

    # Seen Object, wo guidance 
    data_dict['frl_apartment_0']['start_err'] = 15.29629643925506
    data_dict['frl_apartment_0']['end_err'] = 68.55762112831724
    data_dict['frl_apartment_0']['waypoints_err'] = 49.02417489871886
    data_dict['frl_apartment_0']['feet_height'] = 0.059646330773830414
    data_dict['frl_apartment_0']['fs'] = 4.288153506264075
    data_dict['frl_apartment_0']['contact_percent'] = 0.4477964342636866
    data_dict['frl_apartment_0']['scene_human_penetration'] = 0.0019478552276268601 
    data_dict['frl_apartment_0']['scene_object_penetrtion'] = 0.0005494547076523304
    data_dict['frl_apartment_0']['hand_object_penetrtion'] = 0.007182971574366093
    data_dict['frl_apartment_0']['num_seq'] = 69

    # data_dict['frl_apartment_1']['start_err'] = 14.102032398483876
    # data_dict['frl_apartment_1']['end_err'] = 71.82682205376953
    # data_dict['frl_apartment_1']['waypoints_err'] = 51.45700106975333
    # data_dict['frl_apartment_1']['feet_height'] = 0.06286376714706421
    # data_dict['frl_apartment_1']['fs'] = 4.635441794992669
    # data_dict['frl_apartment_1']['contact_percent'] = 0.46073545020206985
    # data_dict['frl_apartment_1']['scene_human_penetration'] = 0.02412508614361286
    # data_dict['frl_apartment_1']['scene_object_penetrtion'] = 0.025187702849507332
    # data_dict['frl_apartment_1']['hand_object_penetrtion'] = 0.004928759299218655
    # data_dict['frl_apartment_1']['num_seq'] = 145

    data_dict['frl_apartment_4']['start_err'] = 14.862098886320988
    data_dict['frl_apartment_4']['end_err'] = 74.37027158448473
    data_dict['frl_apartment_4']['waypoints_err'] = 52.42606703864618
    data_dict['frl_apartment_4']['feet_height'] = 0.06043344363570213
    data_dict['frl_apartment_4']['fs'] = 4.326536357777564
    data_dict['frl_apartment_4']['contact_percent'] = 0.5146364083072843
    data_dict['frl_apartment_4']['scene_human_penetration'] = 0.005959006492048502
    data_dict['frl_apartment_4']['scene_object_penetrtion'] = 0.0015842542052268982
    data_dict['frl_apartment_4']['hand_object_penetrtion'] = 0.006828381214290857
    data_dict['frl_apartment_4']['num_seq'] = 96

    k_metric = list(data_dict['frl_apartment_0'].keys())
    final_res_dict = {} 
    for k in k_metric:
        if "num_seq" not in k:
            total_seq_num = 0
            total_val = 0
            for scene_name in data_dict:
                curr_scene_val = data_dict[scene_name][k] * data_dict[scene_name]['num_seq']
                total_val += curr_scene_val
                total_seq_num += data_dict[scene_name]['num_seq']
            final_res_dict[k] = total_val/total_seq_num 

    print("Seen Objects, wo guidance res:{0}".format(final_res_dict))

def get_mean_val_seen_object_w_guidance():
    data_dict = {}
    data_dict['frl_apartment_0'] = {}
    # data_dict['frl_apartment_1'] = {}
    data_dict['frl_apartment_4'] = {} 

    # Seen Object, wo guidance 
    data_dict['frl_apartment_0']['start_err'] = 19.07439612230097
    data_dict['frl_apartment_0']['end_err'] = 108.22472439242014
    data_dict['frl_apartment_0']['waypoints_err'] = 54.78605870067498
    data_dict['frl_apartment_0']['feet_height'] = 0.041794225573539734
    data_dict['frl_apartment_0']['fs'] = 4.585912914668201
    data_dict['frl_apartment_0']['contact_percent'] = 0.6422291411838837
    data_dict['frl_apartment_0']['scene_human_penetration'] = 0.00016671649063937366
    data_dict['frl_apartment_0']['scene_object_penetrtion'] = 0.0001226048480020836
    data_dict['frl_apartment_0']['hand_object_penetrtion'] = 0.007086490280926228
    data_dict['frl_apartment_0']['num_seq'] = 69

    # data_dict['frl_apartment_1']['start_err'] = 50.04908258801904
    # data_dict['frl_apartment_1']['end_err'] = 136.241734930282
    # data_dict['frl_apartment_1']['waypoints_err'] = 82.72720701692775
    # data_dict['frl_apartment_1']['feet_height'] = 0.04667331278324127
    # data_dict['frl_apartment_1']['fs'] = 4.327120402846643
    # data_dict['frl_apartment_1']['contact_percent'] = 0.5293454792211737
    # data_dict['frl_apartment_1']['scene_human_penetration'] = 0.004587466828525066
    # data_dict['frl_apartment_1']['scene_object_penetrtion'] = 0.007837371900677681
    # data_dict['frl_apartment_1']['hand_object_penetrtion'] = 0.0058834124356508255
    # data_dict['frl_apartment_1']['num_seq'] = 145

    data_dict['frl_apartment_4']['start_err'] = 24.513748367705073
    data_dict['frl_apartment_4']['end_err'] = 93.13383511228797
    data_dict['frl_apartment_4']['waypoints_err'] = 59.10077339891965
    data_dict['frl_apartment_4']['feet_height'] = 0.04853618144989014
    data_dict['frl_apartment_4']['fs'] = 4.538569871042316
    data_dict['frl_apartment_4']['contact_percent'] = 0.6140807639480913
    data_dict['frl_apartment_4']['scene_human_penetration'] = 0.0008639117586426437
    data_dict['frl_apartment_4']['scene_object_penetrtion'] = 0.00027816349756903946
    data_dict['frl_apartment_4']['hand_object_penetrtion'] = 0.006773812230676413
    data_dict['frl_apartment_4']['num_seq'] = 96

    k_metric = list(data_dict['frl_apartment_0'].keys())
    final_res_dict = {} 
    for k in k_metric:
        if "num_seq" not in k:
            total_seq_num = 0
            total_val = 0
            for scene_name in data_dict:
                curr_scene_val = data_dict[scene_name][k] * data_dict[scene_name]['num_seq']
                total_val += curr_scene_val
                total_seq_num += data_dict[scene_name]['num_seq']
            final_res_dict[k] = total_val/total_seq_num 

    print("Seen Objects, w guidance:{0}".format(final_res_dict))

def get_mean_val_unseen_object_wo_guidance():
    data_dict = {}
    data_dict['frl_apartment_0'] = {}
    # data_dict['frl_apartment_1'] = {}
    data_dict['frl_apartment_4'] = {} 

    # Seen Object, wo guidance 
    data_dict['frl_apartment_0']['start_err'] = 57.70183842074364
    data_dict['frl_apartment_0']['end_err'] = 102.66199624760354
    data_dict['frl_apartment_0']['waypoints_err'] = 51.47129331603118
    data_dict['frl_apartment_0']['feet_height'] = 0.0380651094019413
    data_dict['frl_apartment_0']['fs'] = 3.9779628540636223
    data_dict['frl_apartment_0']['contact_percent'] = 0.49138869238722316
    data_dict['frl_apartment_0']['scene_human_penetration'] = 0.0010320765431970358
    data_dict['frl_apartment_0']['scene_object_penetrtion'] = 0.0016726396279409528
    data_dict['frl_apartment_0']['hand_object_penetrtion'] = 0.006217147223651409
    data_dict['frl_apartment_0']['num_seq'] = 47

    # data_dict['frl_apartment_1']['start_err'] = 54.3965610030752
    # data_dict['frl_apartment_1']['end_err'] = 105.06583544640587
    # data_dict['frl_apartment_1']['waypoints_err'] = 52.056684961112644
    # data_dict['frl_apartment_1']['feet_height'] = 0.053446173667907715
    # data_dict['frl_apartment_1']['fs'] = 4.351618779409874
    # data_dict['frl_apartment_1']['contact_percent'] = 0.42031298227206154
    # data_dict['frl_apartment_1']['scene_human_penetration'] = 0.017809459939599037
    # data_dict['frl_apartment_1']['scene_object_penetrtion'] = 0.02320903353393078
    # data_dict['frl_apartment_1']['hand_object_penetrtion'] = 0.00034235732164233923
    # data_dict['frl_apartment_1']['num_seq'] = 130

    data_dict['frl_apartment_4']['start_err'] = 65.21398326648134
    data_dict['frl_apartment_4']['end_err'] = 89.9273060439598
    data_dict['frl_apartment_4']['waypoints_err'] = 53.128296639474605
    data_dict['frl_apartment_4']['feet_height'] = 0.051962777972221375
    data_dict['frl_apartment_4']['fs'] = 3.8419656896023553
    data_dict['frl_apartment_4']['contact_percent'] = 0.38432225063938624
    data_dict['frl_apartment_4']['scene_human_penetration'] = 0.003579400945454836
    data_dict['frl_apartment_4']['scene_object_penetrtion'] = 0.0032552191987633705
    data_dict['frl_apartment_4']['hand_object_penetrtion'] = 0.00516451895236969
    data_dict['frl_apartment_4']['num_seq'] = 105

    k_metric = list(data_dict['frl_apartment_0'].keys())
    final_res_dict = {} 
    for k in k_metric:
        if "num_seq" not in k:
            total_seq_num = 0
            total_val = 0
            for scene_name in data_dict:
                curr_scene_val = data_dict[scene_name][k] * data_dict[scene_name]['num_seq']
                total_val += curr_scene_val
                total_seq_num += data_dict[scene_name]['num_seq']
            final_res_dict[k] = total_val/total_seq_num 

    print("Unseen Objects, wo guidance res:{0}".format(final_res_dict))

def get_mean_val_unseen_object_w_guidance():
    data_dict = {}
    data_dict['frl_apartment_0'] = {}
    # data_dict['frl_apartment_1'] = {}
    data_dict['frl_apartment_4'] = {} 

    # Seen Object, wo guidance 
    data_dict['frl_apartment_0']['start_err'] = 40.39251185121371
    data_dict['frl_apartment_0']['end_err'] = 127.94205938723493
    data_dict['frl_apartment_0']['waypoints_err'] = 58.979284170195996
    data_dict['frl_apartment_0']['feet_height'] = 0.03947695344686508
    data_dict['frl_apartment_0']['fs'] = 4.168092960378966
    data_dict['frl_apartment_0']['contact_percent'] = 0.7866926048865429
    data_dict['frl_apartment_0']['scene_human_penetration'] = 4.256399188307114e-05
    data_dict['frl_apartment_0']['scene_object_penetrtion'] = 0.00011889482266269624
    data_dict['frl_apartment_0']['hand_object_penetrtion'] = 0.0040849740616977215
    data_dict['frl_apartment_0']['num_seq'] = 47

    # data_dict['frl_apartment_1']['start_err'] = 52.798669040203094
    # data_dict['frl_apartment_1']['end_err'] = 170.4502450947005
    # data_dict['frl_apartment_1']['waypoints_err'] = 92.546347146615
    # data_dict['frl_apartment_1']['feet_height'] = 0.04720227047801018
    # data_dict['frl_apartment_1']['fs'] = 4.921383718997271
    # data_dict['frl_apartment_1']['contact_percent'] = 0.5678106323911951
    # data_dict['frl_apartment_1']['scene_human_penetration'] = 0.003026238875463605
    # data_dict['frl_apartment_1']['scene_object_penetrtion'] = 0.003447113558650017
    # data_dict['frl_apartment_1']['hand_object_penetrtion'] = 0.00041412474820390344
    # data_dict['frl_apartment_1']['num_seq'] = 130

    data_dict['frl_apartment_4']['start_err'] = 63.2698881462039
    data_dict['frl_apartment_4']['end_err'] = 117.62296213280587
    data_dict['frl_apartment_4']['waypoints_err'] = 59.66469725032174
    data_dict['frl_apartment_4']['feet_height'] = 0.04416295140981674
    data_dict['frl_apartment_4']['fs'] = 4.036559734742445
    data_dict['frl_apartment_4']['contact_percent'] = 0.5882645232005845
    data_dict['frl_apartment_4']['scene_human_penetration'] = 0.00010706844477681443
    data_dict['frl_apartment_4']['scene_object_penetrtion'] = 0.00024629608378745615
    data_dict['frl_apartment_4']['hand_object_penetrtion'] = 0.0027906633913517
    data_dict['frl_apartment_4']['num_seq'] = 105

    k_metric = list(data_dict['frl_apartment_0'].keys())
    final_res_dict = {} 
    for k in k_metric:
        if "num_seq" not in k:
            total_seq_num = 0
            total_val = 0
            for scene_name in data_dict:
                curr_scene_val = data_dict[scene_name][k] * data_dict[scene_name]['num_seq']
                total_val += curr_scene_val
                total_seq_num += data_dict[scene_name]['num_seq']
            final_res_dict[k] = total_val/total_seq_num 

    print("Unseen Objects, w guidance:{0}".format(final_res_dict))

if __name__ == "__main__":

    dest_json_path = "/viscam/u/jiamanli/github/scene_aware_manip/evaluation_metrics_json_unseen_obj/1_interaction_model_baseline/evaluation_metrics_for_all_test_data.json"
    json_folder = "/viscam/u/jiamanli/github/scene_aware_manip/evaluation_metrics_json_unseen_obj/1_interaction_model_baseline"
    # compute_mean_res(dest_json_path, json_folder) 

    dest_json_path = "/viscam/u/jiamanli/github/scene_aware_manip/evaluation_metrics_json_unseen_obj/1_interaction_model/evaluation_metrics_for_all_test_data.json"
    json_folder = "/viscam/u/jiamanli/github/scene_aware_manip/evaluation_metrics_json_unseen_obj/1_interaction_model"
    # compute_mean_res(dest_json_path, json_folder) 

    dest_json_path = "/viscam/u/jiamanli/github/scene_aware_manip/evaluation_metrics_json_unseen_obj/2_interaction_model_w_guidance/evaluation_metrics_for_all_test_data.json"
    json_folder = "/viscam/u/jiamanli/github/scene_aware_manip/evaluation_metrics_json_unseen_obj/2_interaction_model_w_guidance"
    # compute_mean_res(dest_json_path, json_folder) 

    # get_mean_val_seen_object_wo_guidance() 
    # get_mean_val_seen_object_w_guidance() 


    get_mean_val_unseen_object_wo_guidance()
    get_mean_val_unseen_object_w_guidance() 
